import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing
from timm.data.mixup import Mixup
# 移除 from timm.data import Cutout，因為新版 timm 中不存在此模塊
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.distributed import destroy_process_group
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import logging
import sys
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import math
from functools import partial
import signal
import traceback

# 設置 NCCL 超時環境變數
os.environ["NCCL_BLOCKING_WAIT"] = "1"  # 使用阻塞等待模式，提高穩定性
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"  # 避免使用回環或Docker介面
os.environ["NCCL_DEBUG"] = "INFO"  # 啟用NCCL調試信息
os.environ["NCCL_IB_TIMEOUT"] = "23"  # 增加 InfiniBand 超時時間
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # 啟用非同步錯誤處理
os.environ["NCCL_SOCKET_NTHREADS"] = "4"  # 設置NCCL Socket通訊的線程數
os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"  # 設置每個線程的Socket數量
os.environ["NCCL_TREE_THRESHOLD"] = "0"  # 強制使用樹算法，可能更穩定但較慢
# 設置NCCL超時時間（秒），遠高於默認的600秒
os.environ["NCCL_TIMEOUT"] = str(3600)  # 1小時

# 設置多進程啟動方法為 'spawn'，這可以解決某些連接問題
# 這必須在導入任何其他與 torch.multiprocessing 相關的模塊之前設置
try:
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Import the SwinV2 classifier
from swin_transformer_v2_classifier import swin_transformer_v2_base_classifier

# 設置全局超時處理機制
class NCCLTimeoutHandler:
    def __init__(self, timeout=1800):  # 預設超時時間為30分鐘
        self.timeout = timeout
        self.start_time = None
        self.active = False
        self.prev_handler = None
        self.local_rank = -1

    def set_rank(self, rank):
        self.local_rank = rank

    def start(self):
        if not self.active:
            self.active = True
            self.start_time = time.time()
            self.prev_handler = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.timeout)

    def _handler(self, signum, frame):
        elapsed = time.time() - self.start_time
        print(f"[Rank {self.local_rank}] Operation timeout after {elapsed:.2f} seconds. Terminating process.")
        # 主動終止進程，這比等待超時錯誤更乾淨
        os._exit(1)

    def stop(self):
        if self.active:
            signal.alarm(0)
            if self.prev_handler is not None:
                signal.signal(signal.SIGALRM, self.prev_handler)
            self.active = False

    def __del__(self):
        self.stop()

# 全局超時處理器
timeout_handler = NCCLTimeoutHandler()

# 添加檢查確保圖像尺寸與窗口尺寸相容
def check_img_size_compatibility(img_size, window_size):
    if img_size % window_size != 0:
        print(f"WARNING: img_size {img_size} is not divisible by window_size {window_size}")
        new_img_size = (img_size // window_size) * window_size
        print(f"Adjusting img_size to {new_img_size}")
        return new_img_size
    return img_size

# 設置日誌格式
def setup_logger(local_rank):
    # 創建日誌格式
    log_format = '%(asctime)s - %(levelname)s - Rank[%(rank)s] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 創建一個自定義的過濾器，添加rank信息
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank
            
        def filter(self, record):
            record.rank = self.rank
            return True
    
    # 建立logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清空任何現有的處理器
    
    # 添加過濾器
    rank_filter = RankFilter(local_rank)
    logger.addFilter(rank_filter)
    
    # 創建格式
    formatter = logging.Formatter(log_format, date_format)
    
    # 添加控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 只在主進程(rank 0)添加文件處理器，避免多個進程同時寫入文件
    if local_rank == 0:
        # 使用日期時間來命名日誌文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f'training_{timestamp}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 用於同步所有進程的輔助函數
def synchronize():
    """
    同步所有分散式進程
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

# Updated Label Encoder
class Label_encoder:
    def __init__(self, labels):
        self.labels = {label: idx for idx, label in enumerate(labels)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels.get(label)


# Custom Dataset for Image Classification
class Food101Dataset(Dataset):
    def __init__(self, dataframe, encoder, transform=None):
        self.dataframe = dataframe
        self.encoder = encoder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        label = self.dataframe.iloc[idx]['label']
        label = self.encoder.get_idx(label)

        # 增加更強大的錯誤處理和圖像讀取重試機制
        for attempt in range(3):  # 重試3次
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                if attempt < 2:  # 如果這不是最後一次重試
                    time.sleep(0.1)  # 短暫等待
                    continue
                else:
                    print(f"Error loading image {img_path}: {e}")
                    # 返回一個與訓練設置相符的隨機圖像而不是拋出異常
                    dummy_image = torch.randn(3, 224, 224)
                    # 歸一化隨機張量使其統計特性與實際圖像更接近
                    dummy_image = (dummy_image - dummy_image.mean()) / dummy_image.std()
                    return dummy_image, label


# Updated prepare_dataframe function
def prepare_dataframe(file_path, image_root, encoder):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        category, img_name = line.split('/')
        if category in encoder.labels:
            img_path = f"{image_root}/{category}/{img_name}.jpg"
            # 檢查文件是否存在
            if os.path.exists(img_path):
                data.append({
                    'label': category,
                    'path': img_path
                })

    df = pd.DataFrame(data)
    return shuffle(df)


# 訓練 Function with Mixup and Cutmix
def train_epoch_amp(model, dataloader, optimizer, scheduler, criterion, device, scaler, epoch, logger, mixup_fn=None, gradient_accumulation_steps=4):
    """
    使用混合精度訓練的訓練函數，可加速訓練並降低記憶體使用量
    包含 Mixup、Cutmix 及梯度累積支援
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 追蹤 Mixup 準確率的變數
    mixup_correct = 0
    mixup_total = 0
    
    nan_detected = False
    batch_processed = 0  # 追蹤成功處理的批次數量
    consecutive_nan_count = 0  # 計算連續出現NaN的次數
    timeout_count = 0  # 計算可能的超時批次數

    # 重設優化器，確保每個epoch開始時梯度為零
    optimizer.zero_grad(set_to_none=True)
    
    # 追蹤累積步數
    accumulation_counter = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        # 監控處理批次時間，以偵測潛在的超時問題
        batch_start_time = time.time()
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 記錄原始標籤以計算近似準確率
        if mixup_fn is not None:
            original_targets = targets.clone()  # 在應用 mixup 前保存原始標籤
            inputs, targets = mixup_fn(inputs, targets)
        
        # 使用混合精度前向傳播
        with autocast():
            outputs = model(inputs)
            # 計算loss時除以累積步數，以保持梯度規模一致
            loss = criterion(outputs, targets) / gradient_accumulation_steps
        
        # 檢查 loss 是否為 NaN
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning(f"NaN/Inf detected in loss at epoch {epoch}, batch {i}! Skipping this batch.")
            nan_detected = True
            consecutive_nan_count += 1
            
            # 連續出現多個NaN時降低學習率
            if consecutive_nan_count >= 3:  # 降低連續NaN的閾值
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5  # 顯著降低學習率到原來的50%
                logger.warning(f"Reducing learning rate to {optimizer.param_groups[0]['lr']} due to consecutive NaNs")
                consecutive_nan_count = 0  # 重置計數器
                
            continue
        
        # 如果沒有NaN，重置連續計數
        consecutive_nan_count = 0
        
        # 使用GradScaler處理反向傳播
        try:
            scaler.scale(loss).backward()
        except RuntimeError as e:
            if "NCCL" in str(e) or "timeout" in str(e).lower():
                logger.error(f"NCCL error during backward pass: {e}")
                timeout_count += 1
                if timeout_count >= 3:
                    logger.critical("Too many NCCL errors, attempting recovery...")
                    # 嘗試主動進行同步
                    try:
                        synchronize()
                        timeout_count = 0
                    except:
                        logger.critical("Failed to synchronize processes, skipping batch")
                continue
            else:
                logger.error(f"Error during backward pass: {e}")
                continue
        
        # 更新梯度累積計數器
        accumulation_counter += 1

        # 當達到指定的累積步數或是最後一個批次時，更新參數
        is_last_batch = (i == len(dataloader) - 1)
        if accumulation_counter == gradient_accumulation_steps或is_last_batch:
            # 在更新前應用梯度裁剪 - 使用更小的閾值
            try:
                scaler.unscale_(optimizer)
                # 使用更小的梯度裁剪閾值，可以提高穩定性
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            except RuntimeError as e:
                logger.warning(f"RuntimeError during unscale_: {e}. Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                accumulation_counter = 0
                continue
            
            # 更新參數
            try:
                # 設置超時保護
                timeout_handler.start()
                scaler.step(optimizer)
                # 操作完成，停止超時保護
                timeout_handler.stop()
                scaler.update()
            except RuntimeError as e:
                if "NCCL" in str(e) or "timeout" in str(e).lower():
                    logger.error(f"NCCL error during optimizer step: {e}")
                    timeout_count += 1
                    if timeout_count >= 3:
                        logger.critical("Too many NCCL errors, attempting recovery...")
                        try:
                            synchronize()
                            timeout_count = 0
                        except:
                            pass
                else:
                    logger.error(f"Error during optimizer step: {e}")
                optimizer.zero_grad(set_to_none=True)
                accumulation_counter = 0
                continue
            
            # 只計算成功更新的批次
            total_loss += loss.item() * gradient_accumulation_steps  # 乘以累積步數恢復原始loss
            
            # 計算準確率
            _, predicted = outputs.max(1)
            
            # 如果使用了 mixup，則計算近似準確率（與最大權重標籤比較）
            if mixup_fn is not None:
                # 將 predicted 與原始標籤比較
                mixup_total += original_targets.size(0)
                mixup_correct += predicted.eq(original_targets).sum().item()
            else:
                # 標準準確率計算
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            batch_processed += 1
            
            # 重置優化器梯度和累積計數器
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0
            
            # 每隔一段時間釋放未使用的記憶體
            if i % 20 == 0:  # 增加釋放頻率
                torch.cuda.empty_cache()
        
        # 檢測批次處理時間，如果過長則發出警告
        batch_time = time.time() - batch_start_time
        if batch_time > 30:  # 30秒作為警告閾值
            logger.warning(f"Batch {i} took {batch_time:.2f} seconds to process. Possible bottleneck detected.")

    # 如果整個 epoch 中檢測到 NaN，則發出警告
    if nan_detected:
        logger.warning(f"NaN values detected during training in epoch {epoch}. Consider adjusting learning rate or gradient clipping.")

    # 防止除以零錯誤
    if batch_processed == 0:
        logger.error(f"All batches were skipped in epoch {epoch}! Training cannot proceed.")
        accuracy = 0.0
        avg_loss = float('nan')
    else:
        avg_loss = total_loss / batch_processed
        
        # 計算標準準確率或近似準確率
        if mixup_fn is None and total > 0:
            accuracy = 100. * correct / total
        elif mixup_total > 0:  # 使用 mixup 時的近似準確率
            accuracy = 100. * mixup_correct / mixup_total
        else:
            accuracy = float('nan')
        
    # 報告當前學習率
    current_lr = [group['lr'] for group in optimizer.param_groups]
    
    # 準備日誌消息
    accuracy_msg = f"{accuracy:.2f}%" if not np.isnan(accuracy) else "N/A (using mixup)"
    if mixup_fn is not None and not np.isnan(accuracy):
        accuracy_msg = f"{accuracy:.2f}% (近似值)"
    
    logger.info(f"Train Loss: {avg_loss:.3f} | Train Accuracy: {accuracy_msg} | Batches processed: {batch_processed}/{len(dataloader)} | LR: {current_lr}")
    
    # 更新調度器
    scheduler.step(epoch)
    
    return accuracy, avg_loss


# Testing Function
def test_epoch(model, dataloader, criterion, device, logger):
    """
    記憶體優化版的測試函數，分批處理測試資料並確保正確釋放記憶體
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 檢測計數器
    timeout_count = 0
    
    # 使用較小的批次進行處理，減少記憶體使用
    # 確保使用torch.no_grad()上下文，避免保存不必要的梯度信息
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Testing")):
            # 設置超時保護
            timeout_handler.start()
            
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 使用純淨的推理模式
                outputs = model(inputs)
                
                # 計算損失
                loss = criterion(outputs, targets)
                
                # 累積損失和正確預測數
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 主動清除不需要的張量以釋放記憶體
                del outputs, loss, predicted
                
                # 每10個批次清空一次CUDA快取
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
                # 停止超時保護
                timeout_handler.stop()
                
            except RuntimeError as e:
                timeout_handler.stop()
                if "NCCL" in str(e) or "timeout" in str(e).lower():
                    logger.error(f"NCCL error during testing: {e}")
                    timeout_count += 1
                    if timeout_count >= 3:
                        logger.critical("Too many NCCL errors during testing, attempting recovery...")
                        try:
                            synchronize()
                            timeout_count = 0
                        except:
                            pass
                else:
                    logger.error(f"Error during testing: {e}")
    
    # 計算平均損失和準確率
    accuracy = 100. * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('nan')
    logger.info(f"Test Loss: {avg_loss:.3f} | Test Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss


# Generate CAM for Swin Transformer V2
def generate_cam_swin_v2(model, input_tensor, class_idx=None):
    """
    Generate Class Activation Map for Swin Transformer V2
    :param model: The SwinV2 model
    :param input_tensor: Input image tensor [1, C, H, W]
    :param class_idx: Target class index (None for predicted class)
    :return: CAM as numpy array
    """
    # Make sure input tensor is on the correct device
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    gradients = []
    activations = []
    
    # Register hooks to get gradients and activations from the last stage
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks on the last stage's last layer
    target_layer = model.module.backbone.stages[-1].blocks[-1]
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    model.eval()
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[:, class_idx].backward()
    
    # Get gradients and activations
    grad = gradients[0]
    act = activations[0]
    
    # Calculate weights
    weights = grad.mean(dim=(2, 3), keepdim=True)
    
    # Generate CAM
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    return cam[0, 0].detach().cpu().numpy()


def visualize_cam(image, cam):
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5) * 255
    img_np = img_np.astype(np.uint8)

    cam_resized = cv2.resize(cam, (256, 256))  # 調整為新的圖像大小
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, cam_heatmap, 0.4, 0)
    return overlay


# Main Program
if __name__ == "__main__":
    # NOTE: The process group is initialized by torchrun automatically
    # DO NOT initialize it again with torch.distributed.init_process_group()
    
    try:
        # Get the local rank from environment variable
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # 初始化超時處理器
        timeout_handler.set_rank(local_rank)
        
        # 使用更高的超時值初始化進程組 (修正方法，移除FileStore)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            # timeout=datetime.timedelta(seconds=3600)  # 設置為1小時
        )

        print(f"Process initialized with rank {local_rank}/{world_size}")

        # 設置日誌
        logger = setup_logger(local_rank)
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.cuda, 'nccl') and hasattr(torch.cuda.nccl, 'version'):
            logger.info(f"NCCL Version: {torch.cuda.nccl.version()}")
        logger.info(f"NCCL Timeout set to {os.environ.get('NCCL_TIMEOUT', 'default')} seconds")

        # 嘗試進行初始同步，確認集體通信正常
        try:
            synchronize()
            logger.info("Initial synchronization successful")
        except Exception as e:
            logger.warning(f"Initial synchronization failed: {e}")

        # 更新參數設置 - 從頭訓練優化設定
        BATCH_SIZE = 32  # 降低批次大小，使訓練更穩定
        IMAGE_SIZE = 224  # 降低圖像大小，減少計算量
        WINDOW_SIZE = 7  # 確保與圖像大小匹配的窗口大小
        NUM_EPOCHS = 200  # 增加訓練輪數，從頭訓練需要更長時間
        IMAGE_ROOT = "food-101/images"
        TRAIN_FILE = "food-101/meta/train.txt"
        TEST_FILE = "food-101/meta/test.txt"
        GRAD_ACCUM_STEPS = 4  # 增加梯度累積步驟
        WARMUP_EPOCHS = 20  # 更長的預熱期

        LABELS = [
            'apple_pie',
            'baby_back_ribs',
            'baklava',
            'beef_carpaccio',
            'beef_tartare',
            'beet_salad',
            'beignets',
            'bibimbap',
            'bread_pudding',
            'breakfast_burrito',
            'bruschetta',
            'caesar_salad',
            'cannoli',
            'caprese_salad',
            'carrot_cake',
            'ceviche',
            'cheesecake',
            'cheese_plate',
            'chicken_curry',
            'chicken_quesadilla',
            'chicken_wings',
            'chocolate_cake',
            'chocolate_mousse',
            'churros',
            'clam_chowder',
            'club_sandwich',
            'crab_cakes',
            'creme_brulee',
            'croque_madame',
            'cup_cakes',
            'deviled_eggs',
            'donuts',
            'dumplings',
            'edamame',
            'eggs_benedict',
            'escargots',
            'falafel',
            'filet_mignon',
            'fish_and_chips',
            'foie_gras',
            'french_fries',
            'french_onion_soup',
            'french_toast',
            'fried_calamari',
            'fried_rice',
            'frozen_yogurt',
            'garlic_bread',
            'gnocchi',
            'greek_salad',
            'grilled_cheese_sandwich',
            'grilled_salmon',
            'guacamole',
            'gyoza',
            'hamburger',
            'hot_and_sour_soup',
            'hot_dog',
            'huevos_rancheros',
            'hummus',
            'ice_cream',
            'lasagna',
            'lobster_bisque',
            'lobster_roll_sandwich',
            'macaroni_and_cheese',
            'macarons',
            'miso_soup',
            'mussels',
            'nachos',
            'omelette',
            'onion_rings',
            'oysters',
            'pad_thai',
            'paella',
            'pancakes',
            'panna_cotta',
            'peking_duck',
            'pho',
            'pizza',
            'pork_chop',
            'poutine',
            'prime_rib',
            'pulled_pork_sandwich',
            'ramen',
            'ravioli',
            'red_velvet_cake',
            'risotto',
            'samosa',
            'sashimi',
            'scallops',
            'seaweed_salad',
            'shrimp_and_grits',
            'spaghetti_bolognese',
            'spaghetti_carbonara',
            'spring_rolls',
            'steak',
            'strawberry_shortcake',
            'sushi',
            'tacos',
            'takoyaki',
            'tiramisu',
            'tuna_tartare',
            'waffles'
        ]

        # 更新訓練集增強策略，減少強度以提高穩定性
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 簡化調整大小操作
            transforms.RandomCrop(IMAGE_SIZE, padding=32, padding_mode='reflect'),  # 使用邊界填充的隨機裁剪
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 降低色彩抖動強度
            RandAugment(num_ops=1, magnitude=3),  # 顯著降低RandAugment強度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),  # 進一步降低RandomErasing強度
        ])

        # 測試集轉換保持簡單
        test_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),  # 使用中心裁剪確保一致評估
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 設置 Mixup 和 Cutmix
        # 使用Mixup但降低強度，這對從頭訓練時有幫助但不能過強
        mixup_args = {
            'mixup_alpha': 0.4,  # 降低mixup alpha值
            'cutmix_alpha': 0.4,  # 降低cutmix alpha值
            'cutmix_minmax': None,
            'prob': 0.3,          # 降低mixup/cutmix的應用機率
            'switch_prob': 0.3,   # 降低切換機率
            'mode': 'batch',
            'label_smoothing': 0.1,
            'num_classes': len(LABELS)
        }
        mixup_fn = Mixup(**mixup_args)

        encoder = Label_encoder(LABELS)

        train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder)
        test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)

        train_dataset = Food101Dataset(train_df, encoder, train_transform)
        test_dataset = Food101Dataset(test_df, encoder, test_transform)

        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

        # 更新數據加載器配置 - 優化以更充分利用 GPU 記憶體
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            sampler=train_sampler, 
            num_workers=16,  # 減少工作進程數量，避免 CPU 記憶體壓力
            pin_memory=True, 
            drop_last=True,
            prefetch_factor=3,  # 提高預取因子以減少 GPU 等待時間
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            sampler=test_sampler, 
            num_workers=16,
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=True
        )

        # 初始化 SwinV2 模型，使用容錯性更高的配置
        model = swin_transformer_v2_base_classifier(
            input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
            window_size=WINDOW_SIZE,  # 保持窗口大小與圖像尺寸兼容
            num_classes=len(LABELS),
            use_checkpoint=True,  # 使用checkpoint可以減少內存使用但增加計算
            dropout_path=0.2  # 增加dropout以減少過擬合
        )
        
        model = model.to(device)
        
        # 跳過載入預訓練權重，從頭開始訓練
        logger.info("從頭開始訓練模型，不使用預訓練權重")
        
        # 使用 DDP 包裝模型
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            find_unused_parameters=True,  # 啟用未使用參數檢測以解決DDP訓練問題
            broadcast_buffers=False  # 保持關閉緩衝區廣播以減少通信量
        )
        
        # 禁用可能與分散式訓練衝突的優化選項
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.optimize_ddp = False
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 32  # 限制快取大小
        
        # 優化器設置 - 使用更適合從頭訓練的配置
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,  # 大幅提高初始學習率，從頭訓練時需要更大的學習率
            weight_decay=0.01,  # 降低權重衰減，從頭訓練時過大的衰減會阻礙收斂
            eps=1e-8,  # 保持較高的epsilon值提高數值穩定性
            betas=(0.9, 0.999)  # 默認動量參數
        )
        
        # 使用適合從頭訓練的損失函數
        # 從頭訓練時標籤平滑很重要，但平滑係數應該較小
        base_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # 僅在訓練穩定後使用更複雜的損失函數
        criterion_train = SoftTargetCrossEntropy() if mixup_fn else base_criterion
        criterion_test = nn.CrossEntropyLoss()  # 測試時使用標準交叉熵
        
        # 使用更適合從頭訓練的調度器 - 更長的預熱和較慢的衰減
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=NUM_EPOCHS,
            lr_min=1e-5,  # 提高最小學習率
            warmup_t=WARMUP_EPOCHS,  # 使用更長的預熱期(20個epoch)
            warmup_lr_init=1e-6,  # 從較小的學習率開始預熱
            cycle_limit=1,
            t_in_epochs=True,
            warmup_prefix=True  # 確保預熱期不計入余弦衰減週期
        )

        # 初始化混合精度訓練的scaler，優化設定以更好利用 GPU 記憶體
        scaler = GradScaler(
            init_scale=2**16,  # 使用更高的初始scale值
            growth_factor=2.0,  # 將值提高回預設值
            backoff_factor=0.5,
            growth_interval=100
        )
        
        # 在主進程上初始化TensorBoard writer
        if local_rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tensorboard_dir = f'runs/swin_v2_food101_{timestamp}'
            writer = SummaryWriter(log_dir=tensorboard_dir)
            logger.info(f"TensorBoard log directory: {tensorboard_dir}")
            logger.info(f"Training with:")
            logger.info(f"- Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
            logger.info(f"- Window size: {WINDOW_SIZE}x{WINDOW_SIZE}")
            logger.info(f"- Batch size: {BATCH_SIZE}")
            logger.info(f"- Epochs: {NUM_EPOCHS}")
            logger.info(f"- Learning rate: {5e-5}")
            logger.info(f"- Weight decay: 0.05")
            logger.info(f"- Augmentations: RandAugment, Mixup, Cutmix, RandomErasing")
            logger.info(f"- Stochastic depth: 0.1")
        
        # 添加定期檢查點保存機制，以便在出現錯誤時恢復訓練
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 嘗試加載檢查點（如果存在）
        start_epoch = 0
        best_acc = 0
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.module.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        
        # 主訓練循環
        for epoch in range(start_epoch, NUM_EPOCHS):
            logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            
            # 設置epoch以確保shuffling在每個epoch都不同
            train_sampler.set_epoch(epoch)
            
            # 在訓練開始前進行同步
            try:
                synchronize()
            except:
                logger.warning("Failed to synchronize before training epoch")
            
            # 在訓練時使用 mixup_fn 和較小的梯度累積步數
            try:
                train_acc, train_loss = train_epoch_amp(
                    model, train_loader, optimizer, scheduler, criterion_train, 
                    device, scaler, epoch, logger, mixup_fn=mixup_fn,
                    gradient_accumulation_steps=2  # 使用較小的梯度累積步數，以更好地利用 GPU 資源
                )
            except Exception as e:
                logger.error(f"Error during training epoch: {e}")
                logger.error(traceback.format_exc())
                
                # 儲存緊急檢查點
                if local_rank == 0:
                    emergency_path = os.path.join(checkpoint_dir, f"emergency_epoch_{epoch}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_acc': best_acc
                    }, emergency_path)
                    logger.info(f"Emergency checkpoint saved to {emergency_path}")
                
                # 嘗試重新同步進程
                try:
                    synchronize()
                except:
                    logger.critical("Failed to recover after training error")
                continue
            
            # 在測試前進行同步
            try:
                synchronize()
            except:
                logger.warning("Failed to synchronize before evaluation")
            
            # 在測試時不使用 mixup，使用標準的 CrossEntropyLoss
            try:
                test_acc, test_loss = test_epoch(model, test_loader, criterion_test, device, logger)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                logger.error(traceback.format_exc())
                test_acc = 0.0
                test_loss = float('nan')
            
            # 在主進程上記錄到TensorBoard
            if local_rank == 0:
                # 記錄訓練和測試指標
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)
                if not np.isnan(train_acc):
                    writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/test', test_acc, epoch)
                
                # 記錄學習率
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
                
                # 儲存定期檢查點
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc': best_acc
                }, checkpoint_path)
                
                # 如果是最佳結果則儲存為最佳模型
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.module.state_dict(), f'swinv2_food101_best.pth')
                    logger.info(f"New best model saved with accuracy: {test_acc:.2f}%")
            
            # 在epoch結束後確保所有進程同步
            try:
                synchronize()
            except:
                logger.warning("Failed to synchronize after epoch")
            
            # 主動釋放記憶體
            torch.cuda.empty_cache()
    
    except Exception as e:
        import traceback
        print(f"Error in main process: {e}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        if torch.distributed.is_initialized():
            try:
                destroy_process_group()
            except:
                print("Failed to destroy process group")
