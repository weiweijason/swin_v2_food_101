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

# 設置多進程啟動方法為 'spawn'，這可以解決某些連接問題
# 這必須在導入任何其他與 torch.multiprocessing 相關的模塊之前設置
try:
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Import the SwinV2 classifier
from swin_transformer_v2_classifier import swin_transformer_v2_base_classifier

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

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# Updated prepare_dataframe function
def prepare_dataframe(file_path, image_root, encoder):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        category, img_name = line.split('/')
        if category in encoder.labels:
            data.append({
                'label': category,
                'path': f"{image_root}/{category}/{img_name}.jpg"
            })

    df = pd.DataFrame(data)
    return shuffle(df)


# Training Function with Mixup and Cutmix
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

    # 重設優化器，確保每個epoch開始時梯度為零
    optimizer.zero_grad(set_to_none=True)
    
    # 追蹤累積步數
    accumulation_counter = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
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
            if consecutive_nan_count >= 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8  # 降低學習率到原來的80%
                logger.warning(f"Reducing learning rate due to consecutive NaNs at epoch {epoch}")
                consecutive_nan_count = 0  # 重置計數器
                
            continue
        
        # 如果沒有NaN，重置連續計數
        consecutive_nan_count = 0
        
        # 使用GradScaler處理反向傳播
        scaler.scale(loss).backward()
        
        # 更新梯度累積計數器
        accumulation_counter += 1

        # 當達到指定的累積步數或是最後一個批次時，更新參數
        is_last_batch = (i == len(dataloader) - 1)
        if accumulation_counter == gradient_accumulation_steps or is_last_batch:
            # 在更新前應用梯度裁剪
            try:
                # 檢查梯度是否已被 unscaled，避免重複調用
                if any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in model.parameters() if p.requires_grad):
                    logger.warning(f"NaN/Inf detected in gradients before unscaling at epoch {epoch}, batch {i}!")
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_counter = 0
                    continue
                    
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            except RuntimeError as e:
                logger.warning(f"RuntimeError during unscale_: {e}. Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                accumulation_counter = 0
                continue
            
            # 檢查梯度是否包含 NaN
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning(f"NaN/Inf detected in gradients for {name} at epoch {epoch}, batch {i}!")
                        valid_gradients = False
                        break
            
            # 只有在梯度有效時才更新參數
            if valid_gradients:
                # 更新參數
                scaler.step(optimizer)
                scaler.update()
                
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
            else:
                nan_detected = True
                
            # 重置優化器梯度和累積計數器
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

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
    
    # 使用較小的批次進行處理，減少記憶體使用
    # 確保使用torch.no_grad()上下文，避免保存不必要的梯度信息
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
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
            torch.cuda.empty_cache()  # 顯式清空CUDA快取
    
    # 計算平均損失和準確率
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
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
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        dist.init_process_group(backend='nccl', init_method='env://')

        print(f"Process initialized with rank {local_rank}")

        # 設置日誌
        logger = setup_logger(local_rank)

        # 更新參數設置
        BATCH_SIZE = 64  # 從64增加到1024
        IMAGE_SIZE = 224   # 從224增加到256
        WINDOW_SIZE = 7    # 從7增加到8
        NUM_EPOCHS = 30    # 從20增加到30
        IMAGE_ROOT = "food-101/images"
        TRAIN_FILE = "food-101/meta/train.txt"
        TEST_FILE = "food-101/meta/test.txt"

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

        # 更新訓練集增強策略，添加 RandAugment, RandomErasing
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # 過度大小以便隨機裁剪
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            RandAugment(num_ops=2, magnitude=5),  # 降低 RandAugment 強度以提高穩定性
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            RandomErasing(p=0.15, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # 降低RandomErasing概率
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
        mixup_args = {
            'mixup_alpha': 0.8,
            'cutmix_alpha': 1.0,
            'cutmix_minmax': None,
            'prob': 0.5,
            'switch_prob': 0.5,
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

        # 更新數據加載器配置 - 修復連接問題
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            sampler=train_sampler, 
            num_workers=16,  # 減少工作進程數量，避免連接拒絕錯誤
            pin_memory=True, 
            drop_last=True,
            prefetch_factor=2,  # 降低預取因子以減少資源需求
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE // 2, 
            sampler=test_sampler, 
            num_workers=16,  # 減少工作進程數量
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        # 初始化 SwinV2 模型，使用新的參數
        model = swin_transformer_v2_base_classifier(
            input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
            window_size=WINDOW_SIZE,
            num_classes=len(LABELS),
            use_checkpoint=True,
            # 添加 stochastic depth (dropout_path)
            dropout_path=0.3  # 設置 stochastic depth 率為 0.3
        )
        
        model = model.to(device)
        
        # 跳過載入預訓練權重，從頭開始訓練
        logger.info("從頭開始訓練模型，不使用預訓練權重")
        
        # 先將模型包裝在 DDP 中
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        model._set_static_graph()
        
        # 禁用 torch.compile 與 DDP 的優化以避免衝突
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.optimize_ddp = False
            # 允許在出錯時回退到 eager 模式
            torch._dynamo.config.suppress_errors = True
            
            if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
                try:
                    logger.info("使用 torch.compile() 加速模型，已禁用 optimize_ddp")
                    # 模型已經是 DDP 包裝的，不能直接編譯，所以我們編譯內部模塊
                    model.module = torch.compile(model.module, mode='reduce-overhead')
                except Exception as e:
                    logger.warning(f"torch.compile() 失敗: {e}，跳過編譯")

        # 使用 AdamW 優化器並設置初始學習率為 1e-4，比原來的4e-5更高
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,  # 提高初始學習率
            weight_decay=0.05
        )
        
        # 使用適合 Mixup/Cutmix 的損失函數
        criterion = SoftTargetCrossEntropy()
        
        # 使用 CosineLRScheduler 設置 cosine decay
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=NUM_EPOCHS,
            lr_min=1e-6,
            warmup_t=10,  # 增加預熱期
            warmup_lr_init=1e-7,
            cycle_limit=1,
            t_in_epochs=True,
        )

        # 初始化混合精度訓練的scaler
        scaler = GradScaler()
        
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
            logger.info(f"- Learning rate: {4e-5}")
            logger.info(f"- Weight decay: 0.05")
            logger.info(f"- Augmentations: RandAugment, Mixup, Cutmix, RandomErasing")
            logger.info(f"- Stochastic depth: 0.3")
        
        best_acc = 0
        for epoch in range(NUM_EPOCHS):
            logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            train_sampler.set_epoch(epoch)
            
            # 在訓練時使用 mixup_fn
            train_acc, train_loss = train_epoch_amp(
                model, train_loader, optimizer, scheduler, criterion, 
                device, scaler, epoch, logger, mixup_fn=mixup_fn
            )
            
            # 在測試時不使用 mixup，使用標準的 CrossEntropyLoss
            test_acc, test_loss = test_epoch(model, test_loader, nn.CrossEntropyLoss(), device, logger)
            
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
            
            if test_acc > best_acc:
                best_acc = test_acc
                if local_rank == 0:
                    torch.save(model.state_dict(), f'swinv2_food101_best.pth')
                    logger.info(f"New best model saved with accuracy: {test_acc:.2f}%")
    
    except Exception as e:
        import traceback
        print(f"Error in main process: {e}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        if torch.distributed.is_initialized():
            destroy_process_group()
