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
from swin_transformer_v2_classifier import (
    swin_transformer_v2_base_classifier,
    swin_transformer_v2_tiny_classifier,
    swin_transformer_v2_small_classifier,
    swin_transformer_v2_large_classifier
)

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
    if (img_size % window_size) != 0:
        print(f"WARNING: img_size {img_size} is not divisible by window_size {window_size}")
        new_img_size = (img_size // window_size) * window_size
        print(f"Adjusting img_size to {new_img_size}")
        return new_img_size
    return img_size

# 設置日誌格式
def setup_logger(local_rank):
    # 創建日誌格式
    log_format = '%(asctime)s - %(level別)s - Rank[%(rank)s] - %(message)s'
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
    world size = dist.get_world_size()
    if world size == 1:
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


# Training and Testing Functions
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 移除 scheduler.step() 在這裡的調用，改為在外部每個 epoch 調用一次

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Train Loss: {total_loss / len(dataloader):.3f} | Train Accuracy: {accuracy:.2f}%")
    return accuracy, total_loss / len(dataloader)


def test_epoch(model, dataloader, criterion, device, logger=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Loss: {total_loss / len(dataloader):.3f} | Test Accuracy: {accuracy:.2f}%")
    return accuracy, total_loss / len(dataloader)


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
        
        # 設置分散式調試環境變數，幫助識別未使用的參數
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        
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

        # 更新參數設置 - 符合指定的 Swin V2 Large 模型需求
        BATCH_SIZE = 16  # 對於大型模型降低批次大小
        IMAGE_SIZE = 192  # 按照用戶要求設置為 192
        WINDOW_SIZE = 12  # 按照用戶要求設置為 12
        NUM_EPOCHS = 50  # 減少訓練輪數，因為使用預訓練權重後收斂更快
        IMAGE_ROOT = "food-101/images"
        TRAIN_FILE = "food-101/meta/train.txt"
        TEST_FILE = "food-101/meta/test.txt"
        GRAD_ACCUM_STEPS = 2
        WARMUP_EPOCHS = 5

        # 設置預訓練權重路徑為指定的 large 模型
        PRETRAINED_WEIGHTS = "swinv2_large_patch4_window12_192_22k.pth"

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

        # 更新訓練集增強策略，添加更多數據增強技術
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),  # 添加隨機旋轉
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加顏色抖動
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 添加隨機平移
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)  # 添加隨機擦除區域
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

        # 模型規格選擇 (可選: 'tiny', 'small', 'base', 'large')
        MODEL_SIZE = "large"  # 默認使用 large 模型

        # 根據模型規格和圖像尺寸調整窗口大小
        if MODEL_SIZE == "large":
            # large 模型需要更多顯存，可能需要降低批次大小
            BATCH_SIZE = 16
        elif MODEL_SIZE == "tiny":
            # tiny 模型較輕量，可以使用更大的批次大小
            BATCH_SIZE = 64
        else:
            # small 或 base 模型使用默認批次大小
            BATCH_SIZE = 32

        # 根據選定的模型規格初始化不同的 SwinV2 模型
        logger.info(f"使用 {MODEL_SIZE} 規格的 Swin Transformer V2 模型")
        
        if MODEL_SIZE == "tiny":
            model = swin_transformer_v2_tiny_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=WINDOW_SIZE,
                num_classes=len(LABELS),
                use_checkpoint=True,
                dropout_path=0.2
            )
        elif MODEL_SIZE == "small":
            model = swin_transformer_v2_small_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=WINDOW_SIZE,
                num_classes=len(LABELS),
                use_checkpoint=True,
                dropout_path=0.2
            )
        elif MODEL_SIZE == "large":
            model = swin_transformer_v2_large_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=WINDOW_SIZE,
                num_classes=len(LABELS),
                use_checkpoint=True,
                dropout_path=0.2
            )
        else:  # 默認使用 base 模型
            model = swin_transformer_v2_base_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=WINDOW_SIZE,
                num_classes=len(LABELS),
                use_checkpoint=True,
                dropout_path=0.2
            )

        # 使用預訓練權重
        if os.path.exists(PRETRAINED_WEIGHTS):
            logger.info(f"載入預訓練權重: {PRETRAINED_WEIGHTS}")
            model.load_pretrained(PRETRAINED_WEIGHTS)
        else:
            logger.warning(f"預訓練權重文件 {PRETRAINED_WEIGHTS} 不存在，將從頭開始訓練")

        model = model.to(device)
        
        # 對 checkpoint 功能進行設置
        # 手動修補 torch.utils.checkpoint 函數以避免重入問題
        if hasattr(torch.utils, 'checkpoint'):
            # 嘗試直接替換 checkpoint 函數中的 use_reentrant 參數默認值
            import inspect
            
            original_checkpoint = torch.utils.checkpoint.checkpoint
            
            # 創建一個新的包裝函數，強制設置 use_reentrant=False
            def patched_checkpoint(*args, **kwargs):
                if 'use_reentrant' not in kwargs:
                    kwargs['use_reentrant'] = False
                return original_checkpoint(*args, **kwargs)
            
            # 替換原始函數
            torch.utils.checkpoint.checkpoint = patched_checkpoint
            logger.info("已修補 torch.utils.checkpoint 函數，設置 use_reentrant=False 以解決重入問題")

        model = model.to(device)
        
        # 使用預訓練權重進行模型初始化
        logger.info(f"使用 {PRETRAINED_WEIGHTS} 預訓練權重進行模型初始化")
        
        # 使用 DDP 包裝模型
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            find_unused_parameters=True,  # 重新啟用找尋未使用參數
            broadcast_buffers=False  # 保持關閉緩衝區廣播以減少通信量
        )

        logger.info("已啟用 find_unused_parameters，將嘗試跟踪未使用的參數")
        
        # 設置模型為 DDP 模式，啟用虛擬損失以解決未使用參數問題
        model.module.in_ddp_mode = True
        logger.info("已啟用 DDP 模式的虛擬損失機制，確保所有參數參與梯度計算")
        
        # 禁用可能與分散式訓練衝突的優化選項
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.optimize_ddp = False
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 32  # 限制快取大小
        
        # 優化器設置 - 使用更適合微調的配置 (非從頭訓練)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=2e-5,  # 將學習率降低到與 V1 版本一致
            weight_decay=0.1,  # 增加權重衰減從0.05到0.1以強化正則化
            eps=1e-8,  # 保持較高的epsilon值提高數值穩定性
            betas=(0.9, 0.999)  # 默認動量參數
        )
        
        # 使用適合從頭訓練的損失函數
        # 使用標準交叉熵損失而不是標籤平滑，以避免維度不匹配問題
        base_criterion = nn.CrossEntropyLoss()

        # 不使用 Mixup 和 SoftTargetCrossEntropy，改用標準損失函數
        criterion_train = base_criterion
        criterion_test = base_criterion
        
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
        
        # 嘗試加載特定 epoch 的檢查點（從第35個epoch重新開始）
        start_epoch = 0
        best_acc = 0
        RESUME_FROM_EPOCH = 35  # 從第35個epoch重新開始訓練
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{RESUME_FROM_EPOCH}.pth")
        
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading specific checkpoint from epoch {RESUME_FROM_EPOCH}: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.module.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch']  # 直接使用檢查點中的 epoch
                best_acc = checkpoint['best_acc']
                logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                # 如果無法加載特定 epoch 的檢查點，嘗試加載最新的檢查點
                latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
                if os.path.exists(latest_checkpoint_path):
                    logger.info(f"Trying to load latest checkpoint instead: {latest_checkpoint_path}")
                    try:
                        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                        model.module.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        if 'scaler_state_dict' in checkpoint:
                            scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        start_epoch = checkpoint['epoch'] + 1
                        best_acc = checkpoint['best_acc']
                        logger.info(f"Latest checkpoint loaded. Resuming from epoch {start_epoch}")
                    except Exception as e2:
                        logger.warning(f"Failed to load latest checkpoint: {e2}")
        else:
            logger.warning(f"Specified epoch {RESUME_FROM_EPOCH} checkpoint not found at {checkpoint_path}")
            # 嘗試加載最新的檢查點
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            if os.path.exists(latest_checkpoint_path):
                logger.info(f"Trying to load latest checkpoint instead: {latest_checkpoint_path}")
                try:
                    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if 'scaler_state_dict' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    best_acc = checkpoint['best_acc']
                    logger.info(f"Latest checkpoint loaded. Resuming from epoch {start_epoch}")
                except Exception as e:
                    logger.warning(f"Failed to load latest checkpoint: {e}")
        
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
                train_acc, train_loss = train_epoch(
                    model, train_loader, optimizer, scheduler, criterion_train, 
                    device, epoch
                )
                
                # 在每個 epoch 後調用 scheduler.step()，而不是在每個 batch 後
                scheduler.step(epoch)
                logger.info(f"Learning rate updated: {optimizer.param_groups[0]['lr']:.6f}")
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
