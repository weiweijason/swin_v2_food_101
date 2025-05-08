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
    return accuracy, total_loss / len(dataloader


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
    return accuracy, total_loss / len(dataloader


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

        # 更新參數設置 - 更新配置以提高精確度
        BATCH_SIZE = 32  # 增大批次大小以獲得更穩定的梯度
        IMAGE_SIZE = 224  # 增加圖像尺寸以提高特徵提取能力
        WINDOW_SIZE = 7  # 設置窗口大小為圖像尺寸的約1/32
        NUM_EPOCHS = 50  # 增加訓練輪數
        IMAGE_ROOT = "food-101/images"
        TRAIN_FILE = "food-101/meta/train.txt"
        TEST_FILE = "food-101/meta/test.txt"
        GRAD_ACCUM_STEPS = 1  # 減少梯度累積以更頻繁更新模型
        WARMUP_EPOCHS = 3  # 減少預熱時間

        # 設置預訓練權重路徑為指定的 large 模型
        PRETRAINED_WEIGHTS = "swinv2_imagenet_pretrained.pth"

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

        # 大幅增強訓練集增強策略，採用更強力的數據增強來增加泛化能力
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE+32, IMAGE_SIZE+32)),  # 先放大圖像
            transforms.RandomCrop(IMAGE_SIZE),  # 隨機裁剪以增加多樣性
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-20, 20)),  # 修正: 使用元組而非單一整數
            transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=(-0.15, 0.15)),  # 修正: 使用元組表示範圍
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])

        # 測試集轉換強化，採用多尺度測試策略 (Test Time Augmentation)
        test_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE+32, IMAGE_SIZE+32)),  # 先放大圖像
            transforms.CenterCrop(IMAGE_SIZE),  # 中心裁剪確保評估一致性
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 設置 Mixup 和 Cutmix，但減小混合強度以避免過度混淆類別
        mixup_args = {
            'mixup_alpha': 0.2,  # 降低 mixup alpha 以保持更多原始標籤信息
            'cutmix_alpha': 0.2,  # 降低 cutmix alpha 
            'cutmix_minmax': None,
            'prob': 0.6,  # 增加使用概率以提高訓練多樣性
            'switch_prob': 0.3,
            'mode': 'batch',
            'label_smoothing': 0.05,  # 減少標籤平滑強度，避免模型過度自信
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

        # 更新數據加載器配置 - 平衡效率和內存使用
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            sampler=train_sampler, 
            num_workers=6,  # 減少工作進程數量，避免 CPU 記憶體壓力
            pin_memory=True, 
            drop_last=True,
            prefetch_factor=2,  # 降低預取因子以減少內存壓力
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE*2,  # 增加測試批次大小以加速評估
            sampler=test_sampler, 
            num_workers=6,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        # 使用更好的模型規格
        MODEL_SIZE = "large"  # 使用 base 模型作為基礎，平衡性能和效率

        # 根據模型規格初始化 SwinV2 模型
        logger.info(f"使用 {MODEL_SIZE} 規格的 Swin Transformer V2 模型")
        
        if (MODEL_SIZE == "tiny"):
            model = swin_transformer_v2_tiny_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=WINDOW_SIZE,
                num_classes=len(LABELS),
                use_checkpoint=True,
                dropout_path=0.3  # 增加 dropout 以提高泛化能力
            )
        elif (MODEL_SIZE == "small"):
            model = swin_transformer_v2_small_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=WINDOW_SIZE,
                num_classes=len(LABELS),
                use_checkpoint=True,
                dropout_path=0.3
            )
        elif (MODEL_SIZE == "large"):
            model = swin_transformer_v2_large_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=WINDOW_SIZE,
                num_classes=len(LABELS),
                use_checkpoint=True,
                dropout_path=0.3
            )
        else:  # 默認使用 base 模型
            model = swin_transformer_v2_base_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=WINDOW_SIZE,
                num_classes=len(LABELS),
                use_checkpoint=True,
                dropout_path=0.3
            )

        # 使用預訓練權重，更好地初始化模型
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
            find_unused_parameters=False,  # 關閉找尋未使用參數以提高訓練效率
            broadcast_buffers=False  # 保持關閉緩衝區廣播以減少通信量
        )

        # 採用分層學習率策略，微調時給不同層設置不同學習率
        # 實現方式：給主幹網絡較小的學習率，給分類頭較大的學習率
        backbone_params = []
        head_params = []
        
        # 將參數分為主幹參數和頭部參數
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': 5e-6, 'weight_decay': 0.02},  # 較小的學習率和權重衰減
            {'params': head_params, 'lr': 5e-5, 'weight_decay': 0.05}       # 較大的學習率和權重衰減
        ]
        
        # 優化器設置 - 分層學習率AdamW
        optimizer = optim.AdamW(
            param_groups,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
        
        # 使用標籤平滑交叉熵損失以提高泛化能力
        base_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # 使用mixup時的SoftTarget損失
        criterion_train = SoftTargetCrossEntropy()
        criterion_test = nn.CrossEntropyLoss()  # 測試時使用標準損失
        
        # 使用帶有預熱和余弦退火的學習率調度器
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=NUM_EPOCHS,
            lr_min=1e-7,        # 較低的最小學習率以避免過早收斂
            warmup_t=5,         # 5個epoch的預熱期
            warmup_lr_init=1e-7, # 從很小的學習率開始
            cycle_limit=1,
            t_in_epochs=True,
            warmup_prefix=True  # 確保預熱期不計入余弦衰減週期
        )

        # 初始化混合精度訓練的scaler
        scaler = GradScaler(
            init_scale=2**10,    # 使用較小的初始scale
            growth_factor=2.0,
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
            logger.info(f"- Augmentations: Advanced color jitter, perspective, affine transforms, etc.")
            logger.info(f"- Learning rates: backbone {param_groups[0]['lr']}, head {param_groups[1]['lr']}")
            logger.info(f"- Weight decay: backbone {param_groups[0]['weight_decay']}, head {param_groups[1]['weight_decay']}")
            logger.info(f"- Stochastic depth: 0.3 (increased for better regularization)")
        
        # 檢查點設置
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 從最後一個檢查點開始訓練
        start_epoch = 0
        best_acc = 0
        
        # 嘗試加載最新的檢查點
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint_path):
            logger.info(f"Loading latest checkpoint: {latest_checkpoint_path}")
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
        
        # 實現漸進式學習 - 隨著訓練進行增加數據增強強度
        def adjust_augmentation_strength(transform, epoch, total_epochs):
            """隨著訓練進行調整數據增強強度"""
            progress = epoch / total_epochs
            # 開始時使用較溫和的增強，後期使用更強的增強
            for t in transform.transforms:
                if isinstance(t, transforms.ColorJitter):
                    # 循序漸進增加顏色抖動強度
                    factor = min(1.0, 0.5 + progress)  # 從0.5增加到1.0
                    # 修正: 使用元組格式設置 ColorJitter 參數
                    brightness_factor = 0.3 * factor
                    contrast_factor = 0.3 * factor
                    saturation_factor = 0.3 * factor
                    hue_factor = 0.15 * factor
                    
                    t.brightness = (max(0.7, 1.0 - brightness_factor), 1.0 + brightness_factor)
                    t.contrast = (max(0.7, 1.0 - contrast_factor), 1.0 + contrast_factor)
                    t.saturation = (max(0.7, 1.0 - saturation_factor), 1.0 + saturation_factor)
                    t.hue = (-hue_factor, hue_factor)
                elif isinstance(t, transforms.RandomRotation):
                    # 增加旋轉角度 - 修正：使用元組格式
                    t.degrees = (-int(10 + 10 * progress), int(10 + 10 * progress))  # 從±10度增加到±20度
                elif isinstance(t, transforms.RandomAffine):
                    # 增加變形強度 - 修正：使用元組格式
                    t.degrees = (-int(5 + 5 * progress), int(5 + 5 * progress))  # 從±5度增加到±10度
                    t.translate = (0.1 + 0.05 * progress, 0.1 + 0.05 * progress)
                elif isinstance(t, RandomErasing):
                    # 增加擦除概率
                    t.p = 0.2 + 0.1 * progress  # 從0.2增加到0.3
        
        # 實現更高效的測試函數，使用測試時數據增強 (TTA)
        def test_with_tta(model, dataloader, criterion, device, num_augments=3):
            """使用測試時增強進行評估以提高準確率"""
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in tqdm(dataloader, desc="Testing with TTA"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 原始預測
                    outputs = model(inputs)
                    
                    # TTA - 水平翻轉
                    flipped_inputs = torch.flip(inputs, dims=[3])
                    flipped_outputs = model(flipped_inputs)
                    
                    # TTA - 縮放然後裁剪
                    b, c, h, w = inputs.shape
                    scaled_inputs = torch.nn.functional.interpolate(
                        inputs, size=(int(h*1.1), int(w*1.1)), mode='bilinear', align_corners=False
                    )
                    # 中心裁剪回原始尺寸
                    crop_h, crop_w = h, w
                    start_h = (scaled_inputs.size(2) - crop_h) // 2
                    start_w = (scaled_inputs.size(3) - crop_w) // 2
                    scaled_inputs = scaled_inputs[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
                    scaled_outputs = model(scaled_inputs)
                    
                    # 結合所有預測 (簡單平均)
                    combined_outputs = (outputs + flipped_outputs + scaled_outputs) / 3.0
                    
                    # 計算損失和準確率
                    loss = criterion(combined_outputs, targets)
                    total_loss += loss.item()
                    _, predicted = combined_outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total
            print(f"Test Loss: {total_loss / len(dataloader):.3f} | Test Accuracy: {accuracy:.2f}%")
            return accuracy, total_loss / len(dataloader)
        
        # 主訓練循環
        for epoch in range(start_epoch, NUM_EPOCHS):
            logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            
            # 根據訓練進度調整數據增強強度
            if hasattr(train_transform, 'transforms'):
                adjust_augmentation_strength(train_transform, epoch, NUM_EPOCHS)
                logger.info(f"Adjusted augmentation strength for epoch {epoch+1}")
            
            # 設置epoch以確保shuffling在每個epoch都不同
            train_sampler.set_epoch(epoch)
            
            # 在訓練開始前進行同步
            try:
                synchronize()
            except:
                logger.warning("Failed to synchronize before training epoch")
            
            # 訓練一個epoch
            try:
                train_acc, train_loss = train_epoch(
                    model, train_loader, optimizer, scheduler, criterion_train, 
                    device, epoch
                )
                
                # 在每個 epoch 後調用 scheduler.step()
                scheduler.step(epoch)
                logger.info(f"Learning rate updated: {optimizer.param_groups[0]['lr']:.6f} (backbone), {optimizer.param_groups[1]['lr']:.6f} (head)")
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
            
            # 使用測試時增強進行評估
            try:
                if epoch >= NUM_EPOCHS - 5:  # 在最後5個epoch使用TTA進行更精確的評估
                    test_acc, test_loss = test_with_tta(model, test_loader, criterion_test, device)
                else:
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
                writer.add_scalar('Learning_rate/backbone', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Learning_rate/head', optimizer.param_groups[1]['lr'], epoch)
                
                # 儲存最新檢查點
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc': best_acc
                }, latest_checkpoint_path)
                
                # 每5個epoch儲存一個定期檢查點
                if (epoch + 1) % 5 == 0:
                    epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_acc': best_acc
                    }, epoch_checkpoint_path)
                    logger.info(f"Saved checkpoint at epoch {epoch+1}")
                
                # 如果是最佳結果則儲存為最佳模型
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.module.state_dict(), os.path.join('outputs', 'swinv2_food101_best.pth'))
                    logger.info(f"New best model saved with accuracy: {test_acc:.2f}%")
                    
                    # 也保存一個帶有時間戳的最佳模型副本，避免被覆蓋
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    best_model_with_timestamp = os.path.join('outputs', f'swinv2_food101_best_{timestamp}_{test_acc:.2f}.pth')
                    torch.save(model.module.state_dict(), best_model_with_timestamp)
            
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
