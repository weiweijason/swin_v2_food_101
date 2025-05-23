import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
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


# Training and Testing Functions
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Apply gradient clipping after backward pass but before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Train Loss: {total_loss / len(dataloader):.3f} | Train Accuracy: {accuracy:.2f}%")
    
    # Correct - stepping scheduler once per epoch
    scheduler.step()
    
    return accuracy  # It's helpful to return the accuracy for monitoring


def test_epoch(model, dataloader, criterion, device, logger):
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

    cam_resized = cv2.resize(cam, (224, 224))
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, cam_heatmap, 0.4, 0)
    return overlay


def train_epoch_amp(model, dataloader, optimizer, scheduler, criterion, device, scaler, epoch, logger):
    """
    使用混合精度訓練的訓練函數，可加速訓練並降低記憶體使用量
    增加 NaN 檢測和處理
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    nan_detected = False
    batch_processed = 0  # 追蹤成功處理的批次數量
    consecutive_nan_count = 0  # 計算連續出現NaN的次數

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True可提供更好的性能和內存使用
        
        # 使用混合精度前向傳播
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
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
        
        # 在更新前應用梯度裁剪 - 使用更嚴格的閾值
        try:
            # 檢查梯度是否已被 unscaled，避免重複調用
            if any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in model.parameters() if p.requires_grad):
                logger.warning(f"NaN/Inf detected in gradients before unscaling at epoch {epoch}, batch {i}!")
                continue
                
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 將閾值從1.0降至0.5增加穩定性
        except RuntimeError as e:
            logger.warning(f"RuntimeError during unscale_: {e}. Skipping this batch.")
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
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_processed += 1
        else:
            nan_detected = True
            continue

    # 如果整個 epoch 中檢測到 NaN，則發出警告
    if nan_detected:
        logger.warning(f"NaN values detected during training in epoch {epoch}. Consider adjusting learning rate or gradient clipping.")

    # 防止除以零錯誤
    if total == 0:
        logger.error(f"All batches were skipped in epoch {epoch}! Training cannot proceed.")
        accuracy = 0.0
        avg_loss = float('nan')
    else:
        accuracy = 100. * correct / total
        avg_loss = total_loss / batch_processed if batch_processed > 0 else float('nan')
        
    # 報告當前學習率
    current_lr = [group['lr'] for group in optimizer.param_groups]
    logger.info(f"Train Loss: {avg_loss:.3f} | Train Accuracy: {accuracy:.2f}% | Batches processed: {batch_processed}/{len(dataloader)} | LR: {current_lr}")
    
    # 更新調度器 - 傳入epoch參數
    scheduler.step(epoch)
    
    return accuracy, avg_loss


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

        BATCH_SIZE = 64  # 從原本的32增加到64
        IMAGE_SIZE = 224
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

        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # 強化版資料增強的測試集轉換
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        encoder = Label_encoder(LABELS)

        train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder)
        test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)

        train_dataset = Food101Dataset(train_df, encoder, transform)
        # 更新測試資料集使用適合測試的轉換
        test_dataset = Food101Dataset(test_df, encoder, transform_test)

        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

        num_epochs = 20

        # Initialize the SwinV2 model
        model = swin_transformer_v2_base_classifier(
            input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
            window_size=7,
            num_classes=len(LABELS),
            use_checkpoint=True
        )
        
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        model._set_static_graph()  

        # 使用分層學習率，為骨幹和分類頭設置不同的學習率
        parameters = [
            {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-6},  # 進一步降低骨幹網路學習率
            {'params': [p for n, p in model.named_parameters() if 'head' in n], 'lr': 5e-6}  # 降低分類頭學習率
        ]
        
        optimizer = optim.AdamW(
            parameters,
            weight_decay=0.005  # 進一步降低權重衰減值，防止數值不穩定
        )
        
        # 加入標籤平滑提高泛化能力
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 使用更穩定的學習率調度器
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=1e-7,  # 將最小學習率降低
            warmup_t=5,   # 保持5個epochs的預熱期
            warmup_lr_init=5e-8,  # 降低熱身期初始學習率
            cycle_limit=1,  # 確保只有一個週期
            cycle_decay=0.5  # 增加週期衰減
        )

        # 初始化混合精度訓練的scaler
        scaler = GradScaler()
        
        # 在主進程上初始化TensorBoard writer
        if local_rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tensorboard_dir = f'runs/swin_v2_food101_{timestamp}'
            writer = SummaryWriter(log_dir=tensorboard_dir)
            logger.info(f"TensorBoard log directory: {tensorboard_dir}")
        
        best_acc = 0
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_sampler.set_epoch(epoch)
            train_acc, train_loss = train_epoch_amp(model, train_loader, optimizer, scheduler, criterion, device, scaler, epoch, logger)
            test_acc, test_loss = test_epoch(model, test_loader, criterion, device, logger)
            
            # 在主進程上記錄到TensorBoard
            if local_rank == 0:
                # 記錄訓練和測試指標
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/test', test_acc, epoch)
                
                # 記錄學習率
                for i, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Learning_rate/group_{i}', param_group['lr'], epoch)
            
            if test_acc > best_acc:
                best_acc = test_acc
                if local_rank == 0:
                    torch.save(model.state_dict(), 'swin_v2_model_test.pth')
                    logger.info(f"New best model saved with accuracy: {test_acc:.2f}%")
    
    except Exception as e:
        import traceback
        print(f"Error in main process: {e}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        if torch.distributed.is_initialized():
            destroy_process_group()
