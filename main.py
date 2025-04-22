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
import argparse
import requests
import shutil
from pathlib import Path

# Import the SwinV2 classifier
from swin_transformer_v2_classifier import swin_transformer_v2_base_classifier

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


def test_epoch(model, dataloader, criterion, device):
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
    return accuracy


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


def train_epoch_amp(model, dataloader, optimizer, scheduler, criterion, device, scaler):
    """
    使用混合精度訓練的訓練函數，可加速訓練並降低記憶體使用量
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # 使用混合精度前向傳播
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 使用GradScaler處理反向傳播
        scaler.scale(loss).backward()
        
        # 在更新前應用梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # 更新參數
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Train Loss: {total_loss / len(dataloader):.3f} | Train Accuracy: {accuracy:.2f}%")
    
    # 更新調度器
    scheduler.step(epoch=None)  # CosineLRScheduler從timm不需要傳入epoch參數
    
    return accuracy


def download_pretrained_model(url, save_path):
    """
    Download pretrained model from URL
    
    :param url: URL to download the model from
    :param save_path: Path to save the downloaded model
    :return: Path to the downloaded model
    """
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    if os.path.exists(save_path):
        print(f"Pretrained model already exists at {save_path}")
        return save_path
    
    print(f"Downloading pretrained model from {url}")
    
    # Stream the download to avoid loading the whole file into memory
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"Downloaded pretrained model to {save_path}")
    return save_path


def freeze_backbone_layers(model, num_layers_to_freeze=None):
    """
    Freeze layers in the backbone of the model
    
    :param model: The model to freeze layers in
    :param num_layers_to_freeze: Number of layers to freeze (None freezes all backbone)
    """
    # For DDP model, access the module attribute
    if hasattr(model, 'module'):
        backbone = model.module.backbone
    else:
        backbone = model.backbone
    
    # Count total layers in backbone
    total_stages = len(backbone.stages)
    
    if num_layers_to_freeze is None or num_layers_to_freeze >= total_stages:
        # Freeze all backbone layers
        for param in backbone.parameters():
            param.requires_grad = False
        print(f"Froze all backbone layers")
    else:
        # Freeze specific number of stages from the beginning
        for i in range(num_layers_to_freeze):
            for param in backbone.stages[i].parameters():
                param.requires_grad = False
        print(f"Froze first {num_layers_to_freeze} stages of the backbone")
    
    # Keep the classification head trainable
    if hasattr(model, 'module'):
        for param in model.module.head.parameters():
            param.requires_grad = True
    else:
        for param in model.head.parameters():
            param.requires_grad = True


def unfreeze_all_layers(model):
    """
    Unfreeze all layers in the model
    
    :param model: The model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    print("Unfroze all layers in the model")


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train SwinV2 on Food-101 dataset')
    
    # Dataset arguments
    parser.add_argument('--data-dir', type=str, default='food-101', 
                        help='Path to the Food-101 dataset')
    parser.add_argument('--image-size', type=int, default=224, 
                        help='Input image size')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='base', choices=['tiny', 'small', 'base', 'large'],
                        help='SwinV2 model type')
    parser.add_argument('--pretrained', action='store_true', 
                        help='Use pretrained ImageNet weights')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Path to pretrained weights file')
    parser.add_argument('--imagenet-pretrained-url', type=str, 
                        default='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
                        help='URL to download ImageNet pretrained weights')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=110, 
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, 
                        help='Base learning rate')
    parser.add_argument('--backbone-lr', type=float, default=1e-5, 
                        help='Learning rate for backbone')
    parser.add_argument('--weight-decay', type=float, default=0.05, 
                        help='Weight decay')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision training')
    
    # Transfer learning arguments
    parser.add_argument('--freeze-backbone', action='store_true', 
                        help='Freeze backbone layers')
    parser.add_argument('--freeze-layers', type=int, default=None,
                        help='Number of backbone layers to freeze')
    parser.add_argument('--progressive-unfreeze', action='store_true', 
                        help='Progressively unfreeze layers during training')
    parser.add_argument('--unfreeze-epoch', type=int, default=50,
                        help='Epoch to unfreeze all layers if progressive unfreeze is enabled')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Path to save outputs')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


# Main Program
if __name__ == "__main__":
    # 解析命令行參數
    args = parse_args()
    
    # 設置設備
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 初始化分散式訓練
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    
    print(f"Process initialized with rank {local_rank}")
    
    # 設定目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定資料集參數
    IMAGE_ROOT = os.path.join(args.data_dir, "images")
    TRAIN_FILE = os.path.join(args.data_dir, "meta/train.txt")
    TEST_FILE = os.path.join(args.data_dir, "meta/test.txt")
    
    # 定義 Food-101 類別標籤
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
    
    # 數據轉換
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 準備資料集
    encoder = Label_encoder(LABELS)
    train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder)
    test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)
    
    train_dataset = Food101Dataset(train_df, encoder, transform_train)
    test_dataset = Food101Dataset(test_df, encoder, transform_test)
    
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler)
    
    # 下載預訓練模型（如果需要）
    pretrained_path = None
    if args.pretrained:
        if args.pretrained_path:
            pretrained_path = args.pretrained_path
        else:
            pretrained_dir = os.path.join(args.output_dir, "pretrained")
            os.makedirs(pretrained_dir, exist_ok=True)
            pretrained_path = os.path.join(pretrained_dir, "swinv2_imagenet_pretrained.pth")
            download_pretrained_model(args.imagenet_pretrained_url, pretrained_path)
    
    # 初始化模型
    model = swin_transformer_v2_base_classifier(
        input_resolution=(args.image_size, args.image_size),
        window_size=7,
        num_classes=len(LABELS),
        use_checkpoint=True,
        pretrained_path=pretrained_path  # 使用預訓練權重
    )
    
    # 將模型移至 GPU
    model = model.to(device)
    
    # 應用層凍結（如果需要）
    if args.freeze_backbone:
        freeze_backbone_layers(model, args.freeze_layers)
    
    # 分散式數據並行
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model._set_static_graph()
    
    # 使用分層學習率，為骨幹和分類頭設置不同的學習率
    parameters = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': args.backbone_lr},
        {'params': [p for n, p in model.named_parameters() if 'head' in n and p.requires_grad], 'lr': args.lr}
    ]
    
    optimizer = optim.AdamW(
        parameters,
        weight_decay=args.weight_decay
    )
    
    # 標籤平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 學習率調度器
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=1e-6,
        warmup_t=5,
        warmup_lr_init=1e-7
    )
    
    # 初始化混合精度訓練的 scaler
    scaler = GradScaler() if args.amp else None
    
    # 訓練循環
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 設置采樣器 epoch
        train_sampler.set_epoch(epoch)
        
        # 逐步解凍（如果啟用）
        if args.progressive_unfreeze and epoch == args.unfreeze_epoch:
            if local_rank == 0:
                print(f"Unfreezing all layers at epoch {epoch + 1}")
            unfreeze_all_layers(model)
        
        # 訓練和測試
        if args.amp:
            train_acc = train_epoch_amp(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        else:
            train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            
        test_acc = test_epoch(model, test_loader, criterion, device)
        
        # 保存最佳模型
        if test_acc > best_acc and local_rank == 0:
            best_acc = test_acc
            model_save_path = os.path.join(args.output_dir, f"swinv2_food101_best.pth")
            if hasattr(model, 'module'):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }, model_save_path)
            print(f"Saved best model with accuracy {best_acc:.2f}% to {model_save_path}")
        
        # 定期保存檢查點
        if (epoch + 1) % args.save_interval == 0 and local_rank == 0:
            checkpoint_path = os.path.join(args.output_dir, f"swinv2_food101_checkpoint_{epoch + 1}.pth")
            if hasattr(model, 'module'):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}")
    
    # 結束訓練
    if local_rank == 0:
        print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    
    # 清理
    if torch.distributed.is_initialized():
        destroy_process_group()
