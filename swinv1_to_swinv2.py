"""
swinv1_to_swinv2.py - 根據 swinv1.py 的設置，優化 main.py 中的 SwinV2 模型

此腳本創建了一個新的訓練文件，結合了 swinv1.py 的簡單高效設計和 main.py 中的 SwinV2 模型。
這個實現主要採用了以下優化策略：

1. 使用與 swinv1.py 相同的圖像大小 (224x224)
2. 採用簡單的數據增強策略，避免過度複雜的增強
3. 使用更大的批次大小 (32)，增加訓練穩定性
4. 使用 timm 庫加載 Swin V2 預訓練模型
5. 簡化優化器和學習率設置
6. 減少分佈式訓練中的複雜性
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
import timm
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.distributed import init_process_group, destroy_process_group
import argparse
import logging
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter # <--- 新增 TensorBoard 匯入

# 更新的標籤編碼器
class Label_encoder:
    def __init__(self, labels):
        self.labels = {label: idx for idx, label in enumerate(labels)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels.get(label)


# 食物數據集類
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


# 準備數據框架
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


# 訓練和測試函數
def train_epoch(model, dataloader, optimizer, criterion, device, writer, epoch): # <--- 新增 writer 和 epoch 參數
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = len(dataloader)

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 可選：記錄每個 batch 的 loss (如果需要更細緻的監控)
        # if writer:
        #     writer.add_scalar('Loss/train_batch', loss.item(), epoch * batch_count + batch_idx)

    avg_loss = total_loss / batch_count
    accuracy = 100. * correct / total
    print(f"Train Loss: {avg_loss:.3f} | Train Accuracy: {accuracy:.2f}%")
    
    if writer: # 確保 writer 只在主進程中寫入
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', accuracy, epoch)
        
    return accuracy


def test_epoch(model, dataloader, criterion, device, writer, epoch): # <--- 新增 writer 和 epoch 參數
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = len(dataloader)

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / batch_count
    accuracy = 100. * correct / total
    print(f"Test Loss: {avg_loss:.3f} | Test Accuracy: {accuracy:.2f}%")

    if writer: # 確保 writer 只在主進程中寫入
        writer.add_scalar('Loss/test_epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/test_epoch', accuracy, epoch)
        
    return accuracy


# 生成 Swin Transformer V2 的 CAM
def generate_cam_swin_v2(model, input_tensor, class_idx=None):
    # 確保輸入張量在正確的設備上
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    gradients = []
    activations = []
    
    # 註冊鉤子以獲取梯度和激活值
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # 找到最後一個階段的最後一個塊
    if hasattr(model, 'module'):
        target_layer = model.module.backbone.stages[-1].blocks[-1]
    else:
        target_layer = model.backbone.stages[-1].blocks[-1]
    
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)
    
    # 前向傳播
    model.eval()
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    # 反向傳播
    model.zero_grad()
    output[:, class_idx].backward()
    
    # 獲取梯度和激活
    grad = gradients[0]
    act = activations[0]
    
    # 計算權重
    weights = grad.mean(dim=(2, 3), keepdim=True)
    
    # 生成 CAM
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # 清理鉤子
    handle_fwd.remove()
    handle_bwd.remove()
    
    return cam[0, 0].detach().cpu().numpy()


def visualize_cam(image, cam):
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5) * 255
    img_np = img_np.astype(np.uint8)

    cam_resized = cv2.resize(cam, (224, 224))  # 調整為224x224大小
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, cam_heatmap, 0.4, 0)
    return overlay


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Swin V2 model with simplified settings and TensorBoard logging') # 更新描述
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--image_size', type=int, default=192, help='image size (192 for SwinV2 with window12)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--data_root', type=str, default='food-101', help='data root directory')
    parser.add_argument('--use_v2', action='store_true', help='use Swin V2 instead of V1')
    parser.add_argument('--distributed', action='store_true', help='use distributed training')
    args = parser.parse_args()
    
    # 檢查是否使用分散式訓練
    use_distributed = args.distributed
    local_rank = 0
    is_main_process = True # 預設為主進程

    # 檢查環境變數，決定是否使用分散式訓練
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        use_distributed = True
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        is_main_process = (local_rank == 0)
        print(f"分散式訓練已初始化，local_rank: {local_rank}")
    else:
        print("使用單一GPU訓練模式")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {device}")

    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size
    IMAGE_ROOT = f"{args.data_root}/images"
    TRAIN_FILE = f"{args.data_root}/meta/train.txt"
    TEST_FILE = f"{args.data_root}/meta/test.txt"

    LABELS =  [
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
        'cheese_plate',
        'cheesecake',
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

    # 使用更強的資料增強設置來對抗過度擬合
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)), # 調整以進行更積極的裁剪
        transforms.RandomRotation(20),  # 增加旋轉角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15), # 增強顏色抖動
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    encoder = Label_encoder(LABELS)

    train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder)
    test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)

    train_dataset = Food101Dataset(train_df, encoder, transform)
    test_dataset = Food101Dataset(test_df, encoder, transform)

    if use_distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
        device = torch.device(f"cuda:{local_rank}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = args.epochs      # 選擇使用 Swin V1 或 V2
    if args.use_v2:
        print("使用 Swin Transformer V2 模型")
        # 使用 timm 加載 Swin V2 模型
        try:
            # 使用正確的模型名稱 (根據 timm 提供的模型列表)
            # 加入 drop_rate 和 drop_path_rate 以減少過度擬合
            model = timm.create_model('swinv2_base_window12_192', pretrained=True, num_classes=len(LABELS), drop_rate=0.2, drop_path_rate=0.2)
            print("成功加載 swinv2_base_window12_192 預訓練模型 (帶有 dropout)")
        except Exception as e:
            print(f"無法加載 Swin V2 模型: {e}")
            print("嘗試使用 Swin V1 替代...")
            # 同樣為 Swin V1 加入 dropout
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=len(LABELS), drop_rate=0.2, drop_path_rate=0.2)
    else:
        print("使用 Swin Transformer V1 模型")
        # 為 Swin V1 加入 dropout
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=len(LABELS), drop_rate=0.2, drop_path_rate=0.2)
    
    model = model.to(device)
    
    if use_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        if args.use_v2:
            try:
                target_layer = model.module.layers[-1].blocks[-1].norm2
            except:
                target_layer = None
        else:
            target_layer = model.module.layers[-1].blocks[-1].norm2
    else:
        # 若只使用單一GPU，可以選擇使用 DataParallel 加速
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 個 GPU 運行 DataParallel")
            model = nn.DataParallel(model)
            if args.use_v2:
                try:
                    target_layer = model.module.layers[-1].blocks[-1].norm2
                except:
                    target_layer = None
            else:
                target_layer = model.module.layers[-1].blocks[-1].norm2
        else:
            if args.use_v2:
                try:
                    target_layer = model.layers[-1].blocks[-1].norm2
                except:
                    target_layer = None
            else:
                target_layer = model.layers[-1].blocks[-1].norm2

    # 使用簡單的優化器設置，並明確指定 weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01) # 明確加入 weight_decay
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 設置輸出目錄
    os.makedirs('outputs', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"outputs/train_log_{timestamp}.txt" # 將文字日誌也放到 outputs 資料夾
    
    # 初始化 TensorBoard SummaryWriter (僅在主進程)
    writer = None
    if is_main_process:
        tensorboard_log_dir = f"runs/swin_experiment_{timestamp}"
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f"TensorBoard 日誌將儲存於: {tensorboard_log_dir}")

    # 記錄訓練參數 (僅在主進程)
    if is_main_process:
        with open(log_file, 'w') as f:
            f.write(f"訓練開始時間: {timestamp}\n")
            f.write(f"模型: {'Swin V2' if args.use_v2 else 'Swin V1'}\n")
            f.write(f"圖像大小: {IMAGE_SIZE}\n")
            f.write(f"批次大小: {BATCH_SIZE}\n")
            f.write(f"輪數: {num_epochs}\n")
            f.write(f"分散式訓練: {use_distributed}\n")
            f.write("\n")

    # 嘗試將模型圖寫入 TensorBoard (僅在主進程)
    if is_main_process and writer:
        try:
            # 從 DataLoader 取一個批次的資料作為範例輸入
            sample_inputs, _ = next(iter(train_loader))
            sample_inputs = sample_inputs.to(device)
            # 如果模型是 DDP 或 DP，需要取 .module
            model_to_log = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
            writer.add_graph(model_to_log, sample_inputs)
            print("模型圖已成功寫入 TensorBoard。")
        except Exception as e:
            print(f"寫入模型圖到 TensorBoard 時發生錯誤: {e}")
            print("提示: 這可能是由於模型包含 TensorBoard 不支援的操作，或者範例輸入與模型期望不符。")


    best_acc = 0
    for epoch in range(num_epochs):
        if is_main_process: # 主進程打印 epoch 資訊
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        if use_distributed:
            train_sampler.set_epoch(epoch) # 設定 DistributedSampler 的 epoch
        
        # 將 writer 和 epoch 傳遞給訓練和測試函數
        train_epoch(model, train_loader, optimizer, criterion, device, writer if is_main_process else None, epoch)
        test_acc = test_epoch(model, test_loader, criterion, device, writer if is_main_process else None, epoch)
        
        # 更新學習率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄每個epoch的結果 (僅在主進程)
        if is_main_process:
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1}/{num_epochs}, 測試準確率: {test_acc:.2f}%, 學習率: {current_lr:.6f}\n")
            if writer:
                writer.add_scalar('LearningRate/epoch', current_lr, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            if is_main_process: # 僅主進程儲存模型和打印訊息
                model_save_path = f"outputs/{'swinv2' if args.use_v2 else 'swinv1'}_food101_best.pth"
                if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)
                print(f"模型已保存，準確率: {best_acc:.2f}%")
                
                # 記錄最佳模型資訊
                with open(log_file, 'a') as f:
                    f.write(f"\n最佳模型：準確率 {best_acc:.2f}%，儲存於 {model_save_path}\n")

    if is_main_process: # 主進程打印最終結果和關閉 writer
        print(f"\n訓練完成！最佳準確率：{best_acc:.2f}%")
        if writer:
            writer.close()
    
    # 清理分散式訓練資源
    if use_distributed and torch.distributed.is_initialized():
        destroy_process_group()
