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
import torchvision # <--- 新增 torchvision 匯入

# 新增 log 函數
def log_message(message, is_main_process_flag):
    if is_main_process_flag:
        print(message)

# 新的解碼器和無監督模型定義
class SimpleDecoder(nn.Module):
    def __init__(self, feature_dim, output_channels=3, image_size=224):
        super().__init__()
        self.image_size = image_size
        self.output_channels = output_channels
        initial_hw = image_size // 32  # Assuming 5 stages of x2 upsampling (2^5 = 32)
        self.initial_hw = initial_hw

        # 根據 feature_dim 決定解碼器的基礎隱藏通道數
        # 參考使用者建議: hidden_channels = 512 if feature_dim >= 768 else 256
        if feature_dim >= 768:
            self.base_hidden_channels = 512
        else:
            self.base_hidden_channels = 256
        
        self.fc = nn.Linear(feature_dim, self.base_hidden_channels * initial_hw * initial_hw)

        # 動態定義解碼器層的通道大小
        ch1 = self.base_hidden_channels  # e.g., 512 or 256
        ch2 = max(ch1 // 2, 32)         # e.g., 256 or 128
        ch3 = max(ch2 // 2, 32)         # e.g., 128 or 64
        ch4 = max(ch3 // 2, 32)         # e.g., 64 or 32
        # 確保最後一個卷積層之前的通道數合理 (例如，至少16)
        ch5 = max(ch4 // 2, 16)         # e.g., 32 or 16

        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose2d(ch1, ch2, kernel_size=4, stride=2, padding=1), # initial_hw -> 2*initial_hw
            nn.ReLU(True),
            nn.ConvTranspose2d(ch2, ch3, kernel_size=4, stride=2, padding=1), # 2*initial_hw -> 4*initial_hw
            nn.ReLU(True),
            nn.ConvTranspose2d(ch3, ch4, kernel_size=4, stride=2, padding=1),  # 4*initial_hw -> 8*initial_hw
            nn.ReLU(True),
            nn.ConvTranspose2d(ch4, ch5, kernel_size=4, stride=2, padding=1),   # 8*initial_hw -> 16*initial_hw
            nn.ReLU(True),
            nn.ConvTranspose2d(ch5, output_channels, kernel_size=4, stride=2, padding=1), # 16*initial_hw -> 32*initial_hw (image_size)
            nn.Sigmoid() # 輸出範圍 [0, 1]
        )

    def forward(self, x):
        # 假設 x 的維度是 [batch_size, feature_dim]
        if x.ndim == 4: # 例如，如果編碼器輸出 [B, C, H, W] 形式的特徵圖，先進行全局平均池化或展平
            x = x.mean(dim=[2,3]) # 簡易全局平均池化
        
        x = self.fc(x)
        x = x.view(x.size(0), self.base_hidden_channels, self.initial_hw, self.initial_hw) # 重塑為 (B, base_hidden_channels, initial_hw, initial_hw)
        reconstructed = self.decoder_layers(x)
        return reconstructed

class UnsupervisedSwin(nn.Module):
    def __init__(self, encoder_model_name, image_size, pretrained=True, in_chans=3):
        super().__init__()
        self.image_size = image_size
        # 載入 Swin Transformer 作為編碼器，num_classes=0 表示我們想要特徵而不是分類 logits
        self.encoder = timm.create_model(
            encoder_model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            img_size=image_size # <--- Pass the image_size to configure the model's expected input size
        )
        
        # 動態獲取編碼器的輸出特徵維度
        encoder_feature_dim = 0
        if hasattr(self.encoder, 'num_features') and self.encoder.num_features > 0:
            encoder_feature_dim = self.encoder.num_features
        elif hasattr(self.encoder, 'head') and hasattr(self.encoder.head, 'in_features') and self.encoder.head.in_features > 0:
            encoder_feature_dim = self.encoder.head.in_features
        else:
            # 嘗試從模型的最後一個線性層（如果存在且看起來像分類頭）獲取
            # 這部分比較tricky，因為timm模型結構多樣
            # 作為後備，使用一個基於模型名稱的常見值
            if 'base' in encoder_model_name:
                encoder_feature_dim = 1024
            elif 'large' in encoder_model_name:
                encoder_feature_dim = 1536
            elif 'tiny' in encoder_model_name:
                encoder_feature_dim = 768
            else: # 預設值
                encoder_feature_dim = 1024 
            print(f"警告: 無法精確自動檢測 {encoder_model_name} 的特徵維度。")
            print(f"基於模型名稱，假設特徵維度為: {encoder_feature_dim}")
            print(f"如果模型是 'swinv2_base_window12_192' 或 'swin_base_patch4_window7_224', 通常是 1024。")
            print(f"請檢查並確認此維度是否正確，否則解碼器可能無法正常工作。")


        self.decoder = SimpleDecoder(encoder_feature_dim, output_channels=in_chans, image_size=image_size)

    def forward(self, x):
        # x 的形狀應為 (B, C, H, W)
        features = self.encoder(x) # timm 模型在 num_classes=0 時，通常返回 (B, feature_dim) 的特徵
        reconstructed_image = self.decoder(features)
        return reconstructed_image

# 食物數據集類
class Food101Dataset(Dataset):
    def __init__(self, dataframe, transform=None): # 移除 encoder
        self.dataframe = dataframe
        # self.encoder = encoder # 移除
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        # label = self.dataframe.iloc[idx]['label'] # 移除
        # label = self.encoder.get_idx(label) # 移除

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image # 只返回圖像

# 準備數據框架
def prepare_dataframe(file_path, image_root): # 移除 encoder
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        # 無監督學習不需要檢查標籤是否存在於 encoder.labels 中
        category, img_name = line.split('/')
        data.append({
            # 'label': category, # 移除
            'path': f"{image_root}/{category}/{img_name}.jpg"
        })

    df = pd.DataFrame(data)
    return shuffle(df)


# 訓練和測試函數
def train_epoch(model, dataloader, optimizer, criterion, device, writer, epoch, image_size): # <--- 新增 image_size
    model.train()
    total_loss = 0
    batch_count = len(dataloader)

    for batch_idx, inputs in enumerate(tqdm(dataloader, desc="Training")): # <--- dataloader 只返回 inputs
        inputs = inputs.to(device)
        optimizer.zero_grad()
        reconstructed_outputs = model(inputs)
        loss = criterion(reconstructed_outputs, inputs) # <--- 比較重建輸出和原始輸入
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if writer and batch_idx % 100 == 0: # 每 100 個 batch 記錄一次重建圖像
            if inputs.size(0) >= 4 and reconstructed_outputs.size(0) >= 4: # 確保至少有4張圖像
                img_grid_input = torchvision.utils.make_grid(inputs[:4].cpu(), normalize=True)
                img_grid_reconstruction = torchvision.utils.make_grid(reconstructed_outputs[:4].cpu(), normalize=True)
                writer.add_image(f'Epoch_{epoch}/Input_Images', img_grid_input, global_step=batch_idx)
                writer.add_image(f'Epoch_{epoch}/Reconstructed_Images', img_grid_reconstruction, global_step=batch_idx)


    avg_loss = total_loss / batch_count
    # 使用 log_message 函數 (假設 is_main_process 在此作用域不可見，因此直接 print，稍後在主流程中修正)
    # print(f"Train Loss: {avg_loss:.4f}") # <--- 更新打印信息
    
    if writer: # 確保 writer 只在主進程中寫入
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        
    return avg_loss # <--- 返回平均損失


def test_epoch(model, dataloader, criterion, device, writer, epoch, image_size): # <--- 新增 image_size
    model.eval()
    total_loss = 0
    batch_count = len(dataloader)

    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(dataloader, desc="Testing")): # <--- dataloader 只返回 inputs
            inputs = inputs.to(device)
            reconstructed_outputs = model(inputs)
            loss = criterion(reconstructed_outputs, inputs) # <--- 比較重建輸出和原始輸入

            total_loss += loss.item()
            
            if writer and batch_idx == 0: # 只記錄第一個 test batch 的圖像
                if inputs.size(0) >= 4 and reconstructed_outputs.size(0) >= 4:
                    img_grid_input_test = torchvision.utils.make_grid(inputs[:4].cpu(), normalize=True)
                    img_grid_reconstruction_test = torchvision.utils.make_grid(reconstructed_outputs[:4].cpu(), normalize=True)
                    writer.add_image(f'Epoch_{epoch}/Test_Input_Images', img_grid_input_test, global_step=epoch) # global_step 使用 epoch
                    writer.add_image(f'Epoch_{epoch}/Test_Reconstructed_Images', img_grid_reconstruction_test, global_step=epoch)


    avg_loss = total_loss / batch_count
    # 使用 log_message 函數 (假設 is_main_process 在此作用域不可見，因此直接 print，稍後在主流程中修正)
    # print(f"Test Loss: {avg_loss:.4f}") # <--- 更新打印信息

    if writer: # 確保 writer 只在主進程中寫入
        writer.add_scalar('Loss/test_epoch', avg_loss, epoch)
        
    return avg_loss # <--- 返回平均損失


# 生成 Swin Transformer V2 的 CAM (註釋掉，因為 CAM 主要用於分類)
# def generate_cam_swin_v2(model, input_tensor, class_idx=None):
#     # ... (原有 CAM 代碼) ...
#     pass

# def visualize_cam(image, cam):
#     # ... (原有 CAM 代碼) ...
#     pass


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Unsupervised Swin Transformer (V1/V2) with TensorBoard logging') # 更新描述
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--image_size', type=int, default=224, help='image size (e.g., 224 for SwinV1, 192 or 256 for SwinV2 variants)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--data_root', type=str, default='food-101', help='data root directory')
    parser.add_argument('--use_v2', action='store_true', help='use Swin V2 encoder instead of V1')
    parser.add_argument('--distributed', action='store_true', help='use distributed training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for AdamW')

    args = parser.parse_args()
    
    # 檢查是否使用分散式訓練
    use_distributed = args.distributed
    local_rank = 0
    is_main_process = True

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        use_distributed = True
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        is_main_process = (local_rank == 0)
        log_message(f"分散式訓練已初始化，local_rank: {local_rank}", is_main_process)
    else:
        is_main_process = True
        log_message("使用單一GPU訓練模式", is_main_process)
        # device variable will be set later, after IMAGE_SIZE is finalized

    # --- 模型和圖像大小選擇邏輯 ---
    IMAGE_SIZE_FROM_ARGS = args.image_size # 保存用戶原始請求的圖像大小
    effective_image_size = IMAGE_SIZE_FROM_ARGS

    if args.use_v2:
        if IMAGE_SIZE_FROM_ARGS == 192:
            encoder_model_name = 'swinv2_base_window12_192'
            effective_image_size = 192
        elif IMAGE_SIZE_FROM_ARGS == 256:
            encoder_model_name = 'swinv2_base_window16_256'
            effective_image_size = 256
        else:
            encoder_model_name = 'swinv2_base_window12_192' # 預設 SwinV2 模型
            native_size = 192
            if IMAGE_SIZE_FROM_ARGS != native_size:
                log_message(f"警告: 您請求的圖像大小 {IMAGE_SIZE_FROM_ARGS}x{IMAGE_SIZE_FROM_ARGS} 與所選 SwinV2 模型 '{encoder_model_name}' 的原生輸入尺寸 {native_size}x{native_size} 不匹配。", is_main_process)
                log_message(f"將使用模型原生輸入尺寸 {native_size}x{native_size} 進行訓練和數據處理。", is_main_process)
                effective_image_size = native_size
        log_message(f"使用 Swin Transformer V2 ({encoder_model_name}) 作為編碼器, 有效輸入尺寸: {effective_image_size}x{effective_image_size}", is_main_process)
    else: # SwinV1
        encoder_model_name = 'swin_base_patch4_window7_224' # 經典的 SwinV1
        native_size = 224
        if IMAGE_SIZE_FROM_ARGS != native_size:
            log_message(f"警告: SwinV1 模型 '{encoder_model_name}' 通常使用 {native_size}x{native_size}。您的設定為 {IMAGE_SIZE_FROM_ARGS}x{IMAGE_SIZE_FROM_ARGS}。", is_main_process)
            log_message(f"將使用模型原生輸入尺寸 {native_size}x{native_size} 進行訓練和數據處理。", is_main_process)
            effective_image_size = native_size
        log_message(f"使用 Swin Transformer V1 ({encoder_model_name}) 作為編碼器, 有效輸入尺寸: {effective_image_size}x{effective_image_size}", is_main_process)

    # 更新 args.image_size 以反映將實際使用的尺寸，或使用新的 effective_image_size 變數
    # 為了減少對後續代碼的更改，我們直接更新 IMAGE_SIZE 全局變數的概念
    IMAGE_SIZE = effective_image_size
    # --- 結束模型和圖像大小選擇邏輯 ---

    BATCH_SIZE = args.batch_size
    # IMAGE_SIZE is now set above
    IMAGE_ROOT = f"{args.data_root}/images"
    TRAIN_FILE = f"{args.data_root}/meta/train.txt"
    TEST_FILE = f"{args.data_root}/meta/test.txt"
    
    # 設定 device (在 IMAGE_SIZE 確定後，以防未來有依賴 IMAGE_SIZE 的 device 選擇)
    if not use_distributed: # 如果是單 GPU 模式，在這裡設定 device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_message(f"使用設備: {device}", is_main_process)
    else: # 分散式模式下，device 在前面已經根據 local_rank 設定
        device = torch.device(f"cuda:{local_rank}")


    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT)
    test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT)

    train_dataset = Food101Dataset(train_df, transform)
    test_dataset = Food101Dataset(test_df, transform)

    if use_distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=32, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=32, pin_memory=True)
        # device is already set for distributed
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32, pin_memory=True)
        # device is already set for non-distributed

    num_epochs = args.epochs
    
    try:
        # IMAGE_SIZE 現在是 effective_image_size
        model = UnsupervisedSwin(encoder_model_name=encoder_model_name, image_size=IMAGE_SIZE, pretrained=True)
        log_message(f"成功加載無監督模型，編碼器: {encoder_model_name}, 輸入尺寸 {IMAGE_SIZE}x{IMAGE_SIZE}", is_main_process)
    except Exception as e:
        log_message(f"無法加載無監督 Swin 模型 ({encoder_model_name} for image size {IMAGE_SIZE}): {e}", is_main_process)
        log_message("請檢查 timm 是否已安裝、模型名稱是否正確，以及圖像大小是否與模型兼容。", is_main_process)
        exit()

    model = model.to(device)
    
    if use_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) # find_unused_parameters 可能需要
    elif torch.cuda.device_count() > 1:
        log_message(f"使用 {torch.cuda.device_count()} 個 GPU 運行 DataParallel", is_main_process)
        model = nn.DataParallel(model)

    # 使用 MSELoss 進行重建
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        log_message(f"TensorBoard 日誌將儲存於: {tensorboard_log_dir}", is_main_process)

    # 記錄訓練參數 (僅在主進程)
    if is_main_process:
        with open(log_file, 'w') as f:
            f.write(f"訓練開始時間: {timestamp}\n")
            f.write(f"模型架構: UnsupervisedSwin\n")
            f.write(f"編碼器: {encoder_model_name} (V2: {args.use_v2})\n")
            f.write(f"圖像大小: {IMAGE_SIZE}\n")
            f.write(f"批次大小: {BATCH_SIZE}\n")
            f.write(f"輪數: {num_epochs}\n")
            f.write(f"學習率: {args.learning_rate}\n")
            f.write(f"權重衰減: {args.weight_decay}\n")
            f.write(f"分散式訓練: {use_distributed}\n")
            f.write("\n")

    # 嘗試將模型圖寫入 TensorBoard (僅在主進程)
    if is_main_process and writer:
        try:
            # 從 DataLoader 取一個批次的資料作為範例輸入
            sample_inputs = next(iter(train_loader)) # DataLoader 現在只返回 inputs
            sample_inputs = sample_inputs.to(device)
            # 如果模型是 DDP 或 DP，需要取 .module
            model_to_log = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
            writer.add_graph(model_to_log, sample_inputs)
            log_message("模型圖已成功寫入 TensorBoard。", is_main_process)
        except Exception as e:
            log_message(f"寫入模型圖到 TensorBoard 時發生錯誤: {e}", is_main_process)
            # 提示可以更具體
            log_message("提示: 確保模型定義和範例輸入兼容。對於複雜模型或特定操作(如 torch.jit.trace)，add_graph 可能會遇到問題。可以考慮註釋掉此部分或使用 torch.jit.trace。", is_main_process)


    best_loss = float('inf') # <--- 初始化為無限大，因為我們要找最小損失
    for epoch in range(num_epochs):
        if is_main_process: # 主進程打印 epoch 資訊
            log_message(f"\nEpoch {epoch + 1}/{num_epochs}", is_main_process)
        
        if use_distributed:
            train_sampler.set_epoch(epoch) # 設定 DistributedSampler 的 epoch
        
        # 將 writer 和 epoch 傳遞給訓練和測試函數
        # 也將 is_main_process 傳遞給 train_epoch 和 test_epoch 以便它們內部可以使用 log_message
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, writer if is_main_process else None, epoch, IMAGE_SIZE)
        if is_main_process: # 僅主進程打印訓練損失
            log_message(f"Train Loss: {train_loss:.4f}", is_main_process)

        test_loss = test_epoch(model, test_loader, criterion, device, writer if is_main_process else None, epoch, IMAGE_SIZE)
        if is_main_process: # 僅主進程打印測試損失
            log_message(f"Test Loss: {test_loss:.4f}", is_main_process)
        
        # 更新學習率
        scheduler.step() # 確保在每個 epoch 後調用
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄每個epoch的結果 (僅在主進程)
        if is_main_process:
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1}/{num_epochs}, 測試損失: {test_loss:.4f}, 學習率: {current_lr:.6f}\n") # <--- 改為記錄損失
            if writer:
                writer.add_scalar('LearningRate/epoch', current_lr, epoch)

        if test_loss < best_loss: # <--- 如果當前測試損失更低
            best_loss = test_loss
            if is_main_process: # 僅主進程儲存模型和打印訊息
                model_save_path = f"outputs/{'unsupervised_swinv2' if args.use_v2 else 'unsupervised_swinv1'}_food101_best_loss.pth"
                # 儲存整個模型或 state_dict
                # 如果是 DDP 或 DP，儲存 model.module.state_dict()
                model_to_save = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
                torch.save(model_to_save.state_dict(), model_save_path)
                log_message(f"模型已保存，最低測試損失: {best_loss:.4f}", is_main_process) # <--- 更新打印信息
                
                # 記錄最佳模型資訊
                with open(log_file, 'a') as f:
                    f.write(f"\n最佳模型：最低測試損失 {best_loss:.4f}，儲存於 {model_save_path}\n")

    if is_main_process: # 主進程打印最終結果和關閉 writer
        log_message(f"\n訓練完成！最低測試損失：{best_loss:.4f}", is_main_process) # <--- 更新打印信息
        if writer:
            writer.close()
    
    # 清理分散式訓練資源
    if use_distributed and torch.distributed.is_initialized():
        destroy_process_group()
