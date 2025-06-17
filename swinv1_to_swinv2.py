import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import timm
from datetime import datetime
import os

# ...existing code...

def create_model(num_classes=101):
    """創建不使用預訓練權重的 Swin Transformer V2 模型"""
    model = timm.create_model(
        'swinv2_base_window16_256',
        pretrained=False,  # 移除預訓練權重
        num_classes=num_classes
    )
    return model

def train_model():
    # 設置 tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/swinv2_food101_scratch_{timestamp}'
    writer = SummaryWriter(log_dir)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 創建模型
    model = create_model(num_classes=101).to(device)
    
    # 調整優化器參數 (無預訓練模型需要較低的學習率)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 降低學習率
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    # 使用 Cosine Annealing LR Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=100,  # 總訓練 epochs
        eta_min=1e-6  # 最小學習率
    )
    
    # 損失函數
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 訓練參數調整
    num_epochs = 100  # 增加訓練輪數
    best_acc = 0.0
    
    # ...existing data loading code...
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪 (對從頭訓練很重要)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # 記錄到 tensorboard
            if batch_idx % 100 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Train_Batch', loss.item(), step)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], step)
        
        # 計算訓練準確率
        train_acc = 100. * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        val_acc = 100. * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新學習率
        scheduler.step()
        
        # 記錄到 tensorboard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate_Epoch', scheduler.get_last_lr()[0], epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {scheduler.get_last_lr()[0]:.2e}')
        print('-' * 50)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, 'best_model_scratch.pth')
            print(f'New best model saved with validation accuracy: {best_acc:.2f}%')
    
    writer.close()
    print(f'Training completed. Best validation accuracy: {best_acc:.2f}%')
    print(f'Tensorboard logs saved to: {log_dir}')

if __name__ == '__main__':
    train_model()