import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import time
import cv2

# 導入專案模組
from swin_transformer_v2_classifier import swin_transformer_v2_base_classifier
from main import Label_encoder, Food101Dataset, generate_cam_swin_v2, visualize_cam, setup_logger, prepare_dataframe

def generate_cam_swin_v2_test(model, input_tensor, class_idx=None):
    """
    為Swin Transformer V2模型生成Class Activation Map，優化版本
    :param model: 直接載入的SwinV2模型（非DDP模型）
    :param input_tensor: 輸入圖像張量 [1, C, H, W]
    :param class_idx: 目標類別索引（若為None則使用預測類別）
    :return: CAM作為numpy陣列
    """
    # 確保輸入張量在正確的裝置上
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 追蹤梯度和激活值
    gradients = []
    activations = []
    
    # 註冊鉤子以獲取梯度和激活值
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # 尋找適合的層來註冊鉤子 - 選擇最後的stage和block
    try:
        # 嘗試找到模型中的最後一個stage的最後一個block
        target_layer = model.backbone.stages[-1].blocks[-1]
        print(f"成功定位到目標層: {target_layer.__class__.__name__}")
    except (AttributeError, IndexError) as e:
        print(f"標準方法無法找到目標層: {e}")
        
        # 第一個後備方法：遍歷模型尋找任何包含"blocks"的最後層
        target_layer = None
        for name, module in model.named_modules():
            if "stages" in name and "blocks" in name:
                target_layer = module
                print(f"使用後備方法找到層: {name}")
        
        # 第二個後備方法：尋找任何卷積層或注意力層
        if target_layer is None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or "attn" in name.lower():
                    target_layer = module
                    print(f"使用卷積/注意力層作為目標: {name}")
                    break
        
        if target_layer is None:
            print("無法找到合適的層用於生成CAM，返回空熱力圖")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
    
    # 註冊鉤子到目標層
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)
    
    try:
        # 前向傳播
        model.eval()
        with torch.set_grad_enabled(True):
            # 確保輸入張量需要梯度
            input_tensor.requires_grad_(True)
            
            # 執行前向傳播
            output = model(input_tensor)
            
            # 如果沒有指定類別，使用預測的類別
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            # 檢查輸出形狀的有效性
            if len(output.shape) <= 1 or output.shape[1] <= class_idx:
                print(f"輸出形狀不正確: {output.shape}, 類別索引: {class_idx}")
                return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
            
            # 反向傳播
            model.zero_grad()
            output_for_backward = output[0, class_idx]
            output_for_backward.backward(retain_graph=True)
            
            # 檢查是否成功計算梯度
            if len(gradients) == 0:
                print("未捕捉到梯度！嘗試使用替代方法...")
                
                # 嘗試直接計算輸入的梯度
                if input_tensor.grad is not None:
                    print("使用輸入梯度代替特徵圖梯度")
                    # 使用輸入梯度生成簡單的熱力圖
                    input_grad = input_tensor.grad[0].sum(dim=0)
                    input_grad = input_grad.abs()
                    input_grad = input_grad / (input_grad.max() + 1e-8)
                    return input_grad.cpu().numpy()
                return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
            
            if len(activations) == 0:
                print("未捕捉到特徵激活值！")
                return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
            
            # 提取梯度和激活值
            grad = gradients[0]
            act = activations[0]
            
            # 處理形狀問題，確保梯度和激活值有正確的形狀
            if grad.ndim != 4:
                print(f"調整梯度形狀 from {grad.shape}")
                if grad.ndim == 3:
                    grad = grad.unsqueeze(0)
                elif grad.ndim == 2:
                    grad = grad.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            if act.ndim != 4:
                print(f"調整激活值形狀 from {act.shape}")
                if act.ndim == 3:
                    act = act.unsqueeze(0)
                elif act.ndim == 2:
                    act = act.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            # 針對Transformer的特殊處理：如果激活值是3D的（例如[B, tokens, C]），
            # 則重塑為4D張量以適應空間區域
            if act.ndim == 3:
                B, N, C = act.shape
                H = W = int(math.sqrt(N))
                act = act.transpose(1, 2).view(B, C, H, W)
            
            # 計算權重 - 針對Transformer結構優化
            # GAP (Global Average Pooling) over the gradient
            weights = grad.mean(dim=(2, 3), keepdim=True)
            
            # 優化：標準化權重以突顯重要特徵
            weights = weights / (weights.norm(dim=1, keepdim=True) + 1e-5)
            
            # 生成CAM
            cam = (weights * act).sum(dim=1, keepdim=True)
            
            # 應用RELU來只保留正面貢獻
            cam = torch.relu(cam)
            
            # 歸一化CAM
            if cam.max() != 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # 優化：應用高斯平滑處理以減少噪點
            from kornia.filters import gaussian_blur2d
            cam = gaussian_blur2d(cam, kernel_size=(5, 5), sigma=(1.5, 1.5))
            
            # 優化：進行門限處理，抑制低值區域
            # 將小於平均值的區域設為0，以突顯顯著區域
            cam_mean = cam.mean()
            cam = torch.where(cam < cam_mean*0.8, torch.zeros_like(cam), cam)
            
            # 再次歸一化
            if cam.max() != 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # 返回CAM作為numpy數組
            return cam[0, 0].cpu().numpy()
                
    except Exception as e:
        print(f"生成CAM時發生錯誤: {e}")
        # 返回空熱力圖
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
    finally:
        # 清理鉤子
        handle_fwd.remove()
        handle_bwd.remove()

def main():
    """
    測試 Swin Transformer V2 模型的主函數
    """
    parser = argparse.ArgumentParser(description='測試 Swin Transformer V2 模型')
    parser.add_argument('--weights', type=str, default='swinv2_food101_best.pth', 
                         help='模型權重檔案路徑')
    parser.add_argument('--image_size', type=int, default=224, 
                         help='輸入圖像大小')
    parser.add_argument('--window_size', type=int, default=7, 
                         help='窗口大小')
    parser.add_argument('--batch_size', type=int, default=32, 
                         help='批次大小')
    parser.add_argument('--data_root', type=str, default='food-101', 
                         help='資料集根目錄')
    parser.add_argument('--visualize', action='store_true', 
                         help='是否生成 CAM 可視化圖像')
    parser.add_argument('--num_visualize', type=int, default=5, 
                         help='要可視化的樣本數量')
    args = parser.parse_args()
    
    # 設置裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # 設置日誌
    logger = setup_logger(0)
    logger.info("開始測試 Swin Transformer V2 模型")
    
    # 設置資料路徑
    IMAGE_ROOT = os.path.join(args.data_root, 'images')
    TEST_FILE = os.path.join(args.data_root, 'meta', 'test.txt')
    
    # 讀取標籤
    with open(os.path.join(args.data_root, 'meta', 'classes.txt'), 'r') as f:
        LABELS = [line.strip() for line in f.readlines()]
    
    # 初始化標籤編碼器
    encoder = Label_encoder(LABELS)
    
    # 設置測試集轉換
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 準備測試資料集
    logger.info("準備測試資料集...")
    test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)
    test_dataset = Food101Dataset(test_df, encoder, transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    # 載入模型
    logger.info(f"載入模型權重: {args.weights}")
    model = swin_transformer_v2_base_classifier(
        input_resolution=(args.image_size, args.image_size),
        window_size=args.window_size,
        num_classes=len(LABELS)
    )
    
    # 載入模型權重
    try:
        state_dict = torch.load(args.weights, map_location=device, weights_only=True)
        
        # 檢查並修正狀態字典中可能的問題
        if 'norm_3d.weight' in state_dict:
            # 為模型創建 norm_3d 參數，以匹配已保存的權重
            model.norm_3d = nn.LayerNorm(model.head[0].in_features).to(device)
            # 如果使用 DDP 訓練，移除 'module.' 前綴
            if list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
                
        # 使用嚴格=False載入，允許忽略不匹配的鍵
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"成功載入模型權重，部分參數可能被忽略")
    except Exception as e:
        logger.error(f"載入模型權重時發生錯誤: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # 設置損失函數
    criterion = nn.CrossEntropyLoss()
    
    # 評估模型
    logger.info("開始評估模型...")
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="測試中"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 儲存預測結果和真實標籤，用於混淆矩陣
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    logger.info(f"測試損失: {avg_loss:.3f} | 測試準確率: {accuracy:.2f}%")
    print(f"測試結果: 準確率 = {accuracy:.2f}%")
    
    # 計算每個類別的準確率
    if args.visualize:
        class_correct = list(0. for i in range(len(LABELS)))
        class_total = list(0. for i in range(len(LABELS)))
        
        for i, (predicted, target) in enumerate(zip(all_preds, all_targets)):
            class_correct[target] += (predicted == target)
            class_total[target] += 1
        
        # 顯示每個類別的準確率
        for i in range(len(LABELS)):
            if class_total[i] > 0:
                print(f'準確率 {LABELS[i]}: {100 * class_correct[i] / class_total[i]:.1f}%')
        
        # 生成混淆矩陣
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            # 選擇最常見的 10 個類別顯示
            top_classes = np.bincount(all_targets).argsort()[-10:]
            
            # 篩選這些類別的預測和真實標籤
            mask = np.isin(all_targets, top_classes)
            filtered_preds = np.array(all_preds)[mask]
            filtered_targets = np.array(all_targets)[mask]
            
            # 計算混淆矩陣
            cm = confusion_matrix(filtered_targets, filtered_preds)
            
            # 顯示混淆矩陣
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=[LABELS[i] for i in top_classes],
                       yticklabels=[LABELS[i] for i in top_classes])
            plt.xlabel('預測標籤')
            plt.ylabel('真實標籤')
            plt.title('混淆矩陣 (Top 10 類別)')
            plt.savefig('confusion_matrix.png')
            print(f"混淆矩陣已保存至 confusion_matrix.png")
        except Exception as e:
            logger.error(f"生成混淆矩陣時發生錯誤: {e}")
    
    # 可視化 Class Activation Map (CAM)
    if args.visualize:
        logger.info(f"生成 {args.num_visualize} 個樣本的 CAM 可視化...")
        # 創建用於可視化的目錄
        os.makedirs('cam_visualization', exist_ok=True)
        
        # 隨機選擇樣本進行可視化
        indices = np.random.choice(len(test_dataset), args.num_visualize, replace=False)
        
        for i, idx in enumerate(indices):
            # 獲取圖像
            img, label = test_dataset[idx]
            
            # 獲取原始圖像用於可視化
            img_path = test_df.iloc[idx]['path']  # 修改這裡，從 'image_path' 改為 'path'
            
            original_img = Image.open(img_path).convert('RGB')
            
            # 生成 CAM
            img_tensor = img.unsqueeze(0).to(device)
            cam = generate_cam_swin_v2_test(model, img_tensor)  # 使用新的非DDP版本的函數
            
            # 重設原始圖像大小以匹配 CAM
            original_img = original_img.resize((args.image_size, args.image_size))
            original_np = np.array(original_img)
            
            # 可視化 CAM
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(original_np)
            plt.title(f'原始圖像: {LABELS[label]}')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(cam, cmap='jet')
            plt.title('Class Activation Map')
            plt.axis('off')
            
            # 疊加 CAM 到原始圖像
            cam_resized = cv2.resize(cam, (args.image_size, args.image_size))
            heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # 使用 alpha blending 疊加原始圖像和 CAM
            overlay = 0.4 * heatmap + 0.6 * original_np
            overlay = overlay / overlay.max() * 255
            overlay = overlay.astype(np.uint8)
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('疊加圖像')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'cam_visualization/cam_{i}_{LABELS[label]}.png')
            plt.close()
        
        logger.info(f"CAM 可視化已保存至 cam_visualization 目錄")

    logger.info("測試完成")

if __name__ == "__main__":
    main()