import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from swin_transformer_v2_classifier import swin_transformer_v2_base_classifier
from test_model import generate_cam_swin_v2_test
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description='生成食物101數據集的CAM可視化')
    parser.add_argument('--model_path', type=str, default='swinv2_food101_best.pth', help='模型權重檔案的路徑')
    parser.add_argument('--image_dir', type=str, default='food-101/images', help='圖像目錄')
    parser.add_argument('--output_dir', type=str, default='cam_visualization', help='輸出目錄')
    parser.add_argument('--num_images', type=int, default=5, help='要可視化的圖像數量')
    parser.add_argument('--image_size', type=int, default=224, help='輸入圖像大小')
    parser.add_argument('--window_size', type=int, default=7, help='窗口大小')
    args = parser.parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 載入類別標籤
    with open('label.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # 初始化模型
    model = swin_transformer_v2_base_classifier(
        input_resolution=(args.image_size, args.image_size),
        window_size=args.window_size,
        num_classes=len(classes)
    )
    
    # 載入權重
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        # 調整state_dict以處理可能的差異
        if 'module.backbone.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # 移除"module."前綴
                new_state_dict[k] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        print(f"成功載入模型權重: {args.model_path}")
    except Exception as e:
        print(f"載入模型權重時出錯: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # 設定圖像預處理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 隨機選擇一些圖像類別和圖像
    selected_classes = np.random.choice(classes, min(5, len(classes)), replace=False)
    
    for class_name in selected_classes:
        class_dir = os.path.join(args.image_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"類別目錄不存在: {class_dir}")
            continue
        
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        if not image_files:
            print(f"在 {class_dir} 中找不到圖像")
            continue
        
        # 選擇一個圖像
        image_file = np.random.choice(image_files)
        image_path = os.path.join(class_dir, image_file)
        
        # 載入和預處理圖像
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 前向傳播
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                prediction = classes[predicted.item()]
            
            # 生成CAM
            cam = generate_cam_swin_v2_test(model, img_tensor)
            
            # 調整原始圖像大小用於顯示
            original_img = img.resize((args.image_size, args.image_size))
            original_np = np.array(original_img)
            
            # 調整CAM大小
            cam_resized = cv2.resize(cam, (args.image_size, args.image_size))
            
            # 創建熱力圖
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # 疊加熱力圖到原始圖像
            combined = 0.6 * original_np + 0.4 * heatmap
            combined = combined / combined.max() * 255
            combined = combined.astype(np.uint8)
            
            # 顯示結果
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_np)
            plt.title(f'原始: {class_name}\n預測: {prediction}')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(cam_resized, cmap='jet')
            plt.title('CAM')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(combined)
            plt.title('疊加圖')
            plt.axis('off')
            
            plt.tight_layout()
            
            # 保存圖像
            output_path = os.path.join(args.output_dir, f'cam_{class_name}_{image_file}.png')
            plt.savefig(output_path)
            plt.close()
            
            print(f"已生成並保存CAM到 {output_path}")
            
        except Exception as e:
            print(f"處理圖像 {image_path} 時出錯: {e}")

if __name__ == '__main__':
    main()