import torch
import timm
import argparse
from PIL import Image
import torchvision.transforms as transforms

# 從 swinv1_to_swinv2.py 複製 LABELS 列表
LABELS =  [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 
    'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 
    'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 
    'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 
    'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 
    'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 
    'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 
    'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 
    'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 
    'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 
    'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 
    'tuna_tartare', 'waffles'
]

def test_load_model(model_name, pretrained_path=None, num_classes=len(LABELS)):
    """
    測試載入指定的 timm 模型。

    Args:
        model_name (str): 要載入的 timm 模型名稱。
        pretrained_path (str, optional): 預訓練權重檔案的路徑。如果為 None，則使用 timm 的預訓練權重。
        num_classes (int): 分類任務的類別數量。
    """
    try:
        if pretrained_path:
            print(f"嘗試從本地檔案 '{pretrained_path}' 載入模型 '{model_name}'...")
            # 1. 創建模型結構 (不加載預訓練權重)
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            
            # 2. 加載本地權重
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # 處理常見的 checkpoint 結構 (例如來自 PyTorch Lightning 或 MMSegmentation)
            # 這些框架可能會將實際的模型權重儲存在 'state_dict' 或 'model' 鍵下
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint # 假設 checkpoint 本身就是 state_dict
            else:
                state_dict = checkpoint # 假設 checkpoint 本身就是 state_dict
            
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                # 移除常見的前綴
                if new_k.startswith('module.'): # DataParallel/DistributedDataParallel 可能會添加 'module.' 前綴
                    new_k = new_k.replace('module.', '', 1)
                if new_k.startswith('backbone.'): # 錯誤訊息中看到的 'backbone.' 前綴
                    new_k = new_k.replace('backbone.', '', 1)
                
                # 嘗試進行一些常見的鍵名替換 (這部分可能需要根據您的 .pth 檔案來源進行調整)
                if 'patch_embedding.linear_embedding.' in new_k:
                    new_k = new_k.replace('patch_embedding.linear_embedding.', 'patch_embed.proj.')
                elif 'patch_embedding.normalization.' in new_k:
                    new_k = new_k.replace('patch_embedding.normalization.', 'patch_embed.norm.')
                
                if 'stages.' in new_k:
                    new_k = new_k.replace('stages.', 'layers.')
                
                if 'downsample.linear_mapping.' in new_k:
                    new_k = new_k.replace('downsample.linear_mapping.', 'downsample.reduction.')
                if 'layers' in new_k and 'downsample.normalization.' in new_k:
                     new_k = new_k.replace('downsample.normalization.', 'downsample.norm.')

                new_state_dict[new_k] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print(f"成功從 '{pretrained_path}' 載入模型 '{model_name}' 的權重 (經過鍵名轉換)。")
            except RuntimeError as e_strict:
                print(f"使用 strict=True 載入轉換後的 state_dict 失敗: {e_strict}")
                print("提示：這通常表示即使在調整前綴和一些常見名稱後，權重檔案的結構")
                print("仍然與 timm 的 SwinTransformerV2 模型架構存在根本差異。")
                print("嘗試使用 strict=False 載入：")
                
                filtered_state_dict_for_strict_false = {}
                model_state_keys = model.state_dict().keys()
                for key_ckpt, val_ckpt in new_state_dict.items():
                    if key_ckpt in model_state_keys:
                        if model.state_dict()[key_ckpt].shape == val_ckpt.shape:
                            filtered_state_dict_for_strict_false[key_ckpt] = val_ckpt
                        else:
                            print(f"  警告: 權重 '{key_ckpt}' 的形狀不匹配。模型: {model.state_dict()[key_ckpt].shape}, 檔案: {val_ckpt.shape}. 將跳過此權重。")

                model.load_state_dict(filtered_state_dict_for_strict_false, strict=False)
                print(f"已使用 strict=False 從 '{pretrained_path}' 載入模型 '{model_name}' 的權重。")
                print("請檢查輸出，確認哪些權重被跳過或未載入。")

        else:
            print(f"嘗試從 timm 載入預訓練模型 '{model_name}'...")
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            print(f"成功從 timm 載入預訓練模型 '{model_name}'。")
        
        print(f"模型 '{model_name}' 已成功載入並配置了 {num_classes} 個輸出類別。")
        return model

    except Exception as e:
        print(f"載入模型 '{model_name}' 失敗: {e}")
        if pretrained_path:
            print("\n--- 載入失敗的可能原因與建議 ---")
            print("1. 權重檔案的架構與 timm 模型不完全相容。")
            print("2. 權重檔案來源問題 (例如 SwinV1 vs SwinV2)。")
            print("3. 前綴或命名差異。")
            print("4. 分類頭 (Head) 不匹配。")
        return None

def predict_image(model, image_path, labels, model_name):
    """
    使用載入的模型對單張圖片進行預測。

    Args:
        model: 已載入的 PyTorch 模型。
        image_path (str): 要預測的圖片檔案路徑。
        labels (list): 類別標籤列表。
        model_name (str): 模型名稱，用於輔助決定預處理方式。
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # 獲取模型的預處理配置
        # SwinV2 Base Window12 192x192 -> input_size = 192
        # Swin Base Patch4 Window7 224x224 -> input_size = 224
        input_size = 192 # 預設值，根據 swinv2_base_window12_192
        if "224" in model_name: # 簡易判斷
            input_size = 224
        elif "384" in model_name:
            input_size = 384

        # 使用 timm 建議的轉換方式，如果模型有 pretrained_cfg
        # 否則使用標準 ImageNet 轉換
        try:
            data_config = timm.data.resolve_data_config({}, model=model)
            val_transforms = timm.data.create_transform(**data_config, is_training=False)
        except Exception:
            print("無法從模型獲取 data_config，使用標準轉換。")
            val_transforms = transforms.Compose([
                transforms.Resize(int(input_size / 0.875)), # 依照常見比例放大再裁切
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        img_tensor = val_transforms(img).unsqueeze(0) # 添加 batch 維度

        model.eval() # 設定為評估模式
        with torch.no_grad(): # 不計算梯度
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_p, top_class_idx = probabilities.topk(1, dim=1) # 取最高機率的類別

        predicted_label = labels[top_class_idx.item()]
        confidence = top_p.item()

        print("-" * 30)
        print(f"圖片 '{image_path}' 的預測結果:")
        print(f"  預測類別: {predicted_label}")
        print(f"  信賴度: {confidence:.4f}")
        print("-" * 30)

    except FileNotFoundError:
        print(f"錯誤: 找不到圖片檔案 '{image_path}'")
    except Exception as e:
        print(f"預測圖片時發生錯誤: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='測試模型載入與預測功能')
    parser.add_argument('--model_name', type=str, required=True, help='要測試的 timm 模型名稱 (例如 swinv2_base_window12_192 或 swin_base_patch4_window7_224)')
    parser.add_argument('--pretrained_path', type=str, default=None, help='(可選) 本地預訓練權重檔案的路徑 (.pth)')
    parser.add_argument('--num_classes', type=int, default=len(LABELS), help='模型的輸出類別數量')
    parser.add_argument('--image_path', type=str, default=None, help='(可選) 要進行預測的圖片路徑')
    
    args = parser.parse_args()

    print(f"開始測試模型載入: {args.model_name}")
    if args.pretrained_path:
        print(f"使用本地權重: {args.pretrained_path}")
    else:
        print("使用 timm 的預訓練權重 (如果可用)")
    print(f"類別數量: {args.num_classes}")
    print("-" * 30)

    loaded_model = test_load_model(args.model_name, args.pretrained_path, args.num_classes)

    if loaded_model:
        print("-" * 30)
        print("模型載入測試成功！")
        if args.image_path:
            predict_image(loaded_model, args.image_path, LABELS, args.model_name)
    else:
        print("-" * 30)
        print("模型載入測試失敗。")