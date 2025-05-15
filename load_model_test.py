import torch
import timm
import argparse

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
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print(f"成功從 '{pretrained_path}' 載入模型 '{model_name}' 的權重。")
        else:
            print(f"嘗試從 timm 載入預訓練模型 '{model_name}'...")
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            print(f"成功從 timm 載入預訓練模型 '{model_name}'。")
        
        # 打印模型結構以確認
        # print("\n模型結構:")
        # print(model)
        print(f"模型 '{model_name}' 已成功載入並配置了 {num_classes} 個輸出類別。")
        return model

    except Exception as e:
        print(f"載入模型 '{model_name}' 失敗: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='測試模型載入功能')
    parser.add_argument('--model_name', type=str, required=True, help='要測試的 timm 模型名稱 (例如 swinv2_base_window12_192 或 swin_base_patch4_window7_224)')
    parser.add_argument('--pretrained_path', type=str, default=None, help='(可選) 本地預訓練權重檔案的路徑 (.pth)')
    parser.add_argument('--num_classes', type=int, default=len(LABELS), help='模型的輸出類別數量')
    
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
    else:
        print("-" * 30)
        print("模型載入測試失敗。")

    # 示例用法:
    # 1. 測試載入 swinv1_to_swinv2.py 中使用的 Swin V2 模型 (使用 timm 預訓練權重)
    # python load_model_test.py --model_name swinv2_base_window12_192
    
    # 2. 測試載入 swinv1_to_swinv2.py 中使用的 Swin V1 模型 (使用 timm 預訓練權重)
    # python load_model_test.py --model_name swin_base_patch4_window7_224

    # 3. 測試從本地 .pth 檔案載入 Swin V2 模型 (假設你有一個名為 'swinv2_food101_best.pth' 的檔案)
    # python load_model_test.py --model_name swinv2_base_window12_192 --pretrained_path outputs/swinv2_food101_best.pth
    
    # 4. 測試載入一個不同的模型，例如 EfficientNet (使用 timm 預訓練權重)
    # python load_model_test.py --model_name efficientnet_b0 --num_classes 1000
