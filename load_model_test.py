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
                # 例如: 'patch_embedding.linear_embedding.' -> 'patch_embed.proj.'
                #       'patch_embedding.normalization.' -> 'patch_embed.norm.'
                #       'stages.' -> 'layers.'
                #       'downsample.linear_mapping.' -> 'downsample.reduction.'
                #       'downsample.normalization.' -> 'downsample.norm.' (需小心，避免與 patch_embed.norm 混淆)

                if 'patch_embedding.linear_embedding.' in new_k:
                    new_k = new_k.replace('patch_embedding.linear_embedding.', 'patch_embed.proj.')
                elif 'patch_embedding.normalization.' in new_k:
                    new_k = new_k.replace('patch_embedding.normalization.', 'patch_embed.norm.')
                
                if 'stages.' in new_k:
                    new_k = new_k.replace('stages.', 'layers.')
                
                if 'downsample.linear_mapping.' in new_k:
                    new_k = new_k.replace('downsample.linear_mapping.', 'downsample.reduction.')
                # 僅當 'downsample.normalization.' 出現在 'layers' 上下文時才替換，以避免與 'patch_embed.norm' 衝突
                if 'layers' in new_k and 'downsample.normalization.' in new_k:
                     new_k = new_k.replace('downsample.normalization.', 'downsample.norm.')

                # 關於 head 的處理:
                # timm 模型期望 head 的鍵名為 'head.fc.weight' 和 'head.fc.bias' (或類似結構)
                # 您的 .pth 檔案中有 'head.2.weight' 等。
                # 如果 num_classes 與原始模型不同，timm 會自動創建新的 head，通常不需要載入舊 head 權重。
                # 如果 num_classes 相同且您想載入 head 權重，則需要更精確的映射。
                # 為簡化，此處我們不過濾 head 鍵，讓 load_state_dict (strict=False 時) 自行處理。
                # 如果 strict=True 失敗，通常是因為 head 不匹配。

                new_state_dict[new_k] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print(f"成功從 '{pretrained_path}' 載入模型 '{model_name}' 的權重 (經過鍵名轉換)。")
            except RuntimeError as e_strict:
                print(f"使用 strict=True 載入轉換後的 state_dict 失敗: {e_strict}")
                print("提示：這通常表示即使在調整前綴和一些常見名稱後，權重檔案的結構 (例如 Attention Block 或 Head 部分)")
                print("仍然與 timm 的 SwinTransformerV2 模型架構存在根本差異。")
                print("如果權重來自 SwinV1，則需要專門的轉換腳本。")
                print("嘗試使用 strict=False 載入，這會忽略不匹配的鍵和大小不符的鍵 (請謹慎使用)：")
                
                # 為了使用 strict=False，過濾掉在目標模型中不存在的鍵，以及大小不匹配的鍵
                filtered_state_dict_for_strict_false = {}
                model_state_keys = model.state_dict().keys()
                for key_ckpt, val_ckpt in new_state_dict.items():
                    if key_ckpt in model_state_keys:
                        if model.state_dict()[key_ckpt].shape == val_ckpt.shape:
                            filtered_state_dict_for_strict_false[key_ckpt] = val_ckpt
                        else:
                            print(f"  警告: 權重 '{key_ckpt}' 的形狀不匹配。模型: {model.state_dict()[key_ckpt].shape}, 檔案: {val_ckpt.shape}. 將跳過此權重。")
                    # else: # 不打印所有不存在的鍵，因為 strict=False 會自動忽略它們
                        # print(f"  警告: 權重 '{key_ckpt}' 在目標模型中不存在。將被忽略。")

                model.load_state_dict(filtered_state_dict_for_strict_false, strict=False)
                print(f"已使用 strict=False 從 '{pretrained_path}' 載入模型 '{model_name}' 的權重。")
                print("請檢查輸出，確認哪些權重被跳過或未載入。模型可能無法正常工作。")

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
        if pretrained_path: # 僅當嘗試從本地檔案載入時提供額外提示
            print("\n--- 載入失敗的可能原因與建議 ---")
            print("1. 權重檔案的架構與 timm 模型不完全相容：")
            print("   - 錯誤訊息中的 'Missing key(s)' 指出 timm 模型需要的層在您的權重檔中找不到。")
            print("   - 錯誤訊息中的 'Unexpected key(s)' 指出您的權重檔中包含 timm 模型無法識別的層。")
            print("   - 特別注意 'attn.relative_position_bias_table' (SwinV1 特徵) vs 'attn.logit_scale', 'attn.cpb_mlp' (timm SwinV2 特徵)。")
            print("2. 權重檔案來源：")
            print("   - 如果此權重是 SwinV1 模型的，您需要使用一個可靠的轉換腳本將其轉換為與 timm SwinV2 相容的格式。")
            print("   - 如果是其他來源的 SwinV2 權重，其內部層命名可能與 timm 不同。")
            print("3. 前綴問題：程式碼已嘗試移除 'module.' 和 'backbone.' 前綴，但可能還有其他前綴或命名差異。")
            print("4. 分類頭 (Head) 不匹配：如果您的權重檔案包含一個與 timm 模型預期不同的分類頭，也會導致錯誤。")
            print("建議：")
            print("   - 仔細檢查權重檔案的來源和原始模型架構。")
            print("   - 如果是 SwinV1，尋找或編寫一個針對 timm SwinV2 的轉換腳本。")
            print("   - 仔細比對 'Missing' 和 'Unexpected' 鍵列表，嘗試手動或透過腳本進行更精確的鍵名映射。")
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
