import timm
import torch

# 從 swinv1_to_swinv2.py 複製 LABELS 列表或其長度
# 這裡我們直接使用其長度，因為模型載入時只需要 num_classes
NUM_LABELS = 101 # Food-101 資料集的類別數量

def test_load_model(use_v2, image_size):
    """
    測試載入 Swin Transformer 模型。

    Args:
        use_v2 (bool): 是否使用 Swin Transformer V2。
        image_size (int): 輸入圖像的大小。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- 測試開始: use_v2={use_v2}, image_size={image_size} ---")
    print(f"使用設備: {device}")

    model_name_v1 = 'swin_base_patch4_window7_224'
    # 根據 image_size 選擇合適的 SwinV2 模型
    # 這裡我們假設 image_size=192 對應 'swinv2_base_window12_192'
    # 如果 image_size 是其他值，可能需要調整模型名稱
    model_name_v2 = 'swinv2_base_window12_192' 
    
    if use_v2:
        print(f"嘗試載入 Swin Transformer V2 模型: {model_name_v2}")
        try:
            model = timm.create_model(
                model_name_v2,
                pretrained=True,
                num_classes=NUM_LABELS,
                # img_size=image_size # 部分 timm 模型接受 img_size 參數
            )
            model = model.to(device)
            print(f"成功載入 Swin Transformer V2 模型 ({model_name_v2}) 並移至 {device}")
            # print(model) # 可選：印出模型結構
        except Exception as e:
            print(f"無法載入 Swin V2 模型 ({model_name_v2}): {e}")
            print("嘗試使用 Swin V1 替代...")
            try:
                model = timm.create_model(
                    model_name_v1,
                    pretrained=True,
                    num_classes=NUM_LABELS
                )
                model = model.to(device)
                print(f"成功載入 Swin Transformer V1 模型 ({model_name_v1}) 作為替代並移至 {device}")
            except Exception as e_v1:
                print(f"無法載入 Swin V1 替代模型 ({model_name_v1}): {e_v1}")
    else:
        print(f"嘗試載入 Swin Transformer V1 模型: {model_name_v1}")
        try:
            model = timm.create_model(
                model_name_v1,
                pretrained=True,
                num_classes=NUM_LABELS
            )
            model = model.to(device)
            print(f"成功載入 Swin Transformer V1 模型 ({model_name_v1}) 並移至 {device}")
            # print(model) # 可選：印出模型結構
        except Exception as e:
            print(f"無法載入 Swin V1 模型 ({model_name_v1}): {e}")
    
    print(f"--- 測試結束: use_v2={use_v2}, image_size={image_size} ---\n")

if __name__ == "__main__":
    # 情境1: 測試 Swin V1 (通常 image_size 為 224)
    test_load_model(use_v2=False, image_size=224)

    # 情境2: 測試 Swin V2 (通常 image_size 為 192 for swinv2_base_window12_192)
    test_load_model(use_v2=True, image_size=192)

    # 情境3: 測試 Swin V2，但故意使用不匹配的 image_size (例如 224)
    # 注意：timm 模型通常有固定的預訓練圖像大小，
    # 'swinv2_base_window12_192' 預期輸入是 192x192。
    # 如果 transform 階段會 resize 到模型預期的大小，這裡的 image_size 參數主要是為了選擇模型。
    # timm.create_model 中的 img_size 參數可以覆蓋預設，但並非所有模型都支援。
    # 在 swinv1_to_swinv2.py 中，IMAGE_SIZE 用於 transforms.Resize，
    # 而模型選擇是基於 --use_v2 和硬編碼的模型名稱。
    print("注意: 下一個測試模擬了 image_size 與 SwinV2 預期不完全匹配的情況。")
    print("在 swinv1_to_swinv2.py 中，實際的圖像大小由 transform 處理。")
    print("此處的 image_size 參數主要用於演示模型選擇邏輯。")
    test_load_model(use_v2=True, image_size=224) # 這裡的 image_size 224 實際上不會直接傳給 create_model 的 img_size 參數，除非我們修改 test_load_model
                                                # 在原始腳本中，'swinv2_base_window12_192' 被固定使用。
                                                # 如果要測試不同預訓練大小的 SwinV2，需要更改 model_name_v2。
                                                
    # 例如，如果要測試 'swinv2_base_window8_256' (預期 256x256)
    # 你可以這樣修改 test_load_model 或傳遞模型名稱：
    # def test_load_model_flexible(use_v2, image_size, model_name_if_v2='swinv2_base_window12_192'):
    # ...
    # test_load_model_flexible(use_v2=True, image_size=256, model_name_if_v2='swinv2_base_window8_256')

    print("所有測試完成。")
