# Swin Transformer V2 於 Food-101 資料集分類

這個專案使用 Swin Transformer V2 模型對 Food-101 食物圖像資料集進行分類訓練。

## 專案簡介

本專案使用最新的 Swin Transformer V2 模型架構，針對 Food-101 資料集（包含 101 類食物圖像）進行分類任務訓練。專案特色：

- 使用 Swin Transformer V2 模型進行深度學習
- 採用分散式訓練（DistributedDataParallel）提高訓練效率
- 應用混合精度訓練（AMP）加速訓練過程
- 整合進階資料增強技術如 RandAugment、Mixup 和 Cutmix
- 採用 Stochastic Depth 提高深層網路泛化能力
- 使用 Tensorboard 視覺化訓練過程和關鍵指標
- 完整的日誌記錄系統
- 自適應錯誤處理機制以提高訓練穩定性

## 模型參數設置 (2025年4月更新)

最新版本的訓練參數：

| 參數 | 數值 | 說明 |
|------|------|------|
| 圖像尺寸 | 192×192 | 降低解析度以減少記憶體需求 |
| 窗口大小 | 6×6 | 調整窗口大小使其能夠被圖像尺寸整除 |
| 批次大小 | 32 | 較小的批次大小以適應記憶體限制 |
| 訓練週期 | 30個epoch | 提供足夠的訓練時間 |
| 學習率 | 1e-4 | 提高初始學習率加速收斂 |
| 權重衰減 | 0.05 | 提供適當的正則化 |
| 隨機深度率 | 0.1 | 降低 Stochastic Depth 比率以提高穩定性 |
| 梯度累積步數 | 4 | 使用梯度累積以模擬更大批次大小 |

## 安裝與環境設置

```bash
# 安裝所需的套件
pip install torch torchvision torchaudio timm tensorboard pillow scikit-learn pandas opencv-python matplotlib tqdm

# 解壓資料集（如果尚未解壓）
unzip food-101.zip
```

## 運行訓練

本專案使用 PyTorch 的分散式訓練功能，透過 `torchrun` 啟動：

### 單機多 GPU 訓練

```bash
# 使用多個 GPU 進行訓練
torchrun --nproc_per_node=N main.py
```
其中 N 為您想使用的 GPU 數量。

### 單 GPU 訓練

```bash
# 僅使用單個 GPU 進行訓練
torchrun --nproc_per_node=1 main.py
```

## 記憶體優化策略

本專案採用多種技術減少記憶體使用量，適合在資源有限的環境下運行：

1. **降低圖像尺寸** - 從常見的 224×224 降至 192×192
2. **減小批次大小** - 設置為 32，適應大多數 GPU
3. **梯度累積** - 使用 4 步梯度累積模擬更大批次
4. **混合精度訓練** - 使用 FP16 進行部分計算
5. **Checkpoint 功能** - 在反向傳播時重新計算某些結果而非保存在記憶體中
6. **優化數據加載器** - 降低預取因子並調整工作進程數

## 高級資料增強策略

本專案採用了多項最新的資料增強技術：

1. **RandAugment** - 自動搜索和應用最佳增強策略組合
   - 設置強度(magnitude)為 5，操作數(num_ops)為 2
   - 降低強度以提高訓練穩定性

2. **Mixup 和 Cutmix** - 混合不同圖像及其標籤
   - Mixup Alpha: 0.8
   - Cutmix Alpha: 1.0
   - 混合概率: 0.5
   - 切換概率: 0.5

3. **RandomErasing** - 隨機遮擋圖像部分區域
   - 應用概率: 0.15（降低以提高穩定性）
   - 面積比例: 0.02-0.33
   - 長寬比: 0.3-3.3

## 學習率調度與優化器

- **優化器**: AdamW 優化器配合 0.05 權重衰減
- **學習率策略**: CosineLRScheduler 實現余弦退火學習率
- **熱身期**: 10 個 epoch 的熱身期，初始熱身學習率為 1e-7
- **最小學習率**: 1e-6，確保末期仍有微調空間

## 穩定性優化與錯誤處理

為提高訓練過程的穩定性，我們實施了以下機制：

1. **自動處理 NaN/Inf 值** - 在檢測到 NaN 時跳過批次並降低學習率
2. **連續 NaN 處理** - 當連續出現 NaN 時自動降低學習率 50%
3. **梯度裁剪** - 最大 norm 設為 1.0，防止梯度爆炸
4. **定期釋放記憶體** - 每 50 批次主動釋放未使用的 CUDA 記憶體

## 使用 Tensorboard 監控訓練

訓練過程中，程式會自動在 `runs/` 目錄中建立 Tensorboard 日誌。您可以使用以下命令啟動 Tensorboard 服務器：

```bash
tensorboard --logdir=runs
```

啟動後，在瀏覽器中訪問 http://localhost:6006 查看即時訓練數據。

## 檔案結構說明

- `main.py` - 主程式，包含訓練和評估邏輯
- `swin_transformer_v2_classifier.py` - Swin V2 分類器定義
- `swin_transformer_v2/` - Swin V2 模型的核心實現
  - `model.py` - 模型架構定義
  - `model_parts.py` - 各個組件的實現
- `custom_model_parts.py` - 自定義模型組件
- `tools/` - 輔助工具腳本

## 問題排解

若遇到形狀不匹配錯誤（shape error）:
- 確保窗口大小能夠被圖像尺寸整除
- 配置 `window_size` 參數與 `IMAGE_SIZE` 相容

若遇到 NaN 問題或訓練不穩定：
- 嘗試減小學習率
- 調整梯度裁剪閾值
- 減小批次大小
- 降低 stochastic depth 比率

## 圖片可視化

模型包含 Class Activation Map (CAM) 可視化功能，可用於理解模型如何識別不同食物類別：

```python
# 生成 CAM 圖
image_tensor = transform(image).unsqueeze(0)
cam = generate_cam_swin_v2(model, image_tensor)
overlay = visualize_cam(image_tensor[0], cam)
plt.imshow(overlay)
plt.show()
```
