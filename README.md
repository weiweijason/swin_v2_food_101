# Swin Transformer V2 on Food-101 資料集

這個專案使用Swin Transformer V2模型對Food-101食物圖像資料集進行分類訓練。

## 專案簡介

本專案使用最新的Swin Transformer V2模型架構，針對Food-101資料集（包含101類食物圖像）進行分類任務訓練。專案特色：

- 使用Swin Transformer V2模型進行遷移學習
- 採用分散式訓練（DistributedDataParallel）提高訓練效率
- 應用混合精度訓練加速訓練過程
- 整合進階資料增強技術如RandAugment、Mixup和Cutmix
- 採用Stochastic Depth提高深層網路泛化能力
- 使用Tensorboard視覺化訓練過程
- 完整的日誌記錄系統

## 模型參數設置 (2025年4月更新)

最新版本的訓練參數：

| 參數 | 數值 | 說明 |
|------|------|------|
| 圖像尺寸 | 256×256 | 從原本的224×224提高解析度 |
| 窗口大小 | 8×8 | 從原本的7×7增加窗口大小 |
| 批次大小 | 1024 | 大幅提高以加速訓練過程 |
| 訓練週期 | 30個epoch | 提供足夠的訓練時間 |
| 學習率 | 4e-5 | 使用小但有效的初始學習率 |
| 權重衰減 | 0.05 | 提供適當的正則化 |
| 隨機深度率 | 0.3 | 增加Stochastic Depth提高泛化能力 |

## 安裝與環境設置

```bash
# 安裝所需的套件
pip install torch torchvision torchaudio timm tensorboard pillow scikit-learn pandas opencv-python matplotlib tqdm

# 解壓資料集（如果尚未解壓）
unzip food-101.zip
```

## 運行訓練

本專案使用PyTorch的分散式訓練功能，透過`torchrun`啟動：

### 單機多GPU訓練

```bash
# 使用4個GPU進行訓練
torchrun --nproc_per_node=4 main.py
```

### 單GPU訓練

```bash
# 僅使用單個GPU進行訓練
torchrun --nproc_per_node=1 main.py
```

## 高級資料增強策略

本專案採用了多項最新的資料增強技術：

1. **RandAugment** - 自動搜索和應用最佳增強策略組合
   - 設置強度(magnitude)為9，操作數(num_ops)為2
   - 提供超過14種不同的圖像變換

2. **Mixup和Cutmix** - 混合不同圖像及其標籤
   - Mixup Alpha: 0.8
   - Cutmix Alpha: 1.0
   - 混合概率: 0.5
   - 切換概率: 0.5

3. **RandomErasing** - 隨機遮擋圖像部分區域
   - 應用概率: 0.25
   - 面積比例: 0.02-0.33
   - 長寬比: 0.3-3.3

這些技術顯著改善了模型的泛化能力和對各種變化的抵抗力。

## 使用Tensorboard監控訓練過程

在訓練過程中，程式會自動在`runs/`目錄中建立Tensorboard日誌。您可以使用以下命令啟動Tensorboard服務器：

```bash
tensorboard --logdir=runs
```

啟動後，在瀏覽器中訪問 http://localhost:6006 查看即時訓練數據：

- **損失曲線**：查看訓練和測試的損失變化
- **準確率**：查看訓練和測試的準確率變化
- **學習率**：監控學習率調整曲線

## 訓練日誌查看

訓練過程中的日誌會儲存在根目錄下的`training_[時間戳].log`文件中。

### 查看實時日誌
```bash
# 使用tail命令實時查看日誌
tail -f training_YYYYMMDD_HHMMSS.log
```

### 使用專用日誌監控工具
```bash
# 使用提供的監控工具查看關鍵指標
python tools/monitor_log.py training_YYYYMMDD_HHMMSS.log
```

## 優化技術詳解

### 批次處理與訓練參數
- 批次大小從64增加到1024，顯著加速訓練過程
- 梯度裁剪最大norm設置為5.0，防止梯度爆炸
- 訓練週期從20增加到30，確保充分學習

### 學習率調度
- 使用CosineLRScheduler實現余弦退火學習率
- 5個epoch的熱身期，初始熱身學習率為1e-7
- 最小學習率為1e-6，確保末期仍有微調空間

### Stochastic Depth（隨機深度）
- 設置丟棄路徑比率為0.3
- 通過隨機丟棄層在訓練時減少計算
- 迫使模型學習更具魯棒性的特徵表示

### 混合精度訓練
- 使用PyTorch的autocast和GradScaler
- 加速訓練過程並減少內存使用量
- 內建自動處理NaN/Inf錯誤的機制

## 問題排解

若遇到NaN問題或訓練不穩定：
- 檢查日誌中是否有"NaN/Inf detected"相關警告
- 嘗試減小學習率
- 調整梯度裁剪閾值
- 減小批次大小
- 降低stochastic depth比率
- 檢查是否有CUDA內存溢出情況
