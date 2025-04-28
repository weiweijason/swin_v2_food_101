# Swin Transformer V2 on Food-101 資料集

這個專案使用Swin Transformer V2模型對Food-101食物圖像資料集進行分類訓練。

## 專案簡介

本專案使用最新的Swin Transformer V2模型架構，針對Food-101資料集（包含101類食物圖像）進行分類任務訓練。專案特色：

- 使用Swin Transformer V2模型進行遷移學習
- 採用分散式訓練（DistributedDataParallel）提高訓練效率
- 應用混合精度訓練加速訓練過程
- 使用Tensorboard視覺化訓練過程
- 完整的日誌記錄系統

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

## 使用Tensorboard監控訓練過程

在訓練過程中，程式會自動在`runs/`目錄中建立Tensorboard日誌。您可以使用以下命令啟動Tensorboard服務器：

```bash
tensorboard --logdir=runs
```

啟動後，在瀏覽器中訪問 http://localhost:6006 查看即時訓練數據：

- **損失曲線**：查看訓練和測試的損失變化
- **準確率**：查看訓練和測試的準確率變化
- **學習率**：監控不同參數組的學習率調整

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

## 優化策略

增強數據擴增策略：

- 為訓練集添加了隨機裁剪、水平翻轉、色彩抖動等
- 優化了測試集的數據處理流程，使用了適合評估的中心裁剪

優化學習率與訓練策略：

- 實現了分層學習率，讓骨幹網路使用較低的學習率 (5e-6)，分類頭使用較高學習率 (1e-5)
- 使用了更先進的 CosineLRScheduler，加入了 5 個 epoch 的熱身期
- 降低權重衰減至 0.01 提高訓練穩定性

添加混合精度訓練：

- 使用 PyTorch 的 autocast 和 GradScaler 實現了混合精度訓練
- 這將加速訓練過程 (約 1.3-2 倍)，同時降低 GPU 記憶體占用

增加標籤平滑和正則化：

- 設置了 0.1 的標籤平滑度
- 加入了梯度裁剪 (max_norm=1.0) 以提高訓練穩定性

增加 batch size：

- 將批次大小從 32 提高到 64，提高訓練效率和潛在準確率

## 問題排解

若遇到NaN問題或訓練不穩定：
- 檢查日誌中是否有"NaN/Inf detected"相關警告
- 嘗試減小學習率
- 調整梯度裁剪閾值
- 減小批次大小
