import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from sklearn.utils import shuffle
import timm  # 確保您已安裝 timm 庫

# 定義 Food101Dataset 類
class Food101Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# 提取特徵函數
def extract_features(model, dataloader, device):
    """使用訓練好的模型提取特徵"""
    model.eval()
    features = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # 將特徵展平為 [batch_size, feature_dim]
            outputs_flat = outputs.view(outputs.size(0), -1)
            features.append(outputs_flat.cpu().numpy())
    return np.concatenate(features)

# 分群分析函數
def perform_kmeans_clustering(features, n_clusters=101):
    """使用 KMeans 進行分群"""
    # 降維以便可視化（可選）
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # 使用 KMeans 進行分群
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)

    return reduced_features, cluster_labels

# 主程式
if __name__ == "__main__":
    # 設定設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定義模型結構 - 必須與訓練時一致（101個分類）
    model = timm.create_model('swinv2_base_window12_192', pretrained=False, num_classes=101)
    
    # 載入保存的權重
    model_path = "outputs/swinv2_food101_best.pth"  # 使用正確的檔案名稱
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        # 移除可能的 "module." 前綴
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # 移除 "module." 前綴
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        print("模型權重載入成功！")
    except RuntimeError as e:
        print(f"載入模型權重時發生錯誤: {e}")
        print("請確認保存的模型權重與模型結構一致。")
        exit(1)
    
    # 修改模型以提取特徵（移除最後的分類層）
    model = nn.Sequential(*list(model.children())[:-1])  # 移除最後的分類層
    model = model.to(device)

    # 設定數據增強
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 準備測試數據集（保留真實標籤以便分析）
    test_file = "food-101/meta/test.txt"
    image_root = "food-101/images"
    with open(test_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    true_labels = []  # 保存真實的食物類別
    for line in lines:
        category, img_name = line.split('/')
        data.append({
            'path': f"{image_root}/{category}/{img_name}.jpg",
            'category': category  # 保存類別名稱
        })
        true_labels.append(category)

    test_df = pd.DataFrame(data)
    test_df = shuffle(test_df)
    true_labels = test_df['category'].tolist()  # 更新打亂後的標籤順序

    test_dataset = Food101Dataset(test_df, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 提取特徵
    print("開始提取特徵...")
    features = extract_features(model, test_loader, device)
    print(f"提取的特徵形狀: {features.shape}")

    # 分群分析
    print("開始進行 KMeans 分群...")
    reduced_features, cluster_labels = perform_kmeans_clustering(features, n_clusters=101)
    print(f"降維後特徵形狀: {reduced_features.shape}")
    print(f"分群標籤數量: {len(cluster_labels)}")

    # 分析分群結果
    print("\n=== 分群結果分析 ===")
    unique_true_labels = list(set(true_labels))
    print(f"原始食物類別數量: {len(unique_true_labels)}")
    print(f"KMeans 分群數量: 101")
    
    # 統計每個群組包含的食物類別
    from collections import defaultdict, Counter
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    
    cluster_to_foods = defaultdict(list)
    for i, (cluster_id, food_category) in enumerate(zip(cluster_labels, true_labels)):
        cluster_to_foods[cluster_id].append(food_category)
    
    # ✅ 1. NMI（Normalized Mutual Information）
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
    print(f"NMI 分數: {nmi_score:.4f}")
    
    # ✅ 2. ARI（Adjusted Rand Index）
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    print(f"ARI 分數: {ari_score:.4f}")
    
    # ✅ 3. 每群組中「最常見食物類別」的準確率統計
    correct = 0
    total = 0
    for cluster_id in cluster_to_foods:
        if len(cluster_to_foods[cluster_id]) > 0:  # 確保群組不為空
            most_common = Counter(cluster_to_foods[cluster_id]).most_common(1)
            correct += most_common[0][1]
            total += len(cluster_to_foods[cluster_id])
    
    cluster_accuracy = correct / total if total > 0 else 0
    print(f"以群內最多類別為準的準確率: {cluster_accuracy:.4f}")
    print(f"正確分群的圖像數量: {correct}/{total}")
    
    for cluster_id in range(101):
        foods_in_cluster = cluster_to_foods[cluster_id]
        food_counts = Counter(foods_in_cluster)
        most_common_foods = food_counts.most_common(5)  # 顯示前5個最常見的食物
        print(f"\n群組 {cluster_id} (共 {len(foods_in_cluster)} 張圖像):")
        print(f"  主要食物類別: {most_common_foods}")

    # 可視化分群結果
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.title("KMeans Clustering Results (Unsupervised Learning)\nUsing Swin Transformer V2 Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster ID")
    
    # 添加群組中心點
    kmeans = KMeans(n_clusters=101, random_state=42)
    kmeans.fit(reduced_features)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
    plt.legend()
    
    # 保存圖表
    plt.savefig('kmeans_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n=== 總結 ===")
    print("✅ 無監督學習分群成功完成！")
    print("📊 結果解釋：")
    print("   - 101個群組對應101種食物類別")
    print("   - 這是一個與監督學習結果比較的無監督學習實驗")
    print("   - 評估指標顯示了無監督分群與真實標籤的相似程度")
    print(f"📁 可視化圖表已保存為 'kmeans_clustering_results.png'")
    
    print("\n📈 評估指標解釋：")
    print(f"   • NMI (Normalized Mutual Information): {nmi_score:.4f}")
    print("     - 範圍 [0,1]，越接近 1 越好，衡量分群與真實標籤的資訊一致性")
    print(f"   • ARI (Adjusted Rand Index): {ari_score:.4f}")
    print("     - 範圍 [-1,1]，越接近 1 越好，衡量分群與真實分類的相似程度")
    print(f"   • 群內準確率: {cluster_accuracy:.4f}")
    print("     - 每個群組中最常見類別所占的比例")
    
    # 建議嘗試不同的群組數量
    print("\n💡 建議：")
    if nmi_score > 0.8:
        print("   ✅ NMI分數很高，表示無監督分群與真實標籤高度一致！")
    elif nmi_score > 0.6:
        print("   📊 NMI分數中等，無監督分群部分成功")
    else:
        print("   📉 NMI分數較低，可以嘗試不同的特徵提取方法或分群算法")
        
    if ari_score > 0.8:
        print("   ✅ ARI分數很高，分群結果與真實分類非常相似！")
    elif ari_score > 0.6:
        print("   📊 ARI分數中等，分群結果部分符合真實分類")
    else:
        print("   📉 ARI分數較低，建議嘗試其他分群參數或特徵")
        
    print("   - 可以嘗試不同的分群數量 (10, 20, 50) 來觀察層次化分群效果")
    print("   - 可以嘗試其他分群算法如 DBSCAN 或 Gaussian Mixture Model")