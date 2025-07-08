import torch
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
            features.append(outputs.cpu().numpy())
    return np.concatenate(features)

# 分群分析函數
def perform_kmeans_clustering(features, n_clusters=10):
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

    # 定義模型結構
    model = timm.create_model('swinv2_base_window12_192', pretrained=False, num_classes=0)  # num_classes=0 表示提取特徵
    model = model.to(device)

    # 載入保存的權重
    model_path = "outputs/unsupervised_swinv2_food101_best_loss.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 設定數據增強
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 準備測試數據集
    test_file = "food-101/meta/test.txt"
    image_root = "food-101/images"
    with open(test_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        category, img_name = line.split('/')
        data.append({
            'path': f"{image_root}/{category}/{img_name}.jpg"
        })

    test_df = pd.DataFrame(data)
    test_df = shuffle(test_df)

    test_dataset = Food101Dataset(test_df, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 提取特徵
    features = extract_features(model, test_loader, device)

    # 分群分析
    reduced_features, cluster_labels = perform_kmeans_clustering(features, n_clusters=10)

    # 可視化分群結果
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
    plt.title("KMeans Clustering Results")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()