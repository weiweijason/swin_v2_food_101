import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.distributed import destroy_process_group
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import glob
from sklearn.cluster import KMeans

# Import the SwinV2 classifier
from swin_transformer_v2_classifier import swin_transformer_v2_base_classifier, swin_transformer_v2_base_simclr

# Updated Label Encoder
class Label_encoder:
    def __init__(self, labels):
        self.labels = {label: idx for idx, label in enumerate(labels)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels.get(label)


# Custom Dataset for Image Classification
class Food101Dataset(Dataset):
    def __init__(self, dataframe, encoder, transform=None):
        self.dataframe = dataframe
        self.encoder = encoder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        label = self.dataframe.iloc[idx]['label']
        label = self.encoder.get_idx(label)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# Updated prepare_dataframe function
def prepare_dataframe(file_path, image_root, encoder):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        category, img_name = line.split('/')
        if category in encoder.labels:
            data.append({
                'label': category,
                'path': f"{image_root}/{category}/{img_name}.jpg"
            })

    df = pd.DataFrame(data)
    return shuffle(df)


# Training and Testing Functions
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Apply gradient clipping after backward pass but before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Train Loss: {total_loss / len(dataloader):.3f} | Train Accuracy: {accuracy:.2f}%")
    
    # Correct - stepping scheduler once per epoch
    scheduler.step()
    
    return accuracy  # It's helpful to return the accuracy for monitoring


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Loss: {total_loss / len(dataloader):.3f} | Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Generate CAM for Swin Transformer V2
def generate_cam_swin_v2(model, input_tensor, class_idx=None):
    """
    Generate Class Activation Map for Swin Transformer V2
    :param model: The SwinV2 model
    :param input_tensor: Input image tensor [1, C, H, W]
    :param class_idx: Target class index (None for predicted class)
    :return: CAM as numpy array
    """
    # Make sure input tensor is on the correct device
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    gradients = []
    activations = []
    
    # Register hooks to get gradients and activations from the last stage
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks on the last stage's last layer
    target_layer = model.module.backbone.stages[-1].blocks[-1]
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    model.eval()
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[:, class_idx].backward()
    
    # Get gradients and activations
    grad = gradients[0]
    act = activations[0]
    
    # Calculate weights
    weights = grad.mean(dim=(2, 3), keepdim=True)
    
    # Generate CAM
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    return cam[0, 0].detach().cpu().numpy()


def visualize_cam(image, cam):
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5) * 255
    img_np = img_np.astype(np.uint8)

    cam_resized = cv2.resize(cam, (224, 224))
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, cam_heatmap, 0.4, 0)
    return overlay


def train_epoch_amp(model, dataloader, optimizer, scheduler, criterion, device, scaler, epoch=None):
    """
    使用混合精度訓練的訓練函數，可加速訓練並降低記憶體使用量
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # 使用混合精度前向傳播
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 使用GradScaler處理反向傳播
        scaler.scale(loss).backward()
        
        # 在更新前應用梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # 更新參數
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Train Loss: {total_loss / len(dataloader):.3f} | Train Accuracy: {accuracy:.2f}%")
    
    # 更新調度器 - 傳入 epoch 參數給 scheduler.step()
    scheduler.step(epoch)
    
    return accuracy


# Social Event Dataset for SimCLR
class SocialEventDataset(Dataset):
    def __init__(self, image_folder, transform=None, two_views_transform=None):
        """
        Dataset for Social Event Images with SimCLR approach
        :param image_folder: (str) Folder containing images
        :param transform: (callable) Single view transformation
        :param two_views_transform: (callable) Two views transformation for contrastive learning
        """
        self.image_paths = self._get_image_paths(image_folder)
        self.transform = transform
        self.two_views_transform = two_views_transform
    
    def _get_image_paths(self, folder):
        """Find all image files in the folder"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(folder, ext)))
        
        print(f"Found {len(all_images)} images in {folder}")
        return all_images
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Return two augmented versions of the same image for contrastive learning"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.two_views_transform is not None:
            # Get two differently augmented views of the same image
            view1 = self.two_views_transform(image)
            view2 = self.two_views_transform(image)
            return view1, view2
        elif self.transform is not None:
            # Single transformation (for evaluation)
            return self.transform(image)
        else:
            return image


# NT-Xent Loss for Contrastive Learning
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        """
        NT-Xent loss for contrastive learning as used in SimCLR
        :param batch_size: (int) Batch size
        :param temperature: (float) Temperature parameter to scale the similarity scores
        """
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        
        # Create a mask to filter out positive samples from the loss calculation
        self.mask = self._get_correlated_mask().cuda()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self):
        """Create a mask to identify positive pairs"""
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye(2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).bool()
        return mask

    def forward(self, z_i, z_j):
        """
        Calculate NT-Xent loss
        :param z_i: (torch.Tensor) Representations of first views
        :param z_j: (torch.Tensor) Representations of second views
        :return: (torch.Tensor) NT-Xent loss
        """
        # Concatenate views
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Matrix multiplication to get similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # Filter out the positive pairs (diagonal elements)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        
        # Filter out the diagonal elements
        negative_mask = self.mask.clone()
        if negative_mask.device != similarity_matrix.device:
            negative_mask = negative_mask.to(similarity_matrix.device)
            
        negatives = similarity_matrix[negative_mask].view(2 * self.batch_size, -1)
        
        # Concat positive and negative pairs
        logits = torch.cat([positives, negatives], dim=1)
        
        # Labels: positives are at index 0
        labels = torch.zeros(2 * self.batch_size).cuda().long()
        
        # Scale by temperature
        logits = logits / self.temperature
        
        # Calculate cross entropy loss
        loss = self.criterion(logits, labels)
        
        # Normalize the loss
        loss = loss / (2 * self.batch_size)
        
        return loss


# Training functions for contrastive learning
def train_epoch_simclr(model, dataloader, optimizer, criterion, device, scaler):
    """
    Training function for SimCLR contrastive learning
    :param model: (nn.Module) SimCLR model
    :param dataloader: (DataLoader) Dataloader for contrastive pairs
    :param optimizer: (Optimizer) Optimizer
    :param criterion: (nn.Module) Contrastive loss function
    :param device: (torch.device) Device to use
    :param scaler: (GradScaler) Gradient scaler for mixed precision training
    :return: (float) Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for views in tqdm(dataloader, desc="Training"):
        # Get two views of the same images
        view1, view2 = views
        view1, view2 = view1.to(device), view2.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Use mixed precision for forward pass
        with autocast():
            # Get embeddings
            z_i = model(view1)
            z_j = model(view2)
            
            # Calculate loss
            loss = criterion(z_i, z_j)
        
        # Scale loss and do backward pass
        scaler.scale(loss).backward()
        
        # Apply gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update parameters
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


# Evaluation function for feature extraction and clustering
def evaluate_clustering(model, dataloader, device, n_clusters=10):
    """
    Evaluate the learned representations through K-means clustering
    :param model: (nn.Module) SimCLR model
    :param dataloader: (DataLoader) Dataloader for evaluation
    :param device: (torch.device) Device to use
    :param n_clusters: (int) Number of clusters for K-means
    :return: (dict) Dictionary with evaluation metrics
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting Features"):
            # For evaluation, we use single view
            images = images.to(device)
            
            # Get features (before projection head)
            feat = model.forward_features(images)
            
            # Store features
            features.append(feat.cpu().numpy())
    
    # Concatenate all features
    features = np.concatenate(features, axis=0)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    
    # Calculate silhouette score if there are enough samples
    if len(features) > n_clusters:
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(features, kmeans.labels_)
    else:
        silhouette_avg = 0.0
    
    # Calculate inertia (sum of squared distances to closest centroid)
    inertia = kmeans.inertia_
    
    # Return metrics
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'inertia': inertia
    }
    
    print(f"Clustering Metrics: {metrics}")
    return metrics


# Main Program
if __name__ == "__main__":
    # 檢查是否在分散式環境中運行
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    
    try:
        if is_distributed:
            # 在分散式環境中運行
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            
            dist.init_process_group(backend='nccl', init_method='env://')
            print(f"Process initialized with rank {local_rank}")
        else:
            # 在單設備環境中運行
            local_rank = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Running on device: {device}")

        BATCH_SIZE = 64  # 從原本的32增加到64
        IMAGE_SIZE = 224
        IMAGE_ROOT = "food-101/images"
        TRAIN_FILE = "food-101/meta/train.txt"
        TEST_FILE = "food-101/meta/test.txt"

        LABELS = [
            'apple_pie',
            'baby_back_ribs',
            'baklava',
            'beef_carpaccio',
            'beef_tartare',
            'beet_salad',
            'beignets',
            'bibimbap',
            'bread_pudding',
            'breakfast_burrito',
            'bruschetta',
            'caesar_salad',
            'cannoli',
            'caprese_salad',
            'carrot_cake',
            'ceviche',
            'cheesecake',
            'cheese_plate',
            'chicken_curry',
            'chicken_quesadilla',
            'chicken_wings',
            'chocolate_cake',
            'chocolate_mousse',
            'churros',
            'clam_chowder',
            'club_sandwich',
            'crab_cakes',
            'creme_brulee',
            'croque_madame',
            'cup_cakes',
            'deviled_eggs',
            'donuts',
            'dumplings',
            'edamame',
            'eggs_benedict',
            'escargots',
            'falafel',
            'filet_mignon',
            'fish_and_chips',
            'foie_gras',
            'french_fries',
            'french_onion_soup',
            'french_toast',
            'fried_calamari',
            'fried_rice',
            'frozen_yogurt',
            'garlic_bread',
            'gnocchi',
            'greek_salad',
            'grilled_cheese_sandwich',
            'grilled_salmon',
            'guacamole',
            'gyoza',
            'hamburger',
            'hot_and_sour_soup',
            'hot_dog',
            'huevos_rancheros',
            'hummus',
            'ice_cream',
            'lasagna',
            'lobster_bisque',
            'lobster_roll_sandwich',
            'macaroni_and_cheese',
            'macarons',
            'miso_soup',
            'mussels',
            'nachos',
            'omelette',
            'onion_rings',
            'oysters',
            'pad_thai',
            'paella',
            'pancakes',
            'panna_cotta',
            'peking_duck',
            'pho',
            'pizza',
            'pork_chop',
            'poutine',
            'prime_rib',
            'pulled_pork_sandwich',
            'ramen',
            'ravioli',
            'red_velvet_cake',
            'risotto',
            'samosa',
            'sashimi',
            'scallops',
            'seaweed_salad',
            'shrimp_and_grits',
            'spaghetti_bolognese',
            'spaghetti_carbonara',
            'spring_rolls',
            'steak',
            'strawberry_shortcake',
            'sushi',
            'tacos',
            'takoyaki',
            'tiramisu',
            'tuna_tartare',
            'waffles'
        ]

        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # 強化版資料增強的測試集轉換
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        encoder = Label_encoder(LABELS)

        train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder)
        test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)

        train_dataset = Food101Dataset(train_df, encoder, transform)
        # 更新測試資料集使用適合測試的轉換
        test_dataset = Food101Dataset(test_df, encoder, transform_test)

        if is_distributed:
            train_sampler = DistributedSampler(train_dataset)
            test_sampler = DistributedSampler(test_dataset)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
        else:
            train_sampler = None
            test_sampler = None
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        num_epochs = 110

        # Initialize the SwinV2 model
        model = swin_transformer_v2_base_classifier(
            input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
            window_size=7,
            num_classes=len(LABELS),
            use_checkpoint=True
        )
        
        model = model.to(device)
        
        if is_distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            model._set_static_graph()  

        # 使用分層學習率，為骨幹和分類頭設置不同的學習率
        parameters = [
            {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5},  # 骨幹網路較低學習率
            {'params': [p for n, p in model.named_parameters() if 'head' in n], 'lr': 5e-5}  # 分類頭較高學習率
        ]
        
        optimizer = optim.AdamW(
            parameters,
            weight_decay=0.05  # 較高的權重衰減以提供更好的正則化
        )
        
        # 加入標籤平滑提高泛化能力
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 使用更先進的學習率調度器
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=1e-6,
            warmup_t=5,  # 5個epochs的預熱期
            warmup_lr_init=1e-7
        )

        # 初始化混合精度訓練的scaler
        scaler = GradScaler()
        
        best_acc = 0
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            if is_distributed:
                train_sampler.set_epoch(epoch)
                
            train_acc = train_epoch_amp(model, train_loader, optimizer, scheduler, criterion, device, scaler, epoch)
            test_acc = test_epoch(model, test_loader, criterion, device)

            if test_acc > best_acc:
                best_acc = test_acc
                if local_rank == 0 or not is_distributed:
                    torch.save(model.state_dict(), 'swin_v2_model_test.pth')
    
    except Exception as e:
        import traceback
        print(f"Error in main process: {e}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        if is_distributed and torch.distributed.is_initialized():
            destroy_process_group()

    try:
        # 檢查是否在分散式環境中運行
        is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
        
        if is_distributed:
            # 在分散式環境中運行
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            
            # Initialize distributed process group if not already done
            if not torch.distributed.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://')
                
            print(f"Process initialized with rank {local_rank}")
        else:
            # 在單設備環境中運行
            local_rank = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Running SimCLR training on device: {device}")

        # Parameters
        BATCH_SIZE = 64
        IMAGE_SIZE = 224
        # 修改路徑使用現有的 Food-101 數據集而非不存在的 social_event_dataset
        IMAGE_ROOT = "image"  # 使用 Food-101 數據集的路徑
        PROJECTION_DIM = 128
        
        # Define transformations for contrastive learning
        # Two separate strong augmentations for contrastive learning
        two_views_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Less aggressive transformation for evaluation
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        train_dataset = SocialEventDataset(
            image_folder=IMAGE_ROOT, 
            two_views_transform=two_views_transform
        )
        
        eval_dataset = SocialEventDataset(
            image_folder=IMAGE_ROOT,
            transform=eval_transform
        )

        # Create data loaders with appropriate sampler
        if is_distributed:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )
            
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=BATCH_SIZE,
                sampler=eval_sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

        # Number of epochs
        num_epochs = 100

        # Initialize SimCLR model
        model = swin_transformer_v2_base_simclr(
            input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
            window_size=7,
            projection_dim=PROJECTION_DIM,
            use_checkpoint=True
        )
        
        model = model.to(device)
        
        if is_distributed:
            model = nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank], 
                find_unused_parameters=True
            )
            model._set_static_graph()  

        # Initialize NT-Xent loss
        criterion = NT_Xent(batch_size=BATCH_SIZE, temperature=0.07)
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.05  # High weight decay for better regularization
        )
        
        # Learning rate scheduler
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=1e-6,
            warmup_t=10,  # 10 epochs of warmup
            warmup_lr_init=1e-7
        )
        
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Set epoch for distributed sampler
            if is_distributed:
                train_sampler.set_epoch(epoch)
            
            # Train for one epoch
            train_loss = train_epoch_simclr(model, train_loader, optimizer, criterion, device, scaler)
            
            # Update learning rate
            scheduler.step(epoch)
            
            # Save model checkpoint (only on main process)
            if (local_rank == 0 or not is_distributed) and (epoch + 1) % 10 == 0:
                model_to_save = model.module if is_distributed else model
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, 
                    f'swin_v2_simclr_epoch_{epoch+1}.pth'
                )
            
            # Evaluate with clustering (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                model_to_eval = model.module if is_distributed else model
                evaluate_clustering(model_to_eval, eval_loader, device)
                
        # Final evaluation with different cluster counts
        if local_rank == 0 or not is_distributed:
            print("\nFinal Evaluation with Different Cluster Counts")
            model_to_eval = model.module if is_distributed else model
            for n_clusters in [5, 10, 15, 20]:
                evaluate_clustering(model_to_eval, eval_loader, device, n_clusters=n_clusters)
                
            # Save final model
            torch.save(
                {
                    'epoch': num_epochs,
                    'model_state_dict': model_to_eval.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 
                'swin_v2_simclr_final.pth'
            )
                
    except Exception as e:
        import traceback
        print(f"Error in main process: {e}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        if is_distributed and torch.distributed.is_initialized():
            destroy_process_group()
