import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
import timm
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.distributed import init_process_group, destroy_process_group

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

        try:
            image = Image.open(img_path).convert("RGB")
        except OSError as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            print(f"Skipping corrupted image: {img_path}")
            return None # Mark as invalid sample

        if self.transform:
            image = self.transform(image)

        return image, label


# Updated prepare_dataframe function for Food-101 format
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
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Train Loss: {total_loss / len(dataloader):.3f} | Train Accuracy: {accuracy:.2f}%")


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


# Generate CAM for Vision Transformer
def generate_cam(model, input_tensor, target_layer, class_idx=None):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[:, class_idx].backward()

    grad = gradients[0]
    act = activations[0]
    
    if len(grad.shape) == 4: # CNN-like features [B, C, H, W]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
    elif len(grad.shape) == 3: # Transformer-like features [B, N, C]
        pooled_gradients = torch.mean(grad, dim=[1]) # Global average pooling over N (tokens) -> (B,D)
        cam = torch.einsum('bnd,bd->bn', act, pooled_gradients) # (B,N)
        # For ViT, if target_layer.output is (B, N, D), where N is num_patches+cls_token
        num_patches = cam.shape[1]
        if num_patches > 1:
            side = int(np.sqrt(num_patches))
            if side * side == num_patches:
                cam = cam.view(cam.shape[0], side, side)
            else:
                print(f"Warning: CAM for ViT has {num_patches} tokens, cannot reshape to square. Using 1D CAM.")
                cam = cam.unsqueeze(-1)
    else:
        raise ValueError("Unexpected gradient shape: {}".format(grad.shape))

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()

    return cam[0].detach().cpu().numpy()


def visualize_cam(image, cam):
    img_np = image.cpu().permute(1, 2, 0).numpy()
    # Denormalize based on Food101 transform
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)

    if cam.ndim == 1:
        print("Visualizing 1D CAM. For ViT, this represents patch importance.")
        cam_resized = cv2.resize(cam[:, np.newaxis], (50, 224))
    elif cam.ndim == 2:
        cam_resized = cv2.resize(cam, (224, 224))
    else:
        raise ValueError(f"CAM has unexpected dimensions: {cam.ndim}")

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    if img_np.shape[:2] != (224, 224):
        img_np_resized = cv2.resize(img_np, (224, 224))
    else:
        img_np_resized = img_np

    if cam_heatmap.shape[:2] != img_np_resized.shape[:2]:
        cam_heatmap = cv2.resize(cam_heatmap, (img_np_resized.shape[1], img_np_resized.shape[0]))

    overlay = cv2.addWeighted(img_np_resized, 0.6, cam_heatmap, 0.4, 0)
    return overlay


# Main Program
if __name__ == "__main__":
    use_distributed = False
    local_rank = 0
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        use_distributed = True
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"分散式訓練已初始化，local_rank: {local_rank}")
    else:
        print("使用單一GPU訓練模式")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {device}")

    BATCH_SIZE = 32
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
        'cheese_plate',
        'cheesecake',
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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    encoder = Label_encoder(LABELS)

    train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder)
    test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)

    train_dataset = Food101Dataset(train_df, encoder, transform)
    test_dataset = Food101Dataset(test_df, encoder, transform)

    def collate_fn_skip_corrupted(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return torch.Tensor(), torch.Tensor()
        return torch.utils.data.dataloader.default_collate(batch)

    if use_distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn_skip_corrupted, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn_skip_corrupted, num_workers=4)
        device = torch.device(f"cuda:{local_rank}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_skip_corrupted, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_skip_corrupted, num_workers=4)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = 30
    
    # Vision Transformer model
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(LABELS))
    model = model.to(device)
    
    target_layer = None
    if use_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        target_layer = model.module.norm
    else:
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 個 GPU 運行 DataParallel")
            model = nn.DataParallel(model)
            target_layer = model.module.norm
        else:
            target_layer = model.norm
            
    if target_layer is None:
        print("警告: CAM 的 target_layer 未能成功設定。CAM 可能無法正常運作。")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        if use_distributed:
            train_sampler.set_epoch(epoch)
        
        train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        test_acc = test_epoch(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc
            if not use_distributed or local_rank == 0:
                torch.save(model.state_dict(), 'vit_model_food101.pth')
                print(f"模型已保存，準確率: {best_acc:.2f}%")
    
    # Example of generating and visualizing CAM for one image from the test set
    if not use_distributed or local_rank == 0:
        if target_layer is not None and len(test_dataset) > 0:
            print("\nGenerating CAM for a sample image...")
            try:
                sample_img, sample_label_idx = test_dataset[0]
                if sample_img is None:
                    for i in range(1, min(10, len(test_dataset))):
                        sample_img, sample_label_idx = test_dataset[i]
                        if sample_img is not None:
                            break
                
                if sample_img is not None:
                    sample_img_tensor = sample_img.unsqueeze(0).to(device)

                    model_for_cam = model
                    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                        model_for_cam = model.module
                    model_for_cam.eval()

                    cam_output = generate_cam(model_for_cam, sample_img_tensor, target_layer)
                    
                    overlay_image = visualize_cam(sample_img, cam_output)

                    plt.figure(figsize=(6,6))
                    plt.imshow(overlay_image)
                    plt.title(f"CAM for: {encoder.get_label(sample_label_idx)}")
                    plt.axis('off')
                    plt.savefig("vit_cam_example.png")
                    print("CAM example saved to vit_cam_example.png")
                else:
                    print("Could not obtain a valid sample image for CAM generation.")

            except Exception as e:
                print(f"Error generating CAM: {e}")
        elif target_layer is None:
            print("CAM target_layer not set, skipping CAM generation.")
        else:
            print("Test dataset is empty, skipping CAM generation.")

    if use_distributed and torch.distributed.is_initialized():
        destroy_process_group()

print("程式執行完畢。")