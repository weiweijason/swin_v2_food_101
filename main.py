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
from torch.distributed import init_process_group, destroy_process_group

# Import the SwinV2 classifier
from swin_transformer_v2_classifier import swin_transformer_v2_base_classifier

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


# Generate CAM for Swin Transformer V2
def generate_cam_swin_v2(model, input_tensor, class_idx=None):
    """
    Generate Class Activation Map for Swin Transformer V2
    :param model: The SwinV2 model
    :param input_tensor: Input image tensor [1, C, H, W]
    :param class_idx: Target class index (None for predicted class)
    :return: CAM as numpy array
    """
    gradients = []
    activations = []

    # Register hooks to get gradients and activations from the last stage
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks on the last stage's last layer
    target_layer = model.backbone.stages[-1].blocks[-1]
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


# Main Program
if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    IMAGE_ROOT = "food-101/images"
    TRAIN_FILE = "food-101/meta/train.txt"
    TEST_FILE = "food-101/meta/test.txt"

    LABELS =  [
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

    try:
<<<<<<< HEAD
        # Get the local rank from environment variable
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        print(f"Process initialized with rank {local_rank}")

=======
        # Initialize process group
        print("Initializing distributed process group...")
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Process initialized with rank {local_rank}, using device: {device}")
        
>>>>>>> fe1c4933c2c7ab9ee3481f26818a32ad7ff96431
        BATCH_SIZE = 32
        IMAGE_SIZE = 224
        IMAGE_ROOT = "food-101/images"
        TRAIN_FILE = "food-101/meta/train.txt"
        TEST_FILE = "food-101/meta/test.txt"
<<<<<<< HEAD

        LABELS = [
                'apple_pie',
                'baby_back_ribs',
                # ... rest of your labels ...
                'waffles'
                ]

=======
        
        # [LABELS definition remains the same]
        
>>>>>>> fe1c4933c2c7ab9ee3481f26818a32ad7ff96431
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
<<<<<<< HEAD
                std=[0.229, 0.224, 0.225])
            ])

        encoder = Label_encoder(LABELS)

        train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder)
        test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)

        train_dataset = Food101Dataset(train_df, encoder, transform)
        test_dataset = Food101Dataset(test_df, encoder, transform)

        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

        num_epochs = 100

        # Initialize the SwinV2 model
        model = swin_transformer_v2_base_classifier(
                input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
                window_size=7,
                num_classes=len(LABELS),
                use_checkpoint=True
                )

        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

=======
                                std=[0.229, 0.224, 0.225])
        ])
        
        encoder = Label_encoder(LABELS)
        
        train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder)
        test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder)
        
        train_dataset = Food101Dataset(train_df, encoder, transform)
        test_dataset = Food101Dataset(test_df, encoder, transform)
        
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
        
        num_epochs = 30
        
        # Initialize the SwinV2 model
        model = swin_transformer_v2_base_classifier(
            input_resolution=(IMAGE_SIZE, IMAGE_SIZE),
            window_size=7,
            num_classes=len(LABELS),
            use_checkpoint=True
        )
        
        # Explicitly move model to the correct device
        model = model.to(device)
        print(f"Model moved to device: {next(model.parameters()).device}")
        
        # Wrap with DDP after moving to CUDA
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Set to True only if needed
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss().to(device)  # Move criterion to device
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
>>>>>>> fe1c4933c2c7ab9ee3481f26818a32ad7ff96431
        best_acc = 0
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_sampler.set_epoch(epoch)
            train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            test_acc = test_epoch(model, test_loader, criterion, device)
<<<<<<< HEAD

            if test_acc > best_acc:
                best_acc = test_acc
                if local_rank == 0:
                    torch.save(model.state_dict(), 'swin_v2_model_test.pth')

=======
            
            if test_acc > best_acc:
                best_acc = test_acc
                if local_rank == 0:  # Only save on rank 0
                    torch.save(model.state_dict(), 'swin_v2_model_test.pth')
                    
>>>>>>> fe1c4933c2c7ab9ee3481f26818a32ad7ff96431
    except Exception as e:
        import traceback
        print(f"Error in main process: {e}")
        print(traceback.format_exc())
    finally:
<<<<<<< HEAD
        # Cleanup
        if torch.distributed.is_initialized():
            destroy_process_group()
=======
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
>>>>>>> fe1c4933c2c7ab9ee3481f26818a32ad7ff96431
