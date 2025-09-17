import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
# Import the functional transforms library for manual control
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# ======================================================================================
# 1. DATASET SETUP (MODIFIED FOR SYNCHRONIZED AUGMENTATIONS)
# ======================================================================================
# This dataset class assumes your data is structured like this:
# /data_root/
#   /good/
#     /images/
#       001.png
#     /masks/
#       001.png
#   /bad/
#     /images/
#       101.png

class OCTQualityDataset(Dataset):
    """
    Custom Dataset for OCT images.
    - 'Good' images have a corresponding mask.
    - 'Bad' images do not have a mask.
    - Augmentations are applied manually in __getitem__ to ensure the image and mask
      receive the exact same spatial transformations (rotation, flips).
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        mask = None
        if label == 1:
            # Construct mask path from image path robustly
            parts = img_path.split(os.sep)
            parts[-2] = 'masks'
            mask_path = os.sep.join(parts)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")

        # Apply the appropriate transformations
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, label, mask

# ======================================================================================
# 2. TRANSFORMATION CLASSES FOR TRAIN AND VALIDATION
# ======================================================================================
class TrainAugmentations:
    def __init__(self, img_size=224):
        self.img_size = img_size
        # Define the ElasticTransform object once
        self.elastic_transformer = transforms.ElasticTransform(alpha=50.0, sigma=5.0)

    def __call__(self, image, mask):
        # --- SYNCHRONIZED AUGMENTATIONS ---
        image = TF.resize(image, [self.img_size, self.img_size])
        if mask is not None:
            mask = TF.resize(mask, [self.img_size, self.img_size])

        if random.random() > 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        angle = random.randint(-45, 45)
        image = TF.rotate(image, angle)
        if mask is not None:
            mask = TF.rotate(mask, angle)

        # --- NEW: ELASTIC TRANSFORMATION ---
        # Apply the same elastic transformation to both image and mask
        if mask is not None:
            # To apply the same transform, we need to get the state of the random number generator
            state = torch.get_rng_state()
            image = self.elastic_transformer(image)
            torch.set_rng_state(state) # Reset state for the mask
            mask = self.elastic_transformer(mask)

        # --- IMAGE-ONLY AUGMENTATIONS ---
        image = transforms.ColorJitter(brightness=0.4, contrast=0.4)(image)

        # --- FINAL CONVERSION AND NORMALIZATION ---
        image = TF.to_tensor(image)
        if mask is not None:
            mask = TF.to_tensor(mask)
        
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image, mask

class ValAugmentations:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, image, mask):
        # Validation only needs resizing, tensor conversion, and normalization
        image = TF.resize(image, [self.img_size, self.img_size])
        if mask is not None:
            mask = TF.resize(mask, [self.img_size, self.img_size])

        image = TF.to_tensor(image)
        if mask is not None:
            mask = TF.to_tensor(mask)
            
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image, mask


# ======================================================================================
# 3. CUSTOM COLLATE FUNCTION
# ======================================================================================
def custom_collate_fn(batch):
    images, labels, masks = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    valid_masks = [m for m in masks if m is not None]
    if valid_masks:
        valid_masks = torch.stack(valid_masks, 0)
    return images, labels, valid_masks

# ======================================================================================
# 4. MODEL DEFINITION WITH GRAD-CAM HOOKS
# ======================================================================================
class AttentionGuidedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

        self.feature_maps = None
        self.gradients = None

        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.feature_maps = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def forward(self, x):
        return self.model(x)

    def get_attention_map(self, target_indices=None):
        if self.feature_maps is None or self.gradients is None:
            return None
        
        feats = self.feature_maps[target_indices]
        grads = self.gradients[target_indices]

        pooled_gradients = torch.mean(grads, dim=[2, 3])
        
        for i in range(feats.shape[1]):
            feats[:, i, :, :] *= pooled_gradients[:, i].unsqueeze(-1).unsqueeze(-1)
            
        heatmap = torch.mean(feats, dim=1)
        heatmap = F.relu(heatmap)
        
        max_vals = torch.amax(heatmap.view(heatmap.shape[0], -1), dim=1, keepdim=True)
        heatmap = heatmap.view(heatmap.shape[0], *heatmap.shape[1:]) / (max_vals.unsqueeze(-1) + 1e-8)
            
        return heatmap

# ======================================================================================
# 5. CUSTOM ATTENTION LOSS
# ======================================================================================
class AttentionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attention_map, mask):
        mask_resized = F.interpolate(mask, size=attention_map.shape[1:], mode='bilinear', align_corners=False)
        mask_squeezed = mask_resized.squeeze(1)
        intersection = torch.sum(attention_map * mask_squeezed, dim=[1, 2])
        union = torch.sum(attention_map, dim=[1, 2]) + torch.sum(mask_squeezed, dim=[1, 2])
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice_score.mean()

# ======================================================================================
# 6. TRAINING SCRIPT
# ======================================================================================
def train_model(root_dir, num_epochs=25, batch_size=8, learning_rate=1e-4, lambda_attn=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"The specified data directory does not exist: {root_dir}")

    # --- GATHER ALL DATA AND SPLIT ---
    all_image_paths = []
    all_labels = []
    good_image_dir = os.path.join(root_dir, 'good', 'images')
    for img_name in os.listdir(good_image_dir):
        all_image_paths.append(os.path.join(good_image_dir, img_name))
        all_labels.append(1)
    bad_image_dir = os.path.join(root_dir, 'bad', 'images')
    for img_name in os.listdir(bad_image_dir):
        all_image_paths.append(os.path.join(bad_image_dir, img_name))
        all_labels.append(0)

    # Use stratify to ensure the same ratio of good/bad images in train and val sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    # --- CREATE DATASETS AND DATALOADERS ---
    train_dataset = OCTQualityDataset(train_paths, train_labels, transform=TrainAugmentations(img_size=224))
    val_dataset = OCTQualityDataset(val_paths, val_labels, transform=ValAugmentations(img_size=224))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    model = AttentionGuidedModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_attn = AttentionLoss()
    
    # --- LEARNING RATE SCHEDULER (verbose argument removed) ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
    
    # --- EARLY STOPPING PARAMETERS ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5 # Number of epochs to wait for improvement before stopping

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_cls_loss, total_attn_loss = 0.0, 0.0
        for images, labels, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            loss_cls = criterion_cls(outputs, labels)
            
            loss_attn = torch.tensor(0.0, device=device)
            good_image_indices = (labels.squeeze() == 1).nonzero(as_tuple=True)[0]
            
            if len(good_image_indices) > 0 and len(masks) > 0:
                masks = masks.to(device)
                good_outputs = outputs[good_image_indices]
                good_outputs.sum().backward(retain_graph=True)
                
                attention_maps = model.get_attention_map(target_indices=good_image_indices)
                loss_attn = criterion_attn(attention_maps, masks)

            total_loss = loss_cls + lambda_attn * loss_attn
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_cls_loss += loss_cls.item()
            total_attn_loss += loss_attn.item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion_cls(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                preds = torch.sigmoid(outputs) > 0.5
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        # --- EPOCH SUMMARY ---
        avg_train_cls_loss = total_cls_loss / len(train_loader)
        avg_train_attn_loss = total_attn_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_samples) * 100

        print(f"Epoch {epoch+1} Summary: "
              f"Train Cls Loss: {avg_train_cls_loss:.4f}, "
              f"Train Attn Loss: {avg_train_attn_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

        # --- STEP THE SCHEDULER ---
        scheduler.step(avg_val_loss)

        # --- EARLY STOPPING AND SAVE BEST MODEL LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_oct_quality_model.pth")
            print(f"Validation loss decreased. Saving best model with Val Loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("Training finished.")

if __name__ == '__main__':
    # ======================================================================================
    # IMPORTANT: SET YOUR DATA PATH HERE
    # ======================================================================================
    DATA_ROOT = "seg_qual_v2/data" # <-- CHANGE THIS TO YOUR DATASET'S ROOT FOLDER
    
    # It's recommended to run this inside the main block for Windows multiprocessing
    train_model(root_dir=DATA_ROOT, num_epochs=25, batch_size=8, lambda_attn=1.5)
