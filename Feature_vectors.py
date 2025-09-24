# Feature_vectors.py
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Configuration (ADJUST THESE PATHS AND SETTINGS) ---
project_root = ""  # Set this to your project's root path if needed
data_base_dir = os.path.join(project_root, "data")
model_path = os.path.join(project_root, "trained_embeddings/mask_cnn/se_resnext50_classifier_mask_only.pth") 

# Define the image and mask directories
IMAGES_DIR_GOOD = os.path.join(data_base_dir, "good_images")
IMAGES_DIR_BAD = os.path.join(data_base_dir, "bad_images")
MASKS_DIR_GOOD = os.path.join(data_base_dir, "predicted_masks_good")
MASKS_DIR_BAD = os.path.join(data_base_dir, "predicted_masks_bad")

# Output CSV file for the embeddings
EMBEDDINGS_OUTPUT_CSV = os.path.join(project_root, "trained_embeddings/mask_img_cnn/image_embeddings_org_img.csv")

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
# ---------------------------------------------------------------------

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running embedding extraction on: {device}")

# --- 1. Model Architecture (ResNet-50 for RGB images) ---
def get_resnet50_classifier():
    from torchvision import models
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# --- 2. Custom Dataset for Embedding Extraction ---
class EmbeddingDataset(Dataset):
    def __init__(self, bad_images_dir: str, good_images_dir: str, bad_masks_dir: str, good_masks_dir: str, transform=None):
        self.data_items = []
        self.transform = transform
        
        print("\n--- Preparing Embedding Dataset ---")
        
        # Process 'bad' images (label 0)
        print(f"  Scanning bad images from: {bad_images_dir}")
        for img_name in os.listdir(bad_images_dir):
            if img_name.endswith('.png'):
                full_image_path = os.path.join(bad_images_dir, img_name)
                base_name, _ = os.path.splitext(img_name)
                full_mask_path = os.path.join(bad_masks_dir, f"{base_name}.png")
                
                if not os.path.exists(full_mask_path):
                    print(f"    WARNING: Mask not found for {img_name}. Skipping.")
                    continue

                self.data_items.append({
                    'image_path': full_image_path,
                    'mask_path': full_mask_path,
                    'label': 0
                })

        # Process 'good' images (label 1)
        print(f"  Scanning good images from: {good_images_dir}")
        for img_name in os.listdir(good_images_dir):
            if img_name.lower().endswith('.jpg') or img_name.lower().endswith('.jpeg'):
                full_image_path = os.path.join(good_images_dir, img_name)
                base_name, _ = os.path.splitext(img_name)
                full_mask_path = os.path.join(good_masks_dir, f"{base_name}.png")
                
                if not os.path.exists(full_mask_path):
                    print(f"    WARNING: Mask not found for {img_name}. Skipping.")
                    continue

                self.data_items.append({
                    'image_path': full_image_path,
                    'mask_path': full_mask_path,
                    'label': 1
                })
        
        print(f"  Total images for embedding extraction: {len(self.data_items)}")
        print("-" * 50)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        img_path = item['image_path']
        label = item['label']

        # Read image in color (3 channels)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not read image at {img_path}. Skipping this item.")
            return None, None, None

        if self.transform:
            input_tensor = self.transform(image)
        else:
            image = cv2.resize(image, IMAGE_SIZE)
            input_tensor = image.astype(np.float32) / 255.0
            input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)

        return input_tensor, torch.tensor(label, dtype=torch.long), os.path.basename(img_path)

# --- Data transformations ---
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# --- Custom collate_fn ---
def custom_embedding_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.empty(0, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), torch.empty(0, dtype=torch.long), []

    images, labels, filenames = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    return images, labels, list(filenames)

# --- Main logic ---
if __name__ == "__main__":
    # Dataset and DataLoader
    embedding_dataset = EmbeddingDataset(
        bad_images_dir=IMAGES_DIR_BAD,
        good_images_dir=IMAGES_DIR_GOOD,
        bad_masks_dir=MASKS_DIR_BAD,
        good_masks_dir=MASKS_DIR_GOOD,
        transform=data_transforms
    )
    embedding_loader = DataLoader(embedding_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_embedding_collate_fn, num_workers=0)

    if len(embedding_dataset) == 0:
        print("No images found for embeddings. Please check your data and paths.")
        exit()

    # Load the model
    model = get_resnet50_classifier().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from: {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model weights file not found at {model_path}. Please train your model first.")
        exit()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        exit()

    model.eval()
    embedding_model = nn.Sequential(*list(model.children())[:-1])
    embedding_model.to(device)
    print("\nModel modified to output last layer embeddings.")

    # Extract embeddings
    print("\nExtracting embeddings for all images...")
    all_embeddings = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for batch_idx, (images, labels, filenames) in enumerate(embedding_loader):
            if images.numel() == 0:
                continue

            images = images.to(device)
            embeddings_batch = embedding_model(images)
            embeddings_batch = embeddings_batch.view(embeddings_batch.size(0), -1)

            all_embeddings.append(embeddings_batch.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)

            print(f"  Processed batch {batch_idx+1}/{len(embedding_loader)}")

    # Save embeddings to CSV
    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"\nExtracted embeddings shape: {all_embeddings.shape}")

        df_embeddings = pd.DataFrame(all_embeddings)
        df_embeddings.columns = [f'embedding_{i}' for i in range(all_embeddings.shape[1])]
        df_embeddings['label'] = all_labels
        df_embeddings['filename'] = all_filenames

        # Reorder columns
        cols = ['filename', 'label'] + [f'embedding_{i}' for i in range(all_embeddings.shape[1])]
        df_embeddings = df_embeddings[cols]

        os.makedirs(os.path.dirname(EMBEDDINGS_OUTPUT_CSV), exist_ok=True)
        df_embeddings.to_csv(EMBEDDINGS_OUTPUT_CSV, index=False)
        print(f"\nSuccessfully saved embeddings to: {EMBEDDINGS_OUTPUT_CSV}")
    else:
        print("\nNo embeddings were extracted. Check your data and paths.")

    print("\nEmbedding extraction script finished.")
