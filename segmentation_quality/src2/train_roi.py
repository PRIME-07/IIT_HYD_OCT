import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from PIL import Image

# --- NEW: Custom Transform for ROI Cropping ---
class ROICrop:
    """
    A transform to find the brightest horizontal line (RPE in OCT) and crop around it.
    """
    def __init__(self, crop_height=128):
        self.crop_height = crop_height

    def __call__(self, img):
        # Convert PIL Image to a numpy array (L for grayscale) to find the RPE
        np_img = np.array(img.convert('L'))
        h, w = np_img.shape

        # Compute vertical projection profile by summing pixel intensities along each row
        vertical_profile = np_img.sum(axis=1)
        
        # Find the row index with the maximum sum, which corresponds to the RPE
        y_rpe = np.argmax(vertical_profile)

        # Define crop boundaries, ensuring they are within the image dimensions
        half_height = self.crop_height // 2
        top = max(0, y_rpe - half_height)
        bottom = top + self.crop_height
        
        # Adjust if the crop goes past the bottom edge
        if bottom > h:
            bottom = h
            top = max(0, h - self.crop_height)

        # Crop the original PIL image using the calculated boundaries
        # The crop is (left, top, right, bottom)
        cropped_img = img.crop((0, top, w, bottom))
        return cropped_img

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "data_split/train"
val_dir = "data_split/val"
num_epochs = 15
batch_size = 32
learning_rate = 1e-5
save_path = "resnet50_roi_quality.pt" # Changed save path for new model
early_stopping_patience = 3

# --- MODIFIED: Integrate ROI Cropping into transforms ---
# Instantiate the custom transform
roi_cropper = ROICrop(crop_height=128) # You can tune this height

train_transform = transforms.Compose([
    roi_cropper, # Apply ROI crop first
    transforms.Resize((224, 224)), # Resize the cropped region
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    roi_cropper, # Apply the same crop to validation data
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets & Loaders
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Model setup (remains the same)
model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name:
        param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# Loss and optimizer (remains the same)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# --- The rest of your training, validation, plotting, and evaluation code remains exactly the same ---
# Tracking
train_acc_list = []
val_acc_list = []
train_loss_list = []

# Training
best_val_acc = 0.0
epochs_no_improve = 0
print(f"\nTraining started on device: {device}\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", leave=False)
    for inputs, labels in train_loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        train_loop.set_postfix({
            "loss": running_loss / (total // batch_size + 1),
            "acc": 100. * correct / total
        })

    epoch_train_acc = correct / total
    epoch_train_loss = running_loss / len(train_loader)
    train_acc_list.append(epoch_train_acc)
    train_loss_list.append(epoch_train_loss)

    # Validation accuracy 
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()

    epoch_val_acc = val_correct / len(val_data)
    val_acc_list.append(epoch_val_acc)

    print(f"Epoch {epoch+1}: Train Acc = {epoch_train_acc:.4f}, Val Acc = {epoch_val_acc:.4f}")

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at epoch {epoch+1} (Val Acc: {epoch_val_acc:.4f})\n")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement. ({epochs_no_improve}/{early_stopping_patience} patience)\n")

    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch+1}.\n")
        break

print("Training completed.\n")

# Plot learning curves
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(train_loss_list, label="Train Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()

plt.tight_layout()
plt.show()


# Evaluation metrics

print("Evaluating best model on validation set...\n")
model.load_state_dict(torch.load(save_path))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())

# Compute metrics
final_acc = accuracy_score(y_true, y_pred)
final_precision = precision_score(y_true, y_pred)
final_recall = recall_score(y_true, y_pred)
final_f1 = f1_score(y_true, y_pred)
final_conf_matrix = confusion_matrix(y_true, y_pred)

print("Final Evaluation Metrics:")
print(f"Accuracy  : {final_acc:.4f}")
print(f"Precision : {final_precision:.4f}")
print(f"Recall    : {final_recall:.4f}")
print(f"F1 Score  : {final_f1:.4f}")
print("Confusion Matrix:")
print(final_conf_matrix)