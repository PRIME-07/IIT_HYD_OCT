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
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- MODIFIED: EdgeCrop class using Contour Detection ---
class EdgeCrop:
    """
    A more robust transform that uses morphological closing and contour detection
    to find the main tissue band and crop it.
    """
    def __init__(self, padding=20, blur_kernel_size=5, canny_threshold1=40, canny_threshold2=120):
        self.padding = padding
        self.blur_kernel_size = blur_kernel_size
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2

    def __call__(self, img):
        np_img = np.array(img.convert('L'))
        h, w = np_img.shape

        blurred_img = cv2.medianBlur(np_img, self.blur_kernel_size)
        edges = cv2.Canny(blurred_img, self.canny_threshold1, self.canny_threshold2)

        # Morphological Closing to connect broken edges
        kernel = np.ones((5, 11), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, cont_w, cont_h = cv2.boundingRect(largest_contour)
            top_boundary = y
            bottom_boundary = y + cont_h
        else:
            top_boundary = h // 4
            bottom_boundary = h - (h // 4)

        top_crop = max(0, top_boundary - self.padding)
        bottom_crop = min(h, bottom_boundary + self.padding)
        cropped_img = img.crop((0, top_crop, w, bottom_crop))
        
        return cropped_img

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "data_split/train"
val_dir = "data_split/val"
num_epochs = 20
batch_size = 32
learning_rate = 1e-5
save_path = "resnet50_contour_crop_quality.pt" # save model
early_stopping_patience = 3

# --- Transforms ---
edge_cropper = EdgeCrop(padding=20)
train_transform = transforms.Compose([
    edge_cropper,
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    edge_cropper,
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets & Loaders
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Model setup (Standard ResNet50)
model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name:
        param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

# Tracking 
train_acc_list = []
val_acc_list = []
train_loss_list = []
best_val_acc = 0.0
epochs_no_improve = 0
print(f"\nTraining with Contour Crop started on device: {device}\n")

# Training Loop 
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
        train_loop.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.2f}%"})

    scheduler.step()

    epoch_train_acc = correct / total
    epoch_train_loss = running_loss / len(train_loader)
    train_acc_list.append(epoch_train_acc)
    train_loss_list.append(epoch_train_loss)

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
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
        print(f"Model saved at epoch {epoch+1} (Val Acc: {best_val_acc:.4f})\n")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement. ({epochs_no_improve}/{early_stopping_patience} patience)\n")

    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch+1}.\n")
        break

print("Training completed.\n")

# Plot Learning Curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_loss_list, label="Train Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Final Evaluation on Validation Set 
print("Evaluating best model on validation set...\n")
model.load_state_dict(torch.load(save_path))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())

# Compute and print metrics
print("Final Evaluation Metrics:")
print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
print(f"  Recall: {recall_score(y_true, y_pred):.4f}")
print(f"  F1 Score: {f1_score(y_true, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))