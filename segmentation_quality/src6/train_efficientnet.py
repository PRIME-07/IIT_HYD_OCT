import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from PIL import Image
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR

class PathCrop:
    """
    A transform that finds the brightest curved path (RPE), crops everything
    above it, and then trims black borders from the top and bottom.
    """
    def __init__(self, smoothing_window_size=31, padding_bottom=20):
        self.smoothing_window_size = smoothing_window_size
        self.padding_bottom = padding_bottom

    def __call__(self, img):
        np_img_color = np.array(img)
        np_img_gray = np.array(img.convert('L'))
        h, w = np_img_gray.shape

        kernel = np.ones(self.smoothing_window_size) / self.smoothing_window_size
        y_coords_raw = np.argmax(np_img_gray, axis=0)
        y_coords_smooth = np.convolve(y_coords_raw, kernel, mode='same')
        
        masked_img_np = np_img_color.copy()
        for x in range(w):
            y_boundary = int(y_coords_smooth[x])
            masked_img_np[:y_boundary, x, :] = 0

        gray_masked = cv2.cvtColor(masked_img_np, cv2.COLOR_RGB2GRAY)
        content_rows = np.where(np.sum(gray_masked, axis=1) > 0)[0]
        
        if len(content_rows) > 0:
            content_top = content_rows[0]
            content_bottom = content_rows[-1]
            content_bottom = min(h, content_bottom + self.padding_bottom)

            if content_top >= content_bottom:
                return img

            final_cropped_img = img.crop((0, content_top, w, content_bottom + 1))
            return final_cropped_img
        else:
            return img

#  Config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "data_split/train"
val_dir = "data_split/val"
num_epochs = 20
batch_size = 32
learning_rate = 1e-4
save_path = "efficientnet_b0_quality.pt"
early_stopping_patience = 4

#  Transforms 
path_cropper = PathCrop(padding_bottom=20)
train_transform = transforms.Compose([
    path_cropper,
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=15,                # Increased rotation range
        translate=(0.1, 0.1),      # Increased translation range
        scale=(0.9, 1.1),          # Added random scaling
        shear=10                   # Added random shearing
    ),
    transforms.ColorJitter(brightness=0.4, contrast=0.4), # Increased intensity
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    path_cropper,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#  Datasets & Loaders 
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

#  Model setup for EfficientNet-B0 
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Freeze all parameters initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last three feature blocks (6, 7, and 8)
for param in model.features[6].parameters():
    param.requires_grad = True

for param in model.features[7].parameters():
    param.requires_grad = True
    
for param in model.features[8].parameters(): # This is the corrected line
    param.requires_grad = True

# Replace and enable training for the classifier head
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

model.to(device)

learning_rate = 5e-5 # Use a slightly smaller base learning rate

optimizer = optim.AdamW([
    {'params': model.features[6].parameters(), 'lr': learning_rate / 10},
    {'params': model.features[7].parameters(), 'lr': learning_rate / 5},
    {'params': model.features[8].parameters(), 'lr': learning_rate / 2}, # This is the corrected line
    {'params': model.classifier.parameters(),  'lr': learning_rate}
], weight_decay=1e-2)

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8)

#  Loss, Optimizer & Scheduler 
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

#  Tracking 
train_acc_list = []
val_acc_list = []
train_loss_list = []
best_val_acc = 0.0
epochs_no_improve = 0
print(f"\nTraining with EfficientNet started on device: {device}\n")

#  Training Loop 
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

#  Plot Learning Curves 
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

#  Final Evaluation on Validation Set 
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