import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- NEW: CBAM (Convolutional Block Attention Module) Implementation ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --- NEW: Wrapper model to insert CBAM into ResNet ---
class ResNetWithCBAM(nn.Module):
    def __init__(self, original_model):
        super(ResNetWithCBAM, self).__init__()
        # Break down the original model
        self.features_pre = nn.Sequential(*list(original_model.children())[:-4]) # Everything before layer3
        self.layer3 = original_model.layer3
        self.cbam3 = CBAM(1024) # After layer3 (1024 channels)
        self.layer4 = original_model.layer4
        self.cbam4 = CBAM(2048) # After layer4 (2048 channels)
        self.avgpool = original_model.avgpool
        self.fc = nn.Linear(2048, 2) # New final layer

    def forward(self, x):
        x = self.features_pre(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "data_split/train"
val_dir = "data_split/val"
num_epochs = 15
batch_size = 32
learning_rate = 1e-5
save_path = "resnet50_attention_quality.pt" # New save path
early_stopping_patience = 3

# --- MODIFIED: Reverted to original transforms ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets & Loaders
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# --- MODIFIED: Model setup ---
# 1. Load a standard pretrained ResNet50
base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

# 2. Create our new model with CBAM blocks
model = ResNetWithCBAM(base_model)

# 3. Set up fine-tuning: freeze early layers, unfreeze layer3, layer4, and CBAM blocks
for param in model.parameters():
    param.requires_grad = False

for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.cbam3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.cbam4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model.to(device)

# Loss and optimizer
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