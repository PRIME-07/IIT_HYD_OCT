import torch
import os
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from torch import nn

# --- REQUIRED: Add ALL class definitions from the training script ---
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

class ResNetWithCBAM(nn.Module):
    def __init__(self, original_model):
        super(ResNetWithCBAM, self).__init__()
        self.features_pre = nn.Sequential(*list(original_model.children())[:-4])
        self.layer3 = original_model.layer3
        self.cbam3 = CBAM(1024)
        self.layer4 = original_model.layer4
        self.cbam4 = CBAM(2048)
        self.avgpool = original_model.avgpool
        self.fc = nn.Linear(2048, 2)

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
# --- End of required definitions ---


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loading
# 1. Build the model structure first.
base_model = models.resnet50(weights=None)
model = ResNetWithCBAM(base_model)

# 2. Load the state dict from your saved attention model
model.load_state_dict(torch.load("resnet50_attention_quality.pt", map_location=device))
model.eval().to(device)

# Image preprocessing (back to original)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- The rest of your testing code remains exactly the same ---
# Class labels
class_names = ['Bad', 'Good']

# Directory containing poor quality images
poor_dir = os.path.join(os.path.dirname(__file__), '../src/dataset', 'poor_seg_data')

# Gather all .png files
image_files = [f for f in os.listdir(poor_dir) if f.lower().endswith('.png')]

# Store results
preds = []
confs = []

for img_name in image_files:
    img_path = os.path.join(poor_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax().item()
        confidence = probs[0, pred_class].item()
    preds.append(pred_class)
    confs.append(confidence)

# All images are poor, so true label is 0 (Bad)
true_labels = [0] * len(preds)

# Calculate accuracy
correct = sum([p == 0 for p in preds])
accuracy = correct / len(preds)
highest_conf = max(confs)
lowest_conf = min(confs)

# Confusion matrix
cm = confusion_matrix(true_labels, preds, labels=[0,1])

# Save results
with open(os.path.join(os.path.dirname(__file__), 'poor_test_results.txt'), 'w') as f:
    f.write(f"Total images: {len(preds)}\n")
    f.write(f"Average accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Highest confidence: {highest_conf*100:.2f}%\n")
    f.write(f"Lowest confidence: {lowest_conf*100:.2f}%\n")
    f.write("\nConfusion Matrix (rows: true, cols: predicted)\n")
    f.write(str(cm))

print(f"Tested {len(preds)} images. Average accuracy: {accuracy*100:.2f}%.") 