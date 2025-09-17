import torch
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("src/resnet50_seg_quality.pt", map_location=device))
model.eval().to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Class labels
class_names = ['Bad', 'Good']

# Directory containing poor quality images
poor_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'poor_seg_data')

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
with open(os.path.join(os.path.dirname(__file__), 'test_results_2.txt'), 'w') as f:
    f.write(f"Total images: {len(preds)}\n")
    f.write(f"Average accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Highest confidence: {highest_conf*100:.2f}%\n")
    f.write(f"Lowest confidence: {lowest_conf*100:.2f}%\n")
    f.write("\nConfusion Matrix (rows: true, cols: predicted)\n")
    f.write(str(cm))

print(f"Tested {len(preds)} images. Average accuracy: {accuracy*100:.2f}%.") 