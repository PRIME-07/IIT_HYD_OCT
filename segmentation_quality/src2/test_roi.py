import torch
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix

# --- NEW: Custom Transform for ROI Cropping (MUST BE IDENTICAL TO TRAINING) ---
class ROICrop:
    """
    A transform to find the brightest horizontal line (RPE in OCT) and crop around it.
    """
    def __init__(self, crop_height=128):
        self.crop_height = crop_height

    def __call__(self, img):
        np_img = np.array(img.convert('L'))
        h, w = np_img.shape
        vertical_profile = np_img.sum(axis=1)
        y_rpe = np.argmax(vertical_profile)
        half_height = self.crop_height // 2
        top = max(0, y_rpe - half_height)
        bottom = top + self.crop_height
        if bottom > h:
            bottom = h
            top = max(0, h - self.crop_height)
        cropped_img = img.crop((0, top, w, bottom))
        return cropped_img

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
# IMPORTANT: Load the new model trained on cropped images
model.load_state_dict(torch.load("resnet50_roi_quality.pt", map_location=device))
model.eval().to(device)

# --- MODIFIED: Image preprocessing to match validation transform ---
transform = transforms.Compose([
    ROICrop(crop_height=128),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Class labels
class_names = ['Bad', 'Good']

# Directory containing poor quality images
# Assuming the script is in the root of your project
poor_dir = os.path.join(os.getcwd(), './src/dataset', 'poor_seg_data')

# The rest of your testing and reporting code remains exactly the same 
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