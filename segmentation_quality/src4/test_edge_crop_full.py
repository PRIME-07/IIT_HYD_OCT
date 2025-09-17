import torch
import os
import numpy as np
from torchvision import models, transforms, datasets
from PIL import Image
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Your EdgeCrop class definition remains the same ---
# Note: Ensure this class definition is the one you trained the model with.
# If you used MedianBlur, update it here.
class EdgeCrop:
    """
    A transform to find the main tissue band using Canny edge detection and crop it.
    """
    def __init__(self, padding=20, blur_kernel=(5, 5), canny_threshold1=50, canny_threshold2=150):
        self.padding = padding
        self.blur_kernel = blur_kernel
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2

    def __call__(self, img):
        np_img = np.array(img.convert('L'))
        h, w = np_img.shape
        blurred_img = cv2.GaussianBlur(np_img, self.blur_kernel, 0)
        edges = cv2.Canny(blurred_img, self.canny_threshold1, self.canny_threshold2)
        edge_profile = np.sum(edges, axis=1)
        significant_rows = np.where(edge_profile > (w * 0.10))[0]
        
        if len(significant_rows) > 0:
            top_boundary = significant_rows.min()
            bottom_boundary = significant_rows.max()
        else:
            top_boundary = h // 4
            bottom_boundary = h - (h // 4)

        top_crop = max(0, top_boundary - self.padding)
        bottom_crop = min(h, bottom_boundary + self.padding)
        cropped_img = img.crop((0, top_crop, w, bottom_crop))
        
        return cropped_img

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32 # You can adjust this based on your GPU memory

# --- Model Loading ---
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_edgecrop_quality.pt", map_location=device))
model.eval().to(device)

# --- Preprocessing (Must match validation transform from training) ---
transform = transforms.Compose([
    EdgeCrop(padding=20),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- MODIFIED: Data Loading Logic ---
# 1. SET THE PATH to the parent folder containing 'good' and 'bad' subfolders
test_dir = os.path.join(os.getcwd(), 'src/', 'dataset_copy') 
results_file = os.path.join(os.getcwd(), 'full_test_results.txt')

if not os.path.exists(test_dir):
    print(f"Error: Test directory not found at {test_dir}")
else:
    # 2. Create a dataset and dataloader
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # ImageFolder assigns labels alphabetically. Print to confirm.
    # e.g., {'Bad': 0, 'Good': 1}
    print(f"Classes found: {test_data.class_to_idx}")
    class_names = [name for name, _ in sorted(test_data.class_to_idx.items(), key=lambda item: item[1])]

    # --- MODIFIED: Prediction Loop ---
    y_pred = []
    y_true = []
    y_confs = []

    print(f"Starting evaluation on {len(test_data)} images...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Get the confidence score for the predicted class
            confidence_scores = probs.gather(1, predicted.view(-1, 1)).squeeze()

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            y_confs.extend(confidence_scores.cpu().numpy())

    # --- MODIFIED: Metrics Calculation ---
    if len(y_pred) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        highest_conf = max(y_confs)
        lowest_conf = min(y_confs)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Save and print results
        with open(results_file, 'w') as f:
            f.write(f"Total images tested: {len(y_pred)}\n")
            f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
            f.write(f"Highest confidence on a prediction: {highest_conf*100:.2f}%\n")
            f.write(f"Lowest confidence on a prediction: {lowest_conf*100:.2f}%\n\n")
            f.write("Confusion Matrix:\n")
            f.write("           Predicted:\n")
            f.write(f"           {class_names[0]:<5}  {class_names[1]:<5}\n")
            f.write(f"Actual {class_names[0]:<5}: {cm[0][0]:<5}  {cm[0][1]:<5}\n")
            f.write(f"Actual {class_names[1]:<5}: {cm[1][0]:<5}  {cm[1][1]:<5}\n")

        print(f"Evaluation complete. Results saved to {results_file}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print("\nConfusion Matrix:")
        print("           Predicted:")
        print(f"           {class_names[0]:<5}  {class_names[1]:<5}")
        print(f"Actual {class_names[0]:<5}: {cm[0][0]:<5}  {cm[0][1]:<5}")
        print(f"Actual {class_names[1]:<5}: {cm[1][0]:<5}  {cm[1][1]:<5}")
    else:
        print("No images found in the directory")