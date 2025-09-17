import torch
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import itertools

# --- Configuration ---
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Path Configuration ---
# NOTE: These relative paths assume you run the script from the project root 
# (i.e., the 'segementation_quality' folder) using a command like:
# python src/seg_quality_classifier_test_g_and_b.py

# Model path
MODEL_PATH = "src/resnet50_seg_quality.pt"

# Data directories
poor_dir = "src/dataset/poor_seg_data"
good_dir = "src/dataset/good_seg_data"

# Output file for results
RESULTS_FILE = "src/test_results_2.txt"

# Class labels (0: Bad, 1: Good)
CLASS_NAMES = ['Bad', 'Good']
# --- End Configuration ---


# --- Model Loading ---
print("Loading model...")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)
print("Model loaded successfully.")

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalization should match what was used during training
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Data Preparation ---
# Create a list of all images with their true labels
# Label 0 for 'poor' and Label 1 for 'good'
images_to_test = []

# Gather poor quality images
if os.path.isdir(poor_dir):
    poor_files = [os.path.join(poor_dir, f) for f in os.listdir(poor_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_to_test.extend(zip(poor_files, itertools.repeat(0))) # Add (filepath, label)
    print(f"Found {len(poor_files)} images in 'poor' directory.")
else:
    print(f"Warning: Directory not found: {poor_dir}")

# Gather good quality images
if os.path.isdir(good_dir):
    good_files = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_to_test.extend(zip(good_files, itertools.repeat(1))) # Add (filepath, label)
    print(f"Found {len(good_files)} images in 'good' directory.")
else:
    print(f"Warning: Directory not found: {good_dir}")


# --- Inference and Evaluation ---
preds = []
true_labels = []
confs = []

print(f"\nStarting inference on {len(images_to_test)} total images...")

for img_path, true_label in images_to_test:
    try:
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax().item()
            confidence = probs[0, pred_class].item()
            
        preds.append(pred_class)
        true_labels.append(true_label)
        confs.append(confidence)
    except Exception as e:
        print(f"Could not process image {img_path}. Error: {e}")

if not images_to_test:
    print("No images were found to test. Exiting.")
    exit()

# --- Calculate Metrics ---
# Get confidence stats
highest_conf = max(confs) if confs else 0
lowest_conf = min(confs) if confs else 0
avg_conf = np.mean(confs) if confs else 0

# Generate confusion matrix
# labels=[0, 1] ensures the matrix is 2x2.
# 'Bad' (0) is the negative class, 'Good' (1) is the positive class.
cm = confusion_matrix(true_labels, preds, labels=[0, 1])

# Calculate overall metrics using a weighted average to account for class imbalance
accuracy = (cm[1][1] + cm[0][0]) / np.sum(cm) if np.sum(cm) > 0 else 0
precision = precision_score(true_labels, preds, average='weighted', zero_division=0)
recall = recall_score(true_labels, preds, average='weighted', zero_division=0)
f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)


# --- Save and Print Results ---
results_summary = (
    f"--- Model Test Results ---\n"
    f"Total images tested: {len(true_labels)}\n\n"
    f"Highest confidence: {highest_conf * 100:.2f}%\n"
    f"Lowest confidence: {lowest_conf * 100:.2f}%\n"
    f"Average confidence: {avg_conf * 100:.2f}%\n\n"
    f"--- Overall Model Metrics ---\n"
    f"Accuracy:  {accuracy:.4f}\n"
    f"Precision: {precision:.4f} (Weighted Avg)\n"
    f"Recall:    {recall:.4f} (Weighted Avg)\n"
    f"F1-Score:  {f1:.4f} (Weighted Avg)\n\n"
    f"--- Confusion Matrix ---\n"
    f"{'':<12}{'Predicted Good':<20}{'Predicted Bad':<20}\n"
    f"{'Actual Good':<12}{cm[1][1]:<20}{cm[1][0]:<20}\n"
    f"{'Actual Bad':<12}{cm[0][1]:<20}{cm[0][0]:<20}\n"
)


# Save results to a file
with open(RESULTS_FILE, 'w') as f:
    f.write(results_summary)

# Print to console
print("\n" + results_summary)
print(f"Results have been saved to {RESULTS_FILE}")

