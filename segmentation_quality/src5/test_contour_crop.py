import torch
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
import cv2

#  Contour Detection 
class ContourCrop:
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
        kernel = np.ones((5, 11), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
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

#  Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Model Loading 
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_contour_crop_quality.pt", map_location=device)) # MODIFIED model path
model.eval().to(device)

#  Preprocessing 
transform = transforms.Compose([
    ContourCrop(padding=20),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Testing Logic
class_names = ['Bad', 'Good']

# Assuming the script is in your project's root directory
# Adjust the path if your 'dataset' folder is elsewhere
poor_dir = os.path.join(os.getcwd(), './src/dataset', 'poor_seg_data')
results_file = os.path.join(os.getcwd(), 'contour_crop_test_results.txt')

if not os.path.exists(poor_dir):
    print(f"Error: Test directory not found at {poor_dir}")
else:
    image_files = [f for f in os.listdir(poor_dir) if f.lower().endswith('.png')]
    preds = []
    confs = []

    print(f"Starting evaluation on {len(image_files)} images...")

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(poor_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax().item()
            confidence = probs[0, pred_class].item()
            
            preds.append(pred_class)
            confs.append(confidence)

    # Assuming all images in this folder are 'poor' quality (class 0)
    true_labels = [0] * len(preds)

# Calculate metrics
    if len(preds) > 0:
        accuracy = sum([p == t for p, t in zip(preds, true_labels)]) / len(preds)
        highest_conf = max(confs)
        lowest_conf = min(confs)
        cm = confusion_matrix(true_labels, preds, labels=[0, 1])

        # Save and print results
        with open(results_file, 'w') as f:
            f.write(f"Total images tested: {len(preds)}\n")
            f.write(f"Correct predictions (Predicted 'Bad'): {cm[0][0]}\n")
            f.write(f"Incorrect predictions (Predicted 'Good'): {cm[0][1]}\n")
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"Highest confidence on a prediction: {highest_conf*100:.2f}%\n")
            f.write(f"Lowest confidence on a prediction: {lowest_conf*100:.2f}%\n\n")
            f.write("Confusion Matrix:\n")
            f.write("           Predicted:\n")
            f.write("           Bad    Good\n")
            f.write(f"Actual Bad:  {cm[0][0]:<5}  {cm[0][1]:<5}\n")
            f.write(f"Actual Good: {cm[1][0]:<5}  {cm[1][1]:<5}\n")

        print(f"Evaluation complete. Results saved to {results_file}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print("\nConfusion Matrix:")
        print("           Predicted:")
        print("           Bad    Good")
        print(f"Actual Bad:  {cm[0][0]:<5}  {cm[0][1]:<5}")
        print(f"Actual Good: {cm[1][0]:<5}  {cm[1][1]:<5}")
    else:
        print("No images found in the directory")