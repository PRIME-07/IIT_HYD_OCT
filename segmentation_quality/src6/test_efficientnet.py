import torch
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
import cv2
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class PathCrop:
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
            if content_top >= content_bottom: return img
            final_cropped_img = img.crop((0, content_top, w, content_bottom + 1))
            return final_cropped_img
        else: return img

#  Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Model Loading 
model = efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("efficientnet_b0_quality.pt", map_location=device))
model.eval().to(device)

#  Preprocessing 
transform = transforms.Compose([
    PathCrop(padding_bottom=20),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#  Testing Logic 
class_names = ['Bad', 'Good']
poor_dir = os.path.join(os.getcwd(), './src/dataset', 'poor_seg_data')
results_file = os.path.join(os.getcwd(), 'efficientnet_test_results.txt')

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

    if len(preds) > 0:
        accuracy = sum([p == t for p, t in zip(preds, true_labels)]) / len(preds)
        highest_conf = max(confs)
        lowest_conf = min(confs)
        cm = confusion_matrix(true_labels, preds, labels=[0, 1])

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
        print("No images found in the directory to test.")