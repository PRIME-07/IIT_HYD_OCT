import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import nn
import os

#  REQUIRED: Add the exact same EdgeCrop class definition from your training script 
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

# Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loading 
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_edgecrop_quality.pt", map_location=device))
model.eval().to(device)

# Hooks for Grad-CAM 
gradients = []
activations = []

def save_gradient(module, grad_in, grad_out):
    gradients.append(grad_out[0])

def save_activation(module, input, output):
    activations.append(output)

target_layer = model.layer4[-1]
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

# Class labels 
class_names = ['Bad', 'Good']

# Main function to generate and display CAM 
def get_cam(image_path):
    gradients.clear()
    activations.clear()

    # 1. Load the original image
    original_img_pil = Image.open(image_path).convert("RGB")

    # 2. Define and apply transforms step-by-step for visualization
    edge_cropper = EdgeCrop(padding=20)
    resizer = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()

    # Get the intermediate and final images
    cropped_img_pil = edge_cropper(original_img_pil)
    processed_img_pil = resizer(cropped_img_pil) # This is the 224x224 image the model sees

    # 3. Create the input tensor
    input_tensor = to_tensor(processed_img_pil).unsqueeze(0).to(device)

    # 4. Forward and backward pass
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = output.argmax().item()
    confidence = probs[0, pred_class].item()

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: {class_names[pred_class]} (Confidence: {confidence * 100:.2f}%)")

    class_score = output[0, pred_class]
    model.zero_grad()
    class_score.backward()

    # 5. Compute Grad-CAM heatmap
    grad = gradients[0].detach()
    act = activations[0].detach()
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = cam.squeeze().cpu().numpy()
    
    # 6. Resize heatmap to match the processed image size (224x224)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = np.uint8(255 * cam)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 7. Overlay the heatmap on the PROCESSED image
    processed_img_np = np.array(processed_img_pil)
    img_bgr = cv2.cvtColor(processed_img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    # 8. Display side-by-side for comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img_pil)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM on Edge-Cropped Input")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Test on an image
# Replace this with the path to one of your test images
get_cam("src/test_imgs/b/019 - GA_OC-1901-25-019-UNSC-OCT_OD_Full_Org_1_000038.png")