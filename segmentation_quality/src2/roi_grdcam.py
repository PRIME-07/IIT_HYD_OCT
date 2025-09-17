import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F

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

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODIFIED: Load the new ROI-trained model ---
model = models.resnet50(weights=None) # No pretrained weights needed for loading state_dict
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_roi_quality.pt", map_location=device))
model.eval().to(device)

# --- Hooks for Grad-CAM (No changes needed here) ---
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

# --- MODIFIED: Main function to generate and display CAM ---
def get_cam(image_path):
    gradients.clear()
    activations.clear()

    # 1. Load the original image
    original_img_pil = Image.open(image_path).convert("RGB")

    # 2. Define and apply the EXACT same transforms as in training
    # We apply them step-by-step to keep the intermediate (cropped) image for visualization
    roi_cropper = ROICrop(crop_height=128)
    resizer = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()

    cropped_img_pil = roi_cropper(original_img_pil)
    processed_img_pil = resizer(cropped_img_pil) # This is the 224x224 image the model sees

    # 3. Create the input tensor for the model
    input_tensor = to_tensor(processed_img_pil).unsqueeze(0).to(device)

    # 4. Forward and backward pass
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = output.argmax().item()
    confidence = probs[0, pred_class].item()

    print(f"\nImage: {image_path.split('/')[-1]}")
    print(f"Prediction: {class_names[pred_class]} (Confidence: {confidence * 100:.2f}%)")

    class_score = output[0, pred_class]
    model.zero_grad()
    class_score.backward()

    # 5. Compute Grad-CAM heatmap
    grad = gradients[0]
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
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img_pil)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM on Cropped Input")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Test on an image ---
# Use one of your images here
get_cam("src/test_imgs/g/b_OS_FU_A_001010.jpg")