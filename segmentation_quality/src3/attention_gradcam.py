import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import nn
from test_attention import ResNetWithCBAM

# --- NEW: Add the exact same CBAM and ResNetWithCBAM class definitions from your training script here ---
# ... (paste the ChannelAttention, SpatialAttention, CBAM, and ResNetWithCBAM classes here) ...

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODIFIED: Model Loading ---
base_model = models.resnet50(weights=None)
model = ResNetWithCBAM(base_model)
model.load_state_dict(torch.load("resnet50_attention_quality.pt", map_location=device))
model.eval().to(device)

# --- Hooks for Grad-CAM ---
gradients = []
activations = []

def save_gradient(module, grad_in, grad_out):
    gradients.append(grad_out[0])

def save_activation(module, input, output):
    activations.append(output)

# --- MODIFIED: Target the final attention block ---
target_layer = model.cbam4
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

# --- MODIFIED: Image preprocessing (back to original) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Class labels
class_names = ['Bad', 'Good']

def get_cam(image_path):
    # This function can now revert to your original Grad-CAM logic,
    # as there is no complex cropping to handle for visualization.
    gradients.clear()
    activations.clear()

    img = Image.open(image_path).convert("RGB")
    # The transform now does all the work in one step
    input_tensor = transform(img).unsqueeze(0).to(device)

    # --- The rest of your Grad-CAM logic remains the same as your first version ---
    # Forward pass
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = output.argmax().item()
    confidence = probs[0, pred_class].item()

    # Terminal output
    print(f"\nImage: {image_path}")
    print(f"Prediction: {class_names[pred_class]}\n(Confidence: {confidence * 100:.2f}%)\n")

    # Backward for Grad-CAM
    class_score = output[0, pred_class]
    model.zero_grad()
    class_score.backward()

    # Grad-CAM computation
    grad = gradients[0]             # [B, C, H, W]
    act = activations[0].detach()   # [B, C, H, W]

    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    cam = cv2.resize(cam, (img.width, img.height))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize
    heatmap = np.uint8(255 * cam)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    # Display side-by-side
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Test on custom images
get_cam("src/test_imgs/g/b_OS_FU_A_001044.jpg")