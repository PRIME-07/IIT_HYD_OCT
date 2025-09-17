import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import nn
import os
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

#  Hooks for Grad-CAM 
gradients = []
activations = []
def save_gradient(module, grad_in, grad_out):
    gradients.append(grad_out[0])
def save_activation(module, input, output):
    activations.append(output)
target_layer = model.features[-1] 
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

class_names = ['Bad', 'Good']

#  Main function to generate and display CAM 
def get_cam(image_path):
    gradients.clear()
    activations.clear()

    original_img_pil = Image.open(image_path).convert("RGB")

    path_cropper = PathCrop(padding_bottom=20)
    resizer = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cropped_img_pil = path_cropper(original_img_pil)
    processed_img_pil = resizer(cropped_img_pil)
    
    input_tensor = normalizer(to_tensor(processed_img_pil)).unsqueeze(0).to(device)

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = output.argmax().item()
    confidence = probs[0, pred_class].item()

    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: {class_names[pred_class]} (Confidence: {confidence * 100:.2f}%)")

    class_score = output[0, pred_class]
    model.zero_grad()
    class_score.backward()

    grad = gradients[0].detach()
    act = activations[0].detach()
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = cam.squeeze().cpu().numpy()
    
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = np.uint8(255 * cam)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    processed_img_np = np.array(processed_img_pil)
    img_bgr = cv2.cvtColor(processed_img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img_pil)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM on Path-Cropped Input")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#  Test on an image 
if __name__ == '__main__':
    # Replace this with a path to one of your images to test it
    test_image_path = "src/test_imgs/g/b_4_AAA_001015.jpg"
    if os.path.exists(test_image_path):
        get_cam(test_image_path)
    else:
        print(f"Test image not found at: {test_image_path}")