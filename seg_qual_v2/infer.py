import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ======================================================================================
# NOTE: The model definition must be available in the inference script
# so PyTorch knows how to load the saved weights.
# ======================================================================================
class AttentionGuidedModel(nn.Module):
    """
    A wrapper around a pretrained ResNet50 model to facilitate Grad-CAM.
    """
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=None) # Weights will be loaded
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

        self.feature_maps = None
        self.gradients = None

        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.feature_maps = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def forward(self, x):
        return self.model(x)

    def get_attention_map(self):
        """Generates the Grad-CAM attention map for a single image."""
        if self.feature_maps is None or self.gradients is None:
            return None
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Use a clone to avoid in-place modification
        feats = self.feature_maps.clone()
        for i in range(feats.shape[1]):
            feats[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(feats, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if heatmap.max() > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap

# ======================================================================================
# INFERENCE AND VISUALIZATION SCRIPT
# ======================================================================================
def predict_and_visualize(model_path, image_path):
    """
    Loads a model and an image, predicts quality, and visualizes the attention map.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- File Path Validation ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    # --- Load Model ---
    model = AttentionGuidedModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- Image Loading and Preprocessing ---
    img_pil = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device) # type: ignore

    # --- Prediction and Attention Map Generation ---
    output = model(img_tensor)
    
    # Backward pass on the output to populate gradients for Grad-CAM
    output.backward()
    
    attention_map = model.get_attention_map().detach().cpu().numpy() # type: ignore
    
    # --- Post-processing for Visualization ---
    pred_prob = torch.sigmoid(output).item()
    prediction = "Good" if pred_prob > 0.5 else "Bad"
    
    # Resize heatmap and overlay on the original image
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(attention_map, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # type: ignore
    
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img_cv, 0.6, 0)

    # --- Display Results ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Prediction: {prediction} ({pred_prob:.2f})")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # ======================================================================================
    # IMPORTANT: SET YOUR FILE PATHS HERE
    # ======================================================================================
    
    # --- Path to the image you want to test ---
    IMAGE_TO_TEST = "seg_qual_v2/data/good/images/a_25_18.jpg" 
    
    # --- Path to your trained model file ---
    MODEL_PATH = "seg_qual_v2/oct_quality_model_non_aug.pth"      
    
    # ======================================================================================
    
    predict_and_visualize(model_path=MODEL_PATH, image_path=IMAGE_TO_TEST)
