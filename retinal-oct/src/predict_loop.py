import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import models
from PIL import Image
import os

# Load Model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("src/best_model.pt", map_location=device))
model.eval().to(device)

# Grad-CAM Setup 
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        predicted_class = output.argmax().item()
        
        class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
        print(f"\nPredicted: {class_names[predicted_class]} (Confidence: {probabilities[predicted_class]*100:.2f}%)")

        if target_class is None:
            target_class = predicted_class

        loss = output[:, target_class]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor

# Loop through uploads and process
upload_folder = "upload"
output_folder = "upload_results"
os.makedirs(output_folder, exist_ok=True)

target_layer = model.layer4[-1]
cam_extractor = GradCAM(model, target_layer)

for image_name in os.listdir(upload_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(upload_folder, image_name)
        print(f"\nProcessing: {image_name}")
        
        orig_img, input_tensor = preprocess_image(image_path)
        cam = cam_extractor.generate(input_tensor)

        # Overlay Heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        orig_img_np = np.array(orig_img.resize((224, 224))) / 255.0
        overlayed = 0.5 * heatmap + 0.5 * orig_img_np
        overlayed = np.clip(overlayed, 0, 1)

        # Convert back to PIL and save
        overlayed_img = Image.fromarray(np.uint8(overlayed * 255))
        save_path = os.path.join(output_folder, f"gradcam_{image_name}")
        overlayed_img.save(save_path)
        print(f"Saved Grad-CAM result to: {save_path}")
