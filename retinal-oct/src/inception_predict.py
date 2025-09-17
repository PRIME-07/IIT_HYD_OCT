import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import models
from PIL import Image
import os

# Load Inception v3 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.inception_v3(weights=None, aux_logits=True, transform_input=True)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, 4)
model.load_state_dict(torch.load("best_inception_model.pt", map_location=device))
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
        output = self.model(input_tensor)  # Only main output in eval mode
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


# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor

# Load Image
test_image_path = "data/test/NORMAL/NORMAL-60471-1.jpeg"  # Replace with your test image
orig_img, input_tensor = preprocess_image(test_image_path)

# Grad-CAM Execution
target_layer = model.Mixed_7c  # Last inception block
cam_extractor = GradCAM(model, target_layer)
cam = cam_extractor.generate(input_tensor)

# Overlay Heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
orig_img_np = np.array(orig_img.resize((299, 299))) / 255.0
overlayed = 0.5 * heatmap + 0.5 * orig_img_np
overlayed = np.clip(overlayed, 0, 1)

# Plot Results
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(orig_img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cam, cmap='jet')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlayed)
plt.axis("off")

plt.tight_layout()
plt.show()
