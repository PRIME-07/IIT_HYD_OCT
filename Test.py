from Model import ResnetGenerator
import torch
from torchvision import transforms
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------
# Load Generators
# --------------------------
G_AB = ResnetGenerator()
G_AB = torch.nn.DataParallel(G_AB)
G_AB.load_state_dict(torch.load("checkpoints/G_AB_epoch_21.pth"))
G_AB.to(device)
G_AB.eval()

G_BA = ResnetGenerator()
G_BA = torch.nn.DataParallel(G_BA)
G_BA.load_state_dict(torch.load("checkpoints/G_BA_epoch_21.pth"))
G_BA.to(device)
G_BA.eval()

# --------------------------
# Transform (fixed GAN input size)
# --------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --------------------------
# Domain-specific sizes
# --------------------------
SPECTRALIS_SIZE = (768, 496)  # (W, H)
TOPCON_SIZE = (360, 885)      # (W, H)

# --------------------------
# Auto-detect and generate
# --------------------------
def generate_auto(img_path, output_path):
    orig = Image.open(img_path).convert("L")
    w, h = orig.size  # (W, H)

    out = ""
    if (w, h) == SPECTRALIS_SIZE:
        print(f"Detected Spectralis ({w}x{h}) → Generating Topcon...")
        model = G_AB
        target_size = TOPCON_SIZE
        out = "topcon"
    elif (w, h) == TOPCON_SIZE:
        print(f"Detected Topcon ({w}x{h}) → Generating Spectralis...")
        model = G_BA
        target_size = SPECTRALIS_SIZE
        out = "spectralis"
    else:
        raise ValueError(f"Unknown input size {w}x{h}. Expected {SPECTRALIS_SIZE} or {TOPCON_SIZE}.")

    img = transform(orig).unsqueeze(0).to(device)
    with torch.no_grad():
        fake = model(img)

    fake = fake.squeeze().cpu() * 0.5 + 0.5
    fake_img = transforms.ToPILImage()(fake)

    # Resize to target domain size
    fake_img = fake_img.resize(target_size, Image.BICUBIC)
    output_path = f"{output_path}_{out}.jpg" # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fake_img.save(output_path)
    print(f"Saved: {output_path}")


# --------------------------
# Example usage
# --------------------------
# Spectralis input → Topcon output

input = "roi/1003/topcon/slice_0174.jpg"
generate_auto(input, f"output_images_roi/21{input[4:8]}/{input[16:-4]}")
