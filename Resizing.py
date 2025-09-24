import os
import torch
from torchvision import transforms
from PIL import Image
from Model import ResnetGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load generator
G_BA = ResnetGenerator()
G_BA = torch.nn.DataParallel(G_BA)
G_BA.load_state_dict(torch.load("checkpoints/G_BA_epoch_21.pth"))
G_BA.to(device)
G_BA.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

inv_normalize = transforms.Normalize(
    mean=[-1],
    std=[2]
)

to_pil = transforms.ToPILImage()

# Generate and save image with resizing
def generate_and_resize(input_path, output_path_dir):
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    # Load original image
    original_img = Image.open(input_path).convert("L")
    original_size = original_img.size  # (W, H)

    # Preprocess
    img = transform(original_img).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        fake = G_BA(img)

    # Postprocess generated image
    fake = fake.squeeze(0).cpu()
    fake = inv_normalize(fake).clamp(0, 1)

    # Save generated image before resizing
    fake_pre_resized = to_pil(fake)
    pre_resize_path = os.path.join(output_path_dir, f"{name}_gen_before_resize{ext}")
    fake_pre_resized.save(pre_resize_path)

    # Resize to original input image size
    fake_resized = fake_pre_resized.resize(original_size, Image.BICUBIC)
    resized_path = os.path.join(output_path_dir, f"{name}_gen_resized{ext}")
    fake_resized.save(resized_path)

    # Concatenate input and generated resized output side-by-side
    combined = Image.new('L', (original_img.width + fake_resized.width, original_img.height))
    combined.paste(original_img, (0, 0))
    combined.paste(fake_resized, (original_img.width, 0))
    concat_path = os.path.join(output_path_dir, f"{name}_concat{ext}")
    combined.save(concat_path)

    print(f"Processed and saved: {name}")

# Batch process folder
input_folder = "roi/1014/topcon/"
output_folder = "output_images_roi/101421"
os.makedirs(output_folder, exist_ok=True)

for img_file in os.listdir(input_folder):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_folder, img_file)
        generate_and_resize(input_path, output_folder)
