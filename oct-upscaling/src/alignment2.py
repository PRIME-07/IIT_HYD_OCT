import pydicom
from PIL import Image
import numpy as np
import os

# Load the DICOM file
dcm_path = "D:/Anuj/Woxsen/IITH/oct-upscaling/paired_images/1004/1004_maestro2_macula_6x6_oct_l_2.16.840.1.114517.10.5.1.4.907063120230808105533.1.1.dcm"
dcm = pydicom.dcmread(dcm_path)

# Extract the pixel volume (3D: depth × height × width)
volume = dcm.pixel_array
print("DICOM volume shape:", volume.shape)

# Get the total number of slices
num_slices = volume.shape[0]

# Calculate the mid slice
mid_slice = num_slices // 2

# Define the slice range: 5 before and 5 after the mid slice
start_slice = max(0, mid_slice - 15)
end_slice = min(num_slices - 1, mid_slice + 15)

# Define the output folder
output_dir = "Topcon_1001_mid_11_slices"
os.makedirs(output_dir, exist_ok=True)

# Extract and save the slices
for i in range(start_slice, end_slice + 1):
    img_slice = volume[i, :, :]

    # Normalize the slice to 0–255
    img_norm = img_slice.astype(float)
    img_norm -= np.min(img_norm)
    if np.max(img_norm) > 0:
        img_norm /= np.max(img_norm)
    img_norm *= 255
    img_norm = img_norm.astype(np.uint8)

    # Convert to PIL image and save
    image = Image.fromarray(img_norm)
    filename = os.path.join(output_dir, f"slice_{i:04d}.jpg")
    image.save(filename)
    print(f"Saved {filename}")

print("✅ Extracted and saved 11 middle slices.")
