import os
import pydicom
import numpy as np
from PIL import Image

def normalize_and_save(img_slice, output_path, slice_idx):
    img_norm = img_slice.astype(float)
    img_norm -= np.min(img_norm)
    if np.max(img_norm) > 0:
        img_norm /= np.max(img_norm)
    img_norm *= 255
    img_norm = img_norm.astype(np.uint8)

    image = Image.fromarray(img_norm)
    filename = os.path.join(output_path, f"slice_{slice_idx:04d}.jpg")
    image.save(filename)
    print(f"Saved: {filename}")

# Base paths
paired_images_path = "../paired_images"
roi_path = "../roi"

# Loop through each patient
for patient_id in os.listdir(paired_images_path):
    patient_folder = os.path.join(paired_images_path, patient_id)
    if not os.path.isdir(patient_folder):
        continue

    for filename in os.listdir(patient_folder):
        file_path = os.path.join(patient_folder, filename)
        if not filename.endswith('.dcm'):
            continue

        print(f"\nProcessing {file_path}")
        dcm = pydicom.dcmread(file_path)
        volume = dcm.pixel_array
        total_slices = volume.shape[0]
        mid_slice = total_slices // 2

        # Maestro2 (topcon)
        if "maestro2" in filename:
            output_folder = os.path.join(roi_path, patient_id, "topcon")
            os.makedirs(output_folder, exist_ok=True)

            # Get 5 frames above and below mid with step 3, plus mid
            offsets = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
            for offset in offsets:
                idx = mid_slice + offset
                if 0 <= idx < total_slices:
                    normalize_and_save(volume[idx], output_folder, idx)

        # Spectralis
        elif "spectralis" in filename:
            output_folder = os.path.join(roi_path, patient_id, "spectralis")
            os.makedirs(output_folder, exist_ok=True)

            # Get 5 consecutive above and below mid, plus mid
            start = max(0, mid_slice - 5)
            end = min(total_slices, mid_slice + 6)  # +6 to include mid + 5
            for idx in range(start, end):
                normalize_and_save(volume[idx], output_folder, idx)

print("\n✅ All ROI slices extracted.")
