import pydicom
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_multiframe_dicom(path, resize_shape=(512, 512)):
    ds = pydicom.dcmread(path)
    volume = ds.pixel_array  # shape: (num_slices, height, width)
    
    # Normalize pixel values
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-5)

    # Resize each slice
    resized_volume = np.stack([
        cv2.resize(slice, resize_shape) for slice in volume
    ])
    return resized_volume

def detect_fovea_index(volume):
    valley_scores = []
    for i in range(volume.shape[0]):
        vertical_profile = np.sum(volume[i], axis=1)
        valley_scores.append(np.min(vertical_profile))  # valley = less retinal tissue
    return int(np.argmin(valley_scores))

def find_best_matching_slice(ref_slice, target_volume):
    ref_norm = (ref_slice - ref_slice.min()) / (ref_slice.max() - ref_slice.min() + 1e-5)
    best_score = -1
    best_index = -1
    for i in range(target_volume.shape[0]):
        tgt = target_volume[i]
        tgt_norm = (tgt - tgt.min()) / (tgt.max() - tgt.min() + 1e-5)
        score = ssim(ref_norm, tgt_norm, data_range=1.0)
        if score > best_score:
            best_score = score
            best_index = i
    return best_index, best_score

def crop_volume_around_index(volume, center_index, num_slices=20):
    start = max(center_index - num_slices // 2, 0)
    end = min(start + num_slices, volume.shape[0])
    return volume[start:end]

def align_oct_volumes(vol1, vol2, crop_slices=20):
    fovea_1 = detect_fovea_index(vol1)
    print(f"[INFO] Fovea in Volume 1 at slice: {fovea_1}")

    best_match_index, score = find_best_matching_slice(vol1[fovea_1], vol2)
    print(f"[INFO] Best matching slice in Volume 2: {best_match_index} (SSIM: {score:.4f})")

    vol1_cropped = crop_volume_around_index(vol1, fovea_1, crop_slices)
    vol2_cropped = crop_volume_around_index(vol2, best_match_index, crop_slices)

    return vol1_cropped, vol2_cropped

def visualize_alignment(vol1, vol2):
    num_slices = min(vol1.shape[0], vol2.shape[0])  # Use the smaller volume
    fig, axes = plt.subplots(2, num_slices, figsize=(num_slices * 2, 5))

    for i in range(num_slices):
        axes[0, i].imshow(vol1[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(vol2[i], cmap='gray')    
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

def save_aligned_slices(vol1, vol2, output_dir="aligned_data"):
    ziess_dir = os.path.join(output_dir, "ziess")
    heidelberg_dir = os.path.join(output_dir, "heidelberg")

    os.makedirs(ziess_dir, exist_ok=True)
    os.makedirs(heidelberg_dir, exist_ok=True)

    # Get the next starting index
    existing_files = os.listdir(ziess_dir)
    existing_indices = [
        int(f.split('.')[0]) for f in existing_files if f.endswith('.png') and f.split('.')[0].isdigit()
    ]
    start_index = max(existing_indices, default=-1) + 1

    for i in range(min(vol1.shape[0], vol2.shape[0])):
        idx = start_index + i
        filename = f"{idx:03d}.png"
        ziess_path = os.path.join(ziess_dir, filename)
        heidelberg_path = os.path.join(heidelberg_dir, filename)

        cv2.imwrite(ziess_path, (vol1[i] * 255).astype(np.uint8))
        cv2.imwrite(heidelberg_path, (vol2[i] * 255).astype(np.uint8))

    print(f"[INFO] Saved {min(vol1.shape[0], vol2.shape[0])} slices starting from index {start_index}")


if __name__ == "__main__":
    maestro_dcm = "D:/Anuj/Woxsen/IITH/oct-upscaling/paired_images/1011/1011_maestro2_macula_6x6_oct_l_2.16.840.1.114517.10.5.1.4.907063120230816140816.1.1.dcm"
    spectralis_dcm = "D:/Anuj/Woxsen/IITH/oct-upscaling/paired_images/1011/1011_spectralis_ppol_mac_hr_oct_l_1.3.6.1.4.1.33437.11.4.7587979.50556511832883.23310.4.1.dcm"

    print("[INFO] Loading multi-frame DICOMs...")
    vol1 = load_multiframe_dicom(maestro_dcm, resize_shape=(512, 512))
    vol2 = load_multiframe_dicom(spectralis_dcm, resize_shape=(512, 512))

    print(f"[INFO] Volume 1 shape: {vol1.shape}")
    print(f"[INFO] Volume 2 shape: {vol2.shape}")

    aligned1, aligned2 = align_oct_volumes(vol1, vol2, crop_slices=20)

    #aligned1 = "" 

    visualize_alignment(aligned1, aligned2)
    save_aligned_slices(aligned1, aligned2, output_dir="aligned_data")
