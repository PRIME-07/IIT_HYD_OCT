import os
import shutil

# Define paths
heidelberg_path = 'D:/Anuj/Woxsen/IITH/aireadi-data/heidelberg_spectralis'
topcon_path = 'D:/Anuj/Woxsen/IITH/aireadi-data/topcon_maestro2'
output_path = 'paired_images'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Get patient IDs from both folders
heidelberg_patients = set(os.listdir(heidelberg_path))
topcon_patients = set(os.listdir(topcon_path))

# Get common patient IDs
common_patients = heidelberg_patients & topcon_patients

print(f"Found {len(common_patients)} common patients.")

# Helper to check if a filename is left/right and matches expected pattern
def is_valid_file(filename, modality_keywords):
    return any(k in filename for k in modality_keywords)

# Keywords to identify target files
spectralis_keywords = ['spectralis_ppol_mac_hr_oct_l', 'spectralis_ppol_mac_hr_oct_r']
maestro_keywords = ['maestro2_macula_6x6_oct_l', 'maestro2_macula_6x6_oct_r']

# Loop over each common patient
for pid in common_patients:
    pid_folder = os.path.join(output_path, pid)
    os.makedirs(pid_folder, exist_ok=True)

    # Get all files in Heidelberg folder
    heidelberg_files = []
    for root, _, files in os.walk(os.path.join(heidelberg_path, pid)):
        for file in files:
            if is_valid_file(file, spectralis_keywords):
                full_path = os.path.join(root, file)
                heidelberg_files.append(full_path)

    # Get all files in Topcon folder
    topcon_files = []
    for root, _, files in os.walk(os.path.join(topcon_path, pid)):
        for file in files:
            if is_valid_file(file, maestro_keywords):
                full_path = os.path.join(root, file)
                topcon_files.append(full_path)

    # Copy selected files
    for file_path in heidelberg_files + topcon_files:
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(pid_folder, filename))

print("✅ Paired image extraction complete.")
