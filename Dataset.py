import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2
import numpy as np

class OCTDataset(Dataset):
    def __init__(self, root_dir, transform=None, clahe=False):
        # Use the below commented lines in case you want to use the data as Domain A and Domain B folders instead of going through the patient folders for the two domains.
        # In case of using the below lines replace the root_dir in the _init_ function with root_dir_A and root_dir_B
        # self.domainA_paths = [os.path.join(root_dir_A, f) for f in os.listdir(root_dir_A) if f.endswith(".png") or f.endswith(".jpg")]
        # self.domainB_paths = [os.path.join(root_dir_B, f) for f in os.listdir(root_dir_B) if f.endswith(".png") or f.endswith(".jpg")]
        # Below lines are used if you want to go through patient folders in which you have the two domains.
        self.domainA_paths = []
        self.domainB_paths = []

        # The transform function is common for any method
        transform = T.Compose([
                        T.Resize((512,512)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.5],[0.5])   # single channel normalized to [-1, 1]
                    ])
        self.transform = transform
        self.clahe = clahe

        # Loop for all patient folders, comment the below line if you want to use the two domain folders instead.
        for patient_folder in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_folder)
            if not os.path.isdir(patient_path):
                continue
            
            domainA_folder = os.path.join(patient_path, "spectralis")
            domainB_folder = os.path.join(patient_path, "topcon")

            if os.path.exists(domainA_folder):
                self.domainA_paths += [
                    os.path.join(domainA_folder, f) 
                    for f in os.listdir(domainA_folder) 
                    if f.endswith(".png") or f.endswith(".jpg")
                ]
            if os.path.exists(domainB_folder):
                self.domainB_paths += [
                    os.path.join(domainB_folder, f)
                    for f in os.listdir(domainB_folder)
                    if f.endswith(".png") or f.endswith(".jpg")
                ]

    def __len__(self):
        return max(len(self.domainA_paths), len(self.domainB_paths))

    def __getitem__(self, idx):
        A_idx = idx % len(self.domainA_paths)
        B_idx = np.random.randint(0, len(self.domainB_paths))

        A_path = self.domainA_paths[A_idx]
        B_path = self.domainB_paths[B_idx]

        # Read images in grayscale
        A_img = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE)
        B_img = cv2.imread(B_path, cv2.IMREAD_GRAYSCALE)

        if self.clahe:
            clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            A_img = clahe_op.apply(A_img)
            B_img = clahe_op.apply(B_img)

        A_img = Image.fromarray(A_img)
        B_img = Image.fromarray(B_img)

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return {"A": A_img, "B": B_img}
