import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import itertools

from Model import ResnetGenerator, NLayerDiscriminator
from Dataset import OCTDataset

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize models
G_AB = ResnetGenerator().to(device)
G_BA = ResnetGenerator().to(device)
D_A = NLayerDiscriminator().to(device)
D_B = NLayerDiscriminator().to(device)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    G_AB = nn.DataParallel(G_AB)
    G_BA = nn.DataParallel(G_BA)
    D_A = nn.DataParallel(D_A)
    D_B = nn.DataParallel(D_B)

# Loss functions
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizers
lr = 0.0002
beta1 = 0.5
optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

# Load previous checkpoints (optional — if restarting training)
def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"Checkpoint not found: {path}")

load_checkpoint(G_AB, optimizer_G, "checkpoints/G_AB_epoch_9.pth")
load_checkpoint(G_BA, optimizer_G, "checkpoints/G_BA_epoch_9.pth")
load_checkpoint(D_A, optimizer_D_A, "checkpoints/D_A_epoch_9.pth")
load_checkpoint(D_B, optimizer_D_B, "checkpoints/D_B_epoch_9.pth")

# Dataset
data_root = "roi"  # path to your patient data folders
dataset = OCTDataset(data_root, clahe=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# Logging
num_epochs = 10
log_file = open("checkpoints/retrain_log.txt", "a")

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        real_A = data["A"].to(device)
        real_B = data["B"].to(device)

        # Generator forward
        fake_B = G_AB(real_A)
        rec_A = G_BA(fake_B)
        fake_A = G_BA(real_B)
        rec_B = G_AB(fake_A)

        # Identity loss
        idt_A = G_BA(real_A)
        loss_idt_A = criterion_identity(idt_A, real_A) * 5.0
        idt_B = G_AB(real_B)
        loss_idt_B = criterion_identity(idt_B, real_B) * 5.0

        # GAN loss
        pred_fake_B = D_B(fake_B)
        loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
        pred_fake_A = D_A(fake_A)
        loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        # Cycle consistency loss
        loss_cycle_A = criterion_cycle(rec_A, real_A) * 10.0
        loss_cycle_B = criterion_cycle(rec_B, real_B) * 10.0

        # Total Generator loss
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Update D_A
        pred_real_A = D_A(real_A)
        loss_D_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
        pred_fake_A = D_A(fake_A.detach())
        loss_D_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A_total = (loss_D_real_A + loss_D_fake_A) * 0.5
        optimizer_D_A.zero_grad()
        loss_D_A_total.backward()
        optimizer_D_A.step()

        # Update D_B
        pred_real_B = D_B(real_B)
        loss_D_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
        pred_fake_B = D_B(fake_B.detach())
        loss_D_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_B_total = (loss_D_real_B + loss_D_fake_B) * 0.5
        optimizer_D_B.zero_grad()
        loss_D_B_total.backward()
        optimizer_D_B.step()

        # Logging every 100 steps
        if i % 100 == 0:
            log_str = f"Epoch {epoch} | Iter {i} | G: {loss_G.item():.4f} | D_A: {loss_D_A_total.item():.4f} | D_B: {loss_D_B_total.item():.4f}"
            print(log_str)
            log_file.write(log_str + "\n")
            log_file.flush()

    # Save images
    save_image(fake_B * 0.5 + 0.5, f"checkpoints/fakeB_epoch_{epoch}.png")
    save_image(fake_A * 0.5 + 0.5, f"checkpoints/fakeA_epoch_{epoch}.png")

    # Save model
    if (epoch + 1) % 3 == 0:
        torch.save(G_AB.state_dict(), f"checkpoints/G_AB_epoch_{epoch+1}.pth")
        torch.save(G_BA.state_dict(), f"checkpoints/G_BA_epoch_{epoch+1}.pth")
        torch.save(D_A.state_dict(), f"checkpoints/D_A_epoch_{epoch+1}.pth")
        torch.save(D_B.state_dict(), f"checkpoints/D_B_epoch_{epoch+1}.pth")

log_file.close()
