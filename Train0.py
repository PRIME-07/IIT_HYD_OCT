import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from Dataset import OCTDataset
from Model import ResnetGenerator, NLayerDiscriminator
from torchvision.utils import save_image
import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Initialize models
G_AB = ResnetGenerator().to(device)
G_BA = ResnetGenerator().to(device)
D_A = NLayerDiscriminator().to(device)
D_B = NLayerDiscriminator().to(device)

# Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizers
lr = 0.0002
beta1 = 0.5
optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

# Dataset
transform = ...

dataset = OCTDataset("roi", transform=transform, clahe=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

num_epochs = 200
log_file = open("checkpoints/training_log.txt", "a")  # append mode, or "w" to overwrite

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        real_A = data["A"].to(device)
        real_B = data["B"].to(device)

        # Generator forward
        fake_B = G_AB(real_A)
        rec_A = G_BA(fake_B)
        fake_A = G_BA(real_B)
        rec_B = G_AB(fake_A)

        # identity loss
        idt_A = G_BA(real_A)
        loss_idt_A = criterion_identity(idt_A, real_A) * 5.0
        idt_B = G_AB(real_B)
        loss_idt_B = criterion_identity(idt_B, real_B) * 5.0

        # GAN loss
        pred_fake_B = D_B(fake_B)
        loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B).to(device))
        pred_fake_A = D_A(fake_A)
        loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A).to(device))

        # cycle loss
        loss_cycle_A = criterion_cycle(rec_A, real_A) * 10.0
        loss_cycle_B = criterion_cycle(rec_B, real_B) * 10.0

        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # update D_A
        pred_real_A = D_A(real_A)
        loss_D_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A).to(device))
        pred_fake_A = D_A(fake_A.detach())
        loss_D_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A).to(device))
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()

        # update D_B
        pred_real_B = D_B(real_B)
        loss_D_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B).to(device))
        pred_fake_B = D_B(fake_B.detach())
        loss_D_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B).to(device))
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()

        # Printing the training log for every 100 iterations
        if i % 100 == 0:
            log_str = f"Epoch {epoch} | Iter {i} | G loss: {loss_G.item():.4f} | D_A: {loss_D_A.item():.4f} | D_B: {loss_D_B.item():.4f}"
            print(log_str)
            log_file.write(log_str + "\n")
            log_file.flush()  # force write to disk

        # Saving the model checkpoints after every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(G_AB.state_dict(), f"checkpoints/G_AB_epoch_{epoch+1}.pth")
            torch.save(G_BA.state_dict(), f"checkpoints/G_BA_epoch_{epoch+1}.pth")
            torch.save(D_A.state_dict(), f"checkpoints/D_A_epoch_{epoch+1}.pth")
            torch.save(D_B.state_dict(), f"checkpoints/D_B_epoch_{epoch+1}.pth")

    # Save images every epoch
    save_image(fake_B * 0.5 + 0.5, f"checkpoints/fakeB_epoch_{epoch}.png")
    save_image(fake_A * 0.5 + 0.5, f"checkpoints/fakeA_epoch_{epoch}.png")
