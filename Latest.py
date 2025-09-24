# train_with_improvements.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import itertools
from collections import OrderedDict

from Model import ResnetGenerator, NLayerDiscriminator
from Dataset import OCTDataset

# -------------------- Config --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths (same as your code)
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Resume settings
resume_epoch = 21   # change if resuming from a different epoch; 0 means train from scratch
start_epoch = resume_epoch + 1

# Training hyperparams
batch_size = 4
num_epochs = 100            # total extra epochs to run (this script will run epochs [start_epoch .. start_epoch+num_epochs-1])
lr = 0.0002
beta1 = 0.5
lambda_cycle = 10.0
lambda_id = 5.0

# Other
save_every = 3              # save checkpoints every `save_every` epochs
log_path = os.path.join(CHECKPOINT_DIR, "retrain_log.txt")

# -------------------- Utils --------------------
def save_full_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optG, optDA, optDB, path):
    state = {
        "epoch": epoch,
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "optG": optG.state_dict(),
        "optDA": optDA.state_dict(),
        "optDB": optDB.state_dict(),
    }
    torch.save(state, path)

def try_load_full_checkpoint(path, G_AB, G_BA, D_A, D_B, optG, optDA, optDB, map_location):
    if not os.path.exists(path):
        return False
    ck = torch.load(path, map_location=map_location)
    # load state dicts if present
    if "G_AB" in ck:
        G_AB.load_state_dict(ck["G_AB"])
    if "G_BA" in ck:
        G_BA.load_state_dict(ck["G_BA"])
    if "D_A" in ck:
        D_A.load_state_dict(ck["D_A"])
    if "D_B" in ck:
        D_B.load_state_dict(ck["D_B"])
    if "optG" in ck and optG is not None:
        try:
            optG.load_state_dict(ck["optG"])
        except Exception:
            print("Warning: optimizer G state could not be loaded cleanly.")
    if "optDA" in ck and optDA is not None:
        try:
            optDA.load_state_dict(ck["optDA"])
        except Exception:
            print("Warning: optimizer D_A state could not be loaded cleanly.")
    if "optDB" in ck and optDB is not None:
        try:
            optDB.load_state_dict(ck["optDB"])
        except Exception:
            print("Warning: optimizer D_B state could not be loaded cleanly.")
    print(f"Loaded full checkpoint from {path} (epoch {ck.get('epoch','?')}).")
    return True

def try_load_individual_state(model, path, map_location):
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location=map_location)
    # sometimes saved under {"model":..., "optimizer":...} or plain state_dict. handle both.
    if isinstance(state, dict) and any(k in state for k in ("G_AB","G_BA","D_A","D_B","model")):
        # this is a full dict saved differently — do nothing here (handled by full loader)
        return False
    try:
        model.load_state_dict(state)
        print(f"Loaded state dict from {path}")
        return True
    except RuntimeError:
        # maybe the saved dict has 'module.' keys from DataParallel
        new_state = OrderedDict()
        for k,v in state.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state[new_key] = v
        model.load_state_dict(new_state)
        print(f"Loaded (stripped) state dict from {path}")
        return True

# -------------------- Replay Buffer --------------------
class ReplayBuffer:
    """Buffer of previously generated samples for discriminator training (as in CycleGAN paper)."""
    def __init__(self, max_size=50):
        assert max_size > 0
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        # data: tensor batch (B, C, H, W)
        out = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                out.append(element)
            else:
                p = torch.rand(1).item()
                if p > 0.5:
                    # use a previously stored and replace it
                    idx = torch.randint(0, len(self.data), (1,)).item()
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    out.append(tmp)
                else:
                    out.append(element)
        return torch.cat(out, dim=0).to(data.device)

# -------------------- Models --------------------
G_AB = ResnetGenerator().to(device)
G_BA = ResnetGenerator().to(device)
D_A = NLayerDiscriminator().to(device)
D_B = NLayerDiscriminator().to(device)

# DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    G_AB = nn.DataParallel(G_AB)
    G_BA = nn.DataParallel(G_BA)
    D_A = nn.DataParallel(D_A)
    D_B = nn.DataParallel(D_B)

# -------------------- Losses & Optimizers --------------------
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

# -------------------- Try load checkpoints --------------------
# Prefer loading a "full" checkpoint (with optimizers) if exists
full_ckpt = os.path.join(CHECKPOINT_DIR, f"full_state_epoch_{resume_epoch}.pth")
loaded = try_load_full_checkpoint(full_ckpt, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, map_location=device)
if not loaded:
    # fallback to individual model .pth files (these might be plain state_dicts)
    try_load_individual_state(G_AB, os.path.join(CHECKPOINT_DIR, f"G_AB_epoch_{resume_epoch}.pth"), map_location=device)
    try_load_individual_state(G_BA, os.path.join(CHECKPOINT_DIR, f"G_BA_epoch_{resume_epoch}.pth"), map_location=device)
    try_load_individual_state(D_A, os.path.join(CHECKPOINT_DIR, f"D_A_epoch_{resume_epoch}.pth"), map_location=device)
    try_load_individual_state(D_B, os.path.join(CHECKPOINT_DIR, f"D_B_epoch_{resume_epoch}.pth"), map_location=device)

# -------------------- Dataloading --------------------
data_root = "roi"
dataset = OCTDataset(data_root, clahe=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# -------------------- Buffers, schedulers, scaler --------------------
fake_A_buffer = ReplayBuffer(max_size=50)
fake_B_buffer = ReplayBuffer(max_size=50)

# LR schedulers: linear decay after half of the (start_epoch+num_epochs) period
total_epochs = start_epoch + num_epochs - 1
decay_after = total_epochs // 2

def lambda_rule(epoch):
    if epoch < decay_after:
        return 1.0
    else:
        return 1.0 - float(epoch - decay_after) / max(1, (total_epochs - decay_after))

scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)

scaler = torch.cuda.amp.GradScaler()  # mixed precision

# -------------------- Training loop --------------------
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
log_file = open(log_path, "a")

for epoch in range(start_epoch, start_epoch + num_epochs):
    for i, data in enumerate(dataloader):
        real_A = data["A"].to(device)
        real_B = data["B"].to(device)

        # ====================
        #  Train Generators
        # ====================
        optimizer_G.zero_grad()
        with torch.cuda.amp.autocast():
            fake_B = G_AB(real_A)
            rec_A = G_BA(fake_B)

            fake_A = G_BA(real_B)
            rec_B = G_AB(fake_A)

            # Identity
            idt_A = G_BA(real_A)
            loss_idt_A = criterion_identity(idt_A, real_A) * lambda_id

            idt_B = G_AB(real_B)
            loss_idt_B = criterion_identity(idt_B, real_B) * lambda_id

            # GAN loss: try to fool discriminators
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle
            loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_cycle
            loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_cycle

            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # ====================
        #  Train Discriminator A
        # ====================
        optimizer_D_A.zero_grad()
        with torch.cuda.amp.autocast():
            # Real
            pred_real_A = D_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))

            # Fake (from buffer)
            fake_A_for_D = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake_A = D_A(fake_A_for_D)
            loss_D_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))

            loss_D_A_total = (loss_D_real_A + loss_D_fake_A) * 0.5

        scaler.scale(loss_D_A_total).backward()
        scaler.step(optimizer_D_A)
        scaler.update()

        # ====================
        #  Train Discriminator B
        # ====================
        optimizer_D_B.zero_grad()
        with torch.cuda.amp.autocast():
            # Real
            pred_real_B = D_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

            # Fake (from buffer)
            fake_B_for_D = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake_B = D_B(fake_B_for_D)
            loss_D_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

            loss_D_B_total = (loss_D_real_B + loss_D_fake_B) * 0.5

        scaler.scale(loss_D_B_total).backward()
        scaler.step(optimizer_D_B)
        scaler.update()

        # --------------------
        # Logging
        # --------------------
        if i % 100 == 0:
            log_str = (f"Epoch {epoch} | Iter {i}/{len(dataloader)} | "
                       f"G: {loss_G.item():.4f} | D_A: {loss_D_A_total.item():.4f} | D_B: {loss_D_B_total.item():.4f} | "
                       f"lr: {optimizer_G.param_groups[0]['lr']:.1e}")
            print(log_str)
            log_file.write(log_str + "\n")
            log_file.flush()

    # Save images (visual check)
    try:
        save_image(fake_B * 0.5 + 0.5, os.path.join(CHECKPOINT_DIR, f"fakeB_epoch_{epoch}.png"))
        save_image(fake_A * 0.5 + 0.5, os.path.join(CHECKPOINT_DIR, f"fakeA_epoch_{epoch}.png"))
    except Exception as e:
        print("Warning: could not save sample images:", e)

    # Save full checkpoint every 'save_every' epochs
    if (epoch + 1) % save_every == 0 or epoch == start_epoch + num_epochs - 1:
        ck_path = os.path.join(CHECKPOINT_DIR, f"full_state_epoch_{epoch}.pth")
        # unwrap DataParallel before saving if needed to keep portable keys
        def maybe_module_state(m):
            if isinstance(m, (nn.DataParallel,)):
                return m.module.state_dict()
            else:
                return m.state_dict()
        state = {
            "epoch": epoch,
            "G_AB": maybe_module_state(G_AB),
            "G_BA": maybe_module_state(G_BA),
            "D_A": maybe_module_state(D_A),
            "D_B": maybe_module_state(D_B),
            "optG": optimizer_G.state_dict(),
            "optDA": optimizer_D_A.state_dict(),
            "optDB": optimizer_D_B.state_dict()
        }
        torch.save(state, ck_path)
        print(f"Saved full checkpoint: {ck_path}")

    # Step LR schedulers
    scheduler_G.step()
    scheduler_D_A.step()
    scheduler_D_B.step()

log_file.close()
print("Training finished.")
