import sys
sys.path.insert(0, '/teamspace/studios/this_studio/sir_ml_project')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/teamspace/studios/this_studio/sir_ml_project'

# ── 1. Load dataset ──────────────────────────────────────────────
t_grid    = np.load(f'{BASE}/data/t_grid.npy')
all_beta  = np.load(f'{BASE}/data/all_beta.npy')
all_gamma = np.load(f'{BASE}/data/all_gamma.npy')
all_S     = np.load(f'{BASE}/data/all_S.npy')
all_I     = np.load(f'{BASE}/data/all_I.npy')
all_R     = np.load(f'{BASE}/data/all_R.npy')

N          = 1000.0
n_params   = len(all_beta)       # 400
n_points   = len(t_grid)         # 161
t_max      = t_grid[-1]          # 160.0

print(f"Dataset loaded: {n_params} param points x {n_points} time steps")

# ── 2. Build flat (beta, gamma, t) → (S, I, R) dataset ──────────
# Each sample: input = [beta, gamma, t/t_max], output = [S/N, I/N, R/N]
X_list, Y_list = [], []
for i in range(n_params):
    for j in range(n_points):
        X_list.append([all_beta[i], all_gamma[i], t_grid[j] / t_max])
        Y_list.append([all_S[i,j] / N, all_I[i,j] / N, all_R[i,j] / N])

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)
print(f"Training samples: {len(X)}")

# ── 3. Train / val split ─────────────────────────────────────────
idx   = np.random.RandomState(0).permutation(len(X))
split = int(0.9 * len(X))
X_train, Y_train = X[idx[:split]], Y[idx[:split]]
X_val,   Y_val   = X[idx[split:]], Y[idx[split:]]

class SIRDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

train_loader = DataLoader(SIRDataset(X_train, Y_train), batch_size=2048, shuffle=True)
val_loader   = DataLoader(SIRDataset(X_val,   Y_val),   batch_size=2048)

# ── 4. Define MLP model ──────────────────────────────────────────
class SIRMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),  nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 3),
        )
    def forward(self, x):
        out = self.net(x)
        # Softmax ensures S+I+R = 1 (conservation law)
        return torch.softmax(out, dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = SIRMLP().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 5. Training loop ─────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
best_val = float('inf')
epochs = 100

print("\nTraining started...")
print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12}")
print("-" * 38)

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    train_loss = total_loss / len(X_train)

    model.eval()
    with torch.no_grad():
        val_loss = sum(
            criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)
            for xb, yb in val_loader
        ) / len(X_val)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), f'{BASE}/data/best_model.pt')

    if epoch % 10 == 0:
        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f}")

print(f"\nBest val loss: {best_val:.6f}")
print("Model saved to data/best_model.pt")

# ── 6. Plot training curves ──────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses,   label='Val loss')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('Training curves'); plt.legend(); plt.yscale('log')
plt.tight_layout()
plt.savefig(f'{BASE}/data/training_curves.png', dpi=120)
print("Training curves saved to data/training_curves.png")

# ── 7. Visual check: predict vs actual ───────────────────────────
model.load_state_dict(torch.load(f'{BASE}/data/best_model.pt'))
model.eval()

# Test on a param point NOT in training — beta=0.4, gamma=0.15
beta_test, gamma_test = 0.4, 0.15
t_norm = torch.tensor(
    [[beta_test, gamma_test, t/t_max] for t in t_grid],
    dtype=torch.float32
).to(device)

with torch.no_grad():
    pred = model(t_norm).cpu().numpy() * N

plt.figure(figsize=(8, 4))
plt.plot(t_grid, pred[:,0], '--', color='steelblue', label='S predicted')
plt.plot(t_grid, pred[:,1], '--', color='tomato',    label='I predicted')
plt.plot(t_grid, pred[:,2], '--', color='seagreen',  label='R predicted')
plt.xlabel('Time'); plt.ylabel('Count')
plt.title(f'Model prediction (β={beta_test}, γ={gamma_test})')
plt.legend(); plt.tight_layout()
plt.savefig(f'{BASE}/data/prediction_check.png', dpi=120)
print("Prediction plot saved to data/prediction_check.png")
print("\nStep 6 complete!")
