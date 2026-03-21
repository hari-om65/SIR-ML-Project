import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint
from torch.utils.data import DataLoader, TensorDataset
import os

BASE   = '/teamspace/studios/this_studio/sir_ml_project'
N      = 1000.0
t_max  = 160.0
t_grid = np.load(f'{BASE}/data/t_grid.npy').astype('float32')

print("=== LOADING EXISTING DATA ===")
all_S     = np.load(f'{BASE}/data/all_S.npy').astype('float32')
all_I     = np.load(f'{BASE}/data/all_I.npy').astype('float32')
all_R     = np.load(f'{BASE}/data/all_R.npy').astype('float32')
all_beta  = np.load(f'{BASE}/data/all_beta.npy').astype('float32')
all_gamma = np.load(f'{BASE}/data/all_gamma.npy').astype('float32')
print(f"Loaded {len(all_beta)} existing trajectories")

print("=== GENERATING EXTRA ODE TRAJECTORIES ===")
def sir_ode(y, t, beta, gamma):
    S,I,R = y
    return [-beta*S*I/N, beta*S*I/N-gamma*I, gamma*I]

extra_S, extra_I, extra_R = [], [], []
extra_beta, extra_gamma   = [], []

np.random.seed(42)
# Dense grid: 40x40 = 1600 extra combos
betas  = np.linspace(0.10, 0.90, 40)
gammas = np.linspace(0.05, 0.50, 40)
count = 0
for b in betas:
    for g in gammas:
        sol = odeint(sir_ode, [N-1, 1, 0], t_grid, args=(float(b), float(g)))
        extra_S.append(sol[:,0].astype('float32'))
        extra_I.append(sol[:,1].astype('float32'))
        extra_R.append(sol[:,2].astype('float32'))
        extra_beta.append(float(b))
        extra_gamma.append(float(g))
        count += 1

extra_S     = np.array(extra_S,     dtype='float32')
extra_I     = np.array(extra_I,     dtype='float32')
extra_R     = np.array(extra_R,     dtype='float32')
extra_beta  = np.array(extra_beta,  dtype='float32')
extra_gamma = np.array(extra_gamma, dtype='float32')
print(f"Generated {count} extra ODE trajectories")

# Combine all data
all_S     = np.concatenate([all_S,     extra_S],     axis=0)
all_I     = np.concatenate([all_I,     extra_I],     axis=0)
all_R     = np.concatenate([all_R,     extra_R],     axis=0)
all_beta  = np.concatenate([all_beta,  extra_beta],  axis=0)
all_gamma = np.concatenate([all_gamma, extra_gamma], axis=0)
print(f"Total trajectories: {len(all_beta)}")

print("=== BUILDING TRAINING TENSORS ===")
# Flatten: each (beta, gamma, t) -> (S, I, R)/N
n_traj  = len(all_beta)
n_t     = len(t_grid)
t_norm  = t_grid / t_max          # [0,1]
b_norm  = all_beta  / 0.90        # [0,1]
g_norm  = all_gamma / 0.50        # [0,1]

X_list, Y_list = [], []
for i in range(n_traj):
    for j in range(n_t):
        X_list.append([b_norm[i], g_norm[i], t_norm[j]])
        Y_list.append([all_S[i,j]/N, all_I[i,j]/N, all_R[i,j]/N])

X = torch.tensor(X_list, dtype=torch.float32)
Y = torch.tensor(Y_list, dtype=torch.float32)
print(f"Dataset size: {len(X)} samples")

# Shuffle and split
idx   = torch.randperm(len(X))
X, Y  = X[idx], Y[idx]
n_val = int(0.1 * len(X))
X_val, Y_val     = X[:n_val],  Y[:n_val]
X_tr,  Y_tr      = X[n_val:],  Y[n_val:]
print(f"Train: {len(X_tr)}  Val: {len(X_val)}")

tr_loader = DataLoader(TensorDataset(X_tr, Y_tr),
                       batch_size=2048, shuffle=True)

print("=== DEFINING IMPROVED MODEL ===")
class ImprovedSIRMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256),  nn.Tanh(),
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 3),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

model = ImprovedSIRMLP()
total_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {total_params:,}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)

print("=== TRAINING ===")
best_val = float('inf')
for epoch in range(1, 301):
    model.train()
    tr_loss = 0.0
    for xb, yb in tr_loader:
        opt.zero_grad()
        loss = torch.mean((model(xb) - yb)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr_loss += loss.item()
    sch.step()

    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_loss = torch.mean((model(X_val) - Y_val)**2).item()
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       f'{BASE}/data/best_model.pt')
            print(f"Epoch {epoch:3d} | tr={tr_loss/len(tr_loader):.6f} "
                  f"| val={val_loss:.6f}  <-- saved")
        else:
            print(f"Epoch {epoch:3d} | tr={tr_loss/len(tr_loader):.6f} "
                  f"| val={val_loss:.6f}")

print(f"\nBest val loss: {best_val:.6f}")

print("=== FINAL EVALUATION ===")
model.load_state_dict(torch.load(f'{BASE}/data/best_model.pt',
                                  map_location='cpu'))
model.eval()

test_cases = [(0.3,0.1),(0.5,0.2),(0.7,0.15),(0.2,0.1),(0.8,0.4)]
for b, g in test_cases:
    b_n = b/0.9; g_n = g/0.5
    inp = torch.tensor([[b_n, g_n, t/t_max] for t in t_grid],
                       dtype=torch.float32)
    with torch.no_grad():
        pred = model(inp).numpy() * N
    sol  = odeint(sir_ode, [N-1,1,0], t_grid, args=(b,g))
    r2   = [round(float(1 - np.sum((pred[:,k]-sol[:,k])**2) /
            (np.sum((sol[:,k]-sol[:,k].mean())**2)+1e-12)), 4)
            for k in range(3)]
    mse  = [round(float(np.mean((pred[:,k]-sol[:,k])**2)), 2)
            for k in range(3)]
    print(f"b={b} g={g} | R2 S/I/R={r2} | MSE S/I/R={mse}")

print("\nDone! best_model.pt updated.")
