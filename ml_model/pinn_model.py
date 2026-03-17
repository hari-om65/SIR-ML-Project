from pysr import PySRRegressor
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint

BASE  = '/teamspace/studios/this_studio/sir_ml_project'
N     = 1000.0
t_max = 160.0

# ── 1. Load dataset ───────────────────────────────────────────────
t_grid    = np.load(f'{BASE}/data/t_grid.npy')
all_beta  = np.load(f'{BASE}/data/all_beta.npy')
all_gamma = np.load(f'{BASE}/data/all_gamma.npy')
all_S     = np.load(f'{BASE}/data/all_S.npy')
all_I     = np.load(f'{BASE}/data/all_I.npy')
all_R     = np.load(f'{BASE}/data/all_R.npy')
print(f"Dataset loaded: {len(all_beta)} param points")

# ── 2. PINN Model ─────────────────────────────────────────────────
class PINN_SIR(nn.Module):
    """
    Same MLP as before BUT training uses two losses:
      data_loss    = MSE vs stochastic mean trajectories
      physics_loss = how much predictions VIOLATE the SIR ODEs
    Total loss = data_loss + lambda * physics_loss
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),   nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

# ── 3. Physics loss function ──────────────────────────────────────
def physics_loss(model, beta_vals, gamma_vals, n_coll=500):
    """
    Sample random (beta, gamma, t) collocation points.
    Compute dS/dt, dI/dt, dR/dt via autograd.
    Penalise deviation from true SIR equations:
      dS/dt = -beta * S * I / N
      dI/dt =  beta * S * I / N - gamma * I
      dR/dt =  gamma * I
    """
    # Random collocation points
    b = torch.tensor(
        np.random.choice(beta_vals,  n_coll).astype(np.float32))
    g = torch.tensor(
        np.random.choice(gamma_vals, n_coll).astype(np.float32))
    t = torch.rand(n_coll)   # t in [0,1] normalised

    inp = torch.stack([b, g, t], dim=1)
    inp.requires_grad_(True)

    out   = model(inp)        # (n_coll, 3) — normalised S,I,R
    S_n   = out[:, 0]
    I_n   = out[:, 1]
    R_n   = out[:, 2]

    # Unnormalise
    S = S_n * N
    I = I_n * N
    R = R_n * N

    # Compute d/dt via autograd (chain rule through t)
    dSdt = torch.autograd.grad(
        S.sum(), inp, create_graph=True)[0][:, 2] / (1.0/t_max)
    dIdt = torch.autograd.grad(
        I.sum(), inp, create_graph=True)[0][:, 2] / (1.0/t_max)
    dRdt = torch.autograd.grad(
        R.sum(), inp, create_graph=True)[0][:, 2] / (1.0/t_max)

    # True SIR derivatives
    SI_N = S * I / N
    dSdt_true = -b * SI_N
    dIdt_true =  b * SI_N - g * I
    dRdt_true =  g * I

    # Physics residual loss
    loss_S = torch.mean((dSdt - dSdt_true)**2)
    loss_I = torch.mean((dIdt - dIdt_true)**2)
    loss_R = torch.mean((dRdt - dRdt_true)**2)

    return (loss_S + loss_I + loss_R) / (N**2)

# ── 4. Build flat training data ───────────────────────────────────
X_list, Y_list = [], []
for i in range(len(all_beta)):
    for j in range(len(t_grid)):
        X_list.append([all_beta[i], all_gamma[i], t_grid[j] / t_max])
        Y_list.append([all_S[i,j]/N, all_I[i,j]/N, all_R[i,j]/N])

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)

idx    = np.random.RandomState(0).permutation(len(X))
split  = int(0.9 * len(X))
X_tr   = torch.tensor(X[idx[:split]])
Y_tr   = torch.tensor(Y[idx[:split]])
X_val  = torch.tensor(X[idx[split:]])
Y_val  = torch.tensor(Y[idx[split:]])
print(f"Train: {len(X_tr)}  Val: {len(X_val)}")

# ── 5. Training with physics loss ─────────────────────────────────
device    = torch.device('cpu')
model     = PINN_SIR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
mse       = nn.MSELoss()

LAMBDA       = 0.1   # weight of physics loss vs data loss
batch_size   = 2048
best_val     = float('inf')
train_losses = []
phys_losses  = []
val_losses   = []

print("\nTraining PINN (data loss + physics loss)...")
print(f"{'Epoch':>6} | {'Data Loss':>12} | {'Phys Loss':>12} | {'Val Loss':>12}")
print("-"*52)

for epoch in range(1, 101):
    model.train()
    perm  = torch.randperm(len(X_tr))
    total_data = 0.0
    total_phys = 0.0
    n_batches  = 0

    for start in range(0, len(X_tr), batch_size):
        xb = X_tr[perm[start:start+batch_size]].to(device)
        yb = Y_tr[perm[start:start+batch_size]].to(device)

        optimizer.zero_grad()
        d_loss = mse(model(xb), yb)
        p_loss = physics_loss(model, all_beta, all_gamma, n_coll=300)
        loss   = d_loss + LAMBDA * p_loss
        loss.backward()
        optimizer.step()

        total_data += d_loss.item()
        total_phys += p_loss.item()
        n_batches  += 1

    scheduler.step()
    avg_data = total_data / n_batches
    avg_phys = total_phys / n_batches

    model.eval()
    with torch.no_grad():
        v_loss = mse(model(X_val.to(device)), Y_val.to(device)).item()

    train_losses.append(avg_data)
    phys_losses.append(avg_phys)
    val_losses.append(v_loss)

    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), f'{BASE}/data/pinn_model.pt')

    if epoch % 10 == 0:
        print(f"{epoch:>6} | {avg_data:>12.6f} | {avg_phys:>12.6f} | {v_loss:>12.6f}")

print(f"\nBest val loss: {best_val:.6f}")
print("PINN model saved to data/pinn_model.pt")

# ── 6. Compare PINN vs plain MLP ──────────────────────────────────
# Load plain MLP for comparison
class PlainMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,128), nn.Tanh(),
            nn.Linear(128,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,3),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

plain = PlainMLP()
plain.load_state_dict(torch.load(f'{BASE}/data/best_model.pt', map_location='cpu'))
plain.eval()

model.load_state_dict(torch.load(f'{BASE}/data/pinn_model.pt'))
model.eval()

def sir_ode(y, t, beta, gamma):
    S,I,R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('PINN vs plain MLP — physics constraints improve accuracy', fontsize=12)

test_cases = [(0.3, 0.1), (0.5, 0.2), (0.7, 0.3)]
for col, (b, g) in enumerate(test_cases):
    sol = odeint(sir_ode, [N-1, 1, 0], t_grid, args=(b, g))
    inp_t = torch.tensor(
        [[b, g, t/t_max] for t in t_grid], dtype=torch.float32)

    with torch.no_grad():
        pred_pinn  = model(inp_t).numpy() * N
        pred_plain = plain(inp_t).numpy() * N

    for row, (pred, title) in enumerate([
        (pred_pinn,  f'PINN  β={b} γ={g}'),
        (pred_plain, f'Plain MLP  β={b} γ={g}')
    ]):
        ax = axes[row, col]
        ax.plot(t_grid, sol[:,0], 'b-',  lw=2,   alpha=0.5, label='S ODE')
        ax.plot(t_grid, sol[:,1], 'r-',  lw=2,   alpha=0.5, label='I ODE')
        ax.plot(t_grid, sol[:,2], 'g-',  lw=2,   alpha=0.5, label='R ODE')
        ax.plot(t_grid, pred[:,0], 'b--', lw=1.5, label='S pred')
        ax.plot(t_grid, pred[:,1], 'r--', lw=1.5, label='I pred')
        ax.plot(t_grid, pred[:,2], 'g--', lw=1.5, label='R pred')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time'); ax.set_ylabel('Count')
        if col == 0: ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{BASE}/data/pinn_comparison.png', dpi=120)
print("Comparison plot saved to data/pinn_comparison.png")

# Loss curves
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Data loss',    color='steelblue')
plt.plot(phys_losses,  label='Physics loss', color='tomato')
plt.plot(val_losses,   label='Val loss',     color='seagreen')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('PINN training — data loss + physics loss')
plt.legend(); plt.yscale('log'); plt.tight_layout()
plt.savefig(f'{BASE}/data/pinn_loss_curves.png', dpi=120)
print("Loss curves saved to data/pinn_loss_curves.png")
print("\nUpgrade 2 complete — PINN done!")
