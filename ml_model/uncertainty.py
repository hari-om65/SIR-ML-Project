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

# ── 2. MLP with Dropout layers ────────────────────────────────────
class SIR_MCDropout(nn.Module):
    """
    Same MLP but with Dropout after every hidden layer.
    Key trick: at INFERENCE time we keep dropout ON (model.train())
    and run forward pass 200 times — each gives a different prediction.
    The spread across 200 predictions = uncertainty estimate.
    This is called Monte Carlo Dropout.
    """
    def __init__(self, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),   nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(128, 256), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(256, 256), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(256, 128), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def predict_with_uncertainty(self, x, n_samples=200):
        """
        Run n_samples stochastic forward passes.
        Returns mean and std across samples.
        """
        self.train()   # keep dropout ON during inference
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x).unsqueeze(0))
        preds = torch.cat(preds, dim=0)   # (n_samples, batch, 3)
        mean  = preds.mean(dim=0)         # (batch, 3)
        std   = preds.std(dim=0)          # (batch, 3)
        return mean, std

# ── 3. Build training data ────────────────────────────────────────
X_list, Y_list = [], []
for i in range(len(all_beta)):
    for j in range(len(t_grid)):
        X_list.append([all_beta[i], all_gamma[i], t_grid[j]/t_max])
        Y_list.append([all_S[i,j]/N, all_I[i,j]/N, all_R[i,j]/N])

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)
idx   = np.random.RandomState(0).permutation(len(X))
split = int(0.9 * len(X))
X_tr  = torch.tensor(X[idx[:split]])
Y_tr  = torch.tensor(Y[idx[:split]])
X_val = torch.tensor(X[idx[split:]])
Y_val = torch.tensor(Y[idx[split:]])
print(f"Train: {len(X_tr)}  Val: {len(X_val)}")

# ── 4. Train MC Dropout model ─────────────────────────────────────
device    = torch.device('cpu')
model     = SIR_MCDropout(p_drop=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.MSELoss()
batch_size = 2048
best_val   = float('inf')
train_losses, val_losses = [], []

print("\nTraining MC Dropout model...")
print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12}")
print("-"*38)

for epoch in range(1, 101):
    model.train()
    perm       = torch.randperm(len(X_tr))
    total_loss = 0.0
    n_batches  = 0
    for start in range(0, len(X_tr), batch_size):
        xb = X_tr[perm[start:start+batch_size]].to(device)
        yb = Y_tr[perm[start:start+batch_size]].to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    scheduler.step()

    # Validation — turn dropout OFF for clean val loss
    model.eval()
    with torch.no_grad():
        v_loss = criterion(
            model(X_val.to(device)),
            Y_val.to(device)
        ).item()

    avg_train = total_loss / n_batches
    train_losses.append(avg_train)
    val_losses.append(v_loss)

    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), f'{BASE}/data/mc_dropout_model.pt')

    if epoch % 10 == 0:
        print(f"{epoch:>6} | {avg_train:>12.6f} | {v_loss:>12.6f}")

print(f"\nBest val loss: {best_val:.6f}")
print("MC Dropout model saved to data/mc_dropout_model.pt")

# ── 5. Generate uncertainty bands ────────────────────────────────
model.load_state_dict(torch.load(f'{BASE}/data/mc_dropout_model.pt'))
print("\nGenerating uncertainty bands (200 MC samples each)...")

def sir_ode(y, t, beta, gamma):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('MC Dropout — prediction uncertainty bands', fontsize=13)

test_cases = [(0.3, 0.1), (0.5, 0.2), (0.7, 0.3)]
colors     = [('steelblue','S'), ('tomato','I'), ('seagreen','R')]

for col, (b, g) in enumerate(test_cases):
    # ODE ground truth
    sol = odeint(sir_ode, [N-1, 1, 0], t_grid, args=(b, g))

    # Stochastic dataset mean (closest param point)
    idx_close = np.argmin(np.abs(all_beta-b) + np.abs(all_gamma-g))
    S_stoch   = all_S[idx_close]
    I_stoch   = all_I[idx_close]
    R_stoch   = all_R[idx_close]

    # MC Dropout prediction
    inp_t = torch.tensor(
        [[b, g, t/t_max] for t in t_grid], dtype=torch.float32)
    mean, std = model.predict_with_uncertainty(inp_t, n_samples=200)
    mean = mean.numpy() * N
    std  = std.numpy()  * N

    # Top row: uncertainty bands vs ODE
    ax = axes[0, col]
    for k, (c, lbl) in enumerate(colors):
        ax.plot(t_grid, sol[:, k], color=c, lw=2, alpha=0.6,
                label=f'{lbl} ODE')
        ax.plot(t_grid, mean[:, k], color=c, lw=1.5,
                linestyle='--', label=f'{lbl} ML mean')
        ax.fill_between(t_grid,
                        mean[:, k] - 2*std[:, k],
                        mean[:, k] + 2*std[:, k],
                        color=c, alpha=0.15,
                        label=f'{lbl} ±2σ')
    ax.set_title(f'β={b}, γ={g}  |  R₀={b/g:.1f}', fontsize=9)
    ax.set_xlabel('Time'); ax.set_ylabel('Count')
    if col == 0: ax.legend(fontsize=6, ncol=2)

    # Bottom row: uncertainty width over time
    ax2 = axes[1, col]
    ax2.plot(t_grid, 2*std[:, 0], color='steelblue', label='2σ S')
    ax2.plot(t_grid, 2*std[:, 1], color='tomato',    label='2σ I')
    ax2.plot(t_grid, 2*std[:, 2], color='seagreen',  label='2σ R')
    ax2.set_title(f'Uncertainty width (β={b}, γ={g})', fontsize=9)
    ax2.set_xlabel('Time'); ax2.set_ylabel('±2σ band width')
    ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{BASE}/data/uncertainty_bands.png', dpi=120)
print("Uncertainty plot saved to data/uncertainty_bands.png")

# ── 6. Print summary stats ────────────────────────────────────────
print("\nUncertainty summary for β=0.3, γ=0.1:")
b, g    = 0.3, 0.1
inp_t   = torch.tensor(
    [[b, g, t/t_max] for t in t_grid], dtype=torch.float32)
mean, std = model.predict_with_uncertainty(inp_t, n_samples=200)
mean = mean.numpy() * N
std  = std.numpy()  * N
peak_idx = mean[:, 1].argmax()
print(f"  Peak I     = {mean[peak_idx,1]:.1f} ± {2*std[peak_idx,1]:.1f}  (at t={t_grid[peak_idx]:.1f})")
print(f"  Final R    = {mean[-1,2]:.1f} ± {2*std[-1,2]:.1f}")
print(f"  Mean 2σ(I) = {(2*std[:,1]).mean():.2f} across all time steps")
print("\nUpgrade 3 complete — Uncertainty bands done!")
