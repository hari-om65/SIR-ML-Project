from pysr import PySRRegressor
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchdiffeq import odeint

BASE  = '/teamspace/studios/this_studio/sir_ml_project'
N     = 1000.0
t_max = 160.0

# ── 1. Load dataset ───────────────────────────────────────────────
t_grid    = torch.tensor(np.load(f'{BASE}/data/t_grid.npy'),    dtype=torch.float32)
all_beta  = np.load(f'{BASE}/data/all_beta.npy')
all_gamma = np.load(f'{BASE}/data/all_gamma.npy')
all_S     = np.load(f'{BASE}/data/all_S.npy')
all_I     = np.load(f'{BASE}/data/all_I.npy')
all_R     = np.load(f'{BASE}/data/all_R.npy')
print(f"Dataset loaded: {len(all_beta)} param points")

# ── 2. Neural ODE: learns dS/dt, dI/dt, dR/dt directly ──────────
class SIRODEFunc(nn.Module):
    """
    This network learns the DERIVATIVE function f(y, t, params).
    Instead of predicting S,I,R directly it learns HOW they change.
    The ODE solver then integrates forward to get S(t), I(t), R(t).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),   nn.Tanh(),   # input: [S,I,R, beta, gamma]
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 3),               # output: [dS/dt, dI/dt, dR/dt]
        )

    def forward(self, t, y):
        # y shape: (batch, 3) = [S, I, R] normalised by N
        return self.net(y)

class NeuralODESIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.func   = SIRODEFunc()
        self.params = nn.Parameter(torch.zeros(2))  # learns [beta, gamma]

    def forward(self, y0, t_span, beta, gamma):
        # Pack parameters into state so ODE func can see them
        batch = y0.shape[0]

        def aug_func(t, y):
            # y: (batch, 5) = [S, I, R, beta, gamma]
            sir   = y[:, :3]
            dydt  = self.func(t, y)
            # beta, gamma are constant — their derivatives are 0
            zeros = torch.zeros(batch, 2, device=y.device)
            return torch.cat([dydt, zeros], dim=1)

        bg    = torch.stack([beta, gamma], dim=1)          # (batch, 2)
        y0_aug = torch.cat([y0, bg], dim=1)                # (batch, 5)
        sol    = odeint(aug_func, y0_aug, t_span,
                        method='rk4',
                        options={'step_size': 2.0})         # (T, batch, 5)
        return torch.softmax(sol[:, :, :3], dim=-1)        # normalised S,I,R

# ── 3. Prepare training data ──────────────────────────────────────
print("Preparing training batches...")
# Use subset of parameter points for speed
idx_train = np.random.RandomState(0).choice(len(all_beta), 200, replace=False)

def make_batch(indices):
    beta_b  = torch.tensor(all_beta[indices],  dtype=torch.float32)
    gamma_b = torch.tensor(all_gamma[indices], dtype=torch.float32)
    S_b = torch.tensor(all_S[indices] / N, dtype=torch.float32)
    I_b = torch.tensor(all_I[indices] / N, dtype=torch.float32)
    R_b = torch.tensor(all_R[indices] / N, dtype=torch.float32)
    y_true = torch.stack([S_b, I_b, R_b], dim=2)  # (batch, T, 3)
    return beta_b, gamma_b, y_true

# ── 4. Training loop ──────────────────────────────────────────────
device    = torch.device('cpu')
node      = NeuralODESIR().to(device)
optimizer = torch.optim.Adam(node.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.MSELoss()

# Use fewer time points for speed during training
t_train   = t_grid[::8].to(device)   # every 8th point = 21 points
t_idx     = list(range(0, len(t_grid), 8))

print("\nTraining Neural ODE...")
print(f"{'Epoch':>6} | {'Loss':>12}")
print("-"*22)

best_loss = float('inf')
losses    = []

for epoch in range(1, 61):
    node.train()
    # Mini-batch
    perm  = np.random.permutation(len(idx_train))[:32]
    bidx  = idx_train[perm]
    beta_b, gamma_b, y_true = make_batch(bidx)

    # Initial condition: [S0, I0, R0] = [(N-1)/N, 1/N, 0]
    y0 = torch.zeros(len(bidx), 3)
    y0[:, 0] = (N - 1) / N
    y0[:, 1] = 1.0 / N
    y0[:, 2] = 0.0

    optimizer.zero_grad()
    y_pred = node(y0, t_train, beta_b, gamma_b)  # (T, batch, 3)
    y_pred = y_pred.permute(1, 0, 2)              # (batch, T, 3)
    loss   = criterion(y_pred, y_true[:, t_idx, :])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(node.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(node.state_dict(), f'{BASE}/data/neural_ode.pt')

    if epoch % 10 == 0:
        print(f"{epoch:>6} | {loss.item():>12.6f}")

print(f"\nBest loss: {best_loss:.6f}")
print("Neural ODE saved to data/neural_ode.pt")

# ── 5. Plot Neural ODE predictions ───────────────────────────────
node.load_state_dict(torch.load(f'{BASE}/data/neural_ode.pt'))
node.eval()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Neural ODE — learns derivative equations directly', fontsize=12)

test_cases = [(0.3, 0.1), (0.5, 0.2), (0.7, 0.3)]
for ax, (b, g) in zip(axes, test_cases):
    y0    = torch.tensor([[(N-1)/N, 1.0/N, 0.0]])
    beta  = torch.tensor([b])
    gamma = torch.tensor([g])

    with torch.no_grad():
        pred = node(y0, t_grid.to(device), beta, gamma)  # (T,1,3)
    pred = pred[:, 0, :].numpy() * N

    ax.plot(t_grid.numpy(), pred[:, 0], color='steelblue', label='S')
    ax.plot(t_grid.numpy(), pred[:, 1], color='tomato',    label='I')
    ax.plot(t_grid.numpy(), pred[:, 2], color='seagreen',  label='R')
    ax.set_title(f'β={b}, γ={g}  R₀={b/g:.1f}')
    ax.set_xlabel('Time'); ax.set_ylabel('Count')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{BASE}/data/neural_ode_result.png', dpi=120)
print("Plot saved to data/neural_ode_result.png")
print("\nUpgrade 1 complete — Neural ODE done!")
