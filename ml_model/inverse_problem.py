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

class SIRMLP(nn.Module):
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

model = SIRMLP()
model.load_state_dict(torch.load(
    f'{BASE}/data/best_model.pt', map_location='cpu'))
model.eval()
print("Model loaded")

t_grid = np.load(f'{BASE}/data/t_grid.npy').astype(np.float32)

def sir_ode(y, t, beta, gamma):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

def infer_params(S_obs, I_obs, R_obs, n_steps=500, lr=0.05, verbose=True):
    S_t = torch.tensor(S_obs / N, dtype=torch.float32)
    I_t = torch.tensor(I_obs / N, dtype=torch.float32)
    R_t = torch.tensor(R_obs / N, dtype=torch.float32)
    Y_t = torch.stack([S_t, I_t, R_t], dim=1)

    log_b = nn.Parameter(torch.tensor([np.log(0.3)], dtype=torch.float32))
    log_g = nn.Parameter(torch.tensor([np.log(0.15)], dtype=torch.float32))

    optimizer = torch.optim.Adam([log_b, log_g], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps)

    b_history, g_history, loss_history = [], [], []

    for step in range(n_steps):
        optimizer.zero_grad()
        b = torch.exp(log_b).clamp(0.05, 1.5)
        g = torch.exp(log_g).clamp(0.01, 0.8)

        t_norm = torch.tensor(t_grid / t_max, dtype=torch.float32)
        b_exp  = b.expand(len(t_grid))
        g_exp  = g.expand(len(t_grid))
        inp    = torch.stack([b_exp, g_exp, t_norm], dim=1).float()

        pred = model(inp)
        loss = torch.mean((pred - Y_t)**2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        b_history.append(b.item())
        g_history.append(g.item())

        if verbose and step % 100 == 0:
            print(f"  step {step:>4} | loss={loss.item():.6f} "
                  f"| β={b.item():.4f}  γ={g.item():.4f}")

    return (torch.exp(log_b).item(),
            torch.exp(log_g).item(),
            loss_history, b_history, g_history)

print("\n" + "="*55)
print("INVERSE PROBLEM — inferring β and γ from observations")
print("="*55)

test_cases = [
    (0.35, 0.12, "Case A — mild"),
    (0.55, 0.18, "Case B — moderate"),
    (0.75, 0.25, "Case C — severe"),
    (0.45, 0.30, "Case D — high recovery"),
]

results = []
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Inverse Problem — inferring β and γ from epidemic curves', fontsize=12)

for col, (b_true, g_true, label) in enumerate(test_cases):
    print(f"\n{label}  (true β={b_true}, γ={g_true}  R₀={b_true/g_true:.2f})")

    sol   = odeint(sir_ode, [N-1, 1, 0], t_grid, args=(b_true, g_true))
    noise = np.random.RandomState(42).normal(0, 5, sol.shape).astype(np.float32)
    S_obs = np.clip(sol[:,0] + noise[:,0], 0, N).astype(np.float32)
    I_obs = np.clip(sol[:,1] + noise[:,1], 0, N).astype(np.float32)
    R_obs = np.clip(sol[:,2] + noise[:,2], 0, N).astype(np.float32)

    b_est, g_est, losses, b_hist, g_hist = infer_params(
        S_obs, I_obs, R_obs, n_steps=500, lr=0.05, verbose=True)

    b_err  = abs(b_est - b_true) / b_true * 100
    g_err  = abs(g_est - g_true) / g_true * 100
    R0_err = abs(b_est/g_est - b_true/g_true) / (b_true/g_true) * 100

    print(f"  TRUE:     β={b_true:.3f}  γ={g_true:.3f}  R₀={b_true/g_true:.2f}")
    print(f"  INFERRED: β={b_est:.3f}  γ={g_est:.3f}  R₀={b_est/g_est:.2f}")
    print(f"  ERROR:    β={b_err:.1f}%   γ={g_err:.1f}%   R₀={R0_err:.1f}%")
    results.append((label, b_true, g_true, b_est, g_est, b_err, g_err))

    inp_t = torch.tensor(
        [[b_est, g_est, t/t_max] for t in t_grid],
        dtype=torch.float32)
    with torch.no_grad():
        pred = model(inp_t).numpy() * N

    ax = axes[0, col]
    ax.scatter(t_grid[::5], S_obs[::5], s=8, color='steelblue', alpha=0.5, label='S obs')
    ax.scatter(t_grid[::5], I_obs[::5], s=8, color='tomato',    alpha=0.5, label='I obs')
    ax.scatter(t_grid[::5], R_obs[::5], s=8, color='seagreen',  alpha=0.5, label='R obs')
    ax.plot(t_grid, pred[:,0], 'b-', lw=2, label='S fit')
    ax.plot(t_grid, pred[:,1], 'r-', lw=2, label='I fit')
    ax.plot(t_grid, pred[:,2], 'g-', lw=2, label='R fit')
    ax.set_title(f'{label}\nTrue β={b_true} γ={g_true}\nEst β={b_est:.3f} γ={g_est:.3f}',
                 fontsize=7)
    ax.set_xlabel('Time'); ax.set_ylabel('Count')
    if col == 0: ax.legend(fontsize=6)

    ax2 = axes[1, col]
    ax2.axhline(b_true, color='steelblue', lw=2, linestyle='--', label=f'True β={b_true}')
    ax2.axhline(g_true, color='tomato',    lw=2, linestyle='--', label=f'True γ={g_true}')
    ax2.plot(b_hist, color='steelblue', lw=1.5, label='Est β')
    ax2.plot(g_hist, color='tomato',    lw=1.5, label='Est γ')
    ax2.set_title('Parameter convergence', fontsize=8)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Value')
    ax2.legend(fontsize=6)

plt.tight_layout()
plt.savefig(f'{BASE}/data/inverse_problem.png', dpi=120)
print("\nPlot saved to data/inverse_problem.png")

print("\n" + "="*55)
print(f"{'Case':<25} {'β err':>8} {'γ err':>8}")
print("-"*45)
for (label, b_true, g_true, b_est, g_est, b_err, g_err) in results:
    print(f"{label:<25} {b_err:>7.1f}%  {g_err:>7.1f}%")
print("="*55)
print("\nUpgrade 4 complete — Inverse problem done!")
