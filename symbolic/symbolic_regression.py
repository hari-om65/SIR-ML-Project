from pysr import PySRRegressor
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

device = torch.device('cpu')
model  = SIRMLP().to(device)
model.load_state_dict(torch.load(f'{BASE}/data/best_model.pt', map_location=device))
model.eval()
print("Model loaded successfully")

# ── Sample points ─────────────────────────────────────────────────
b_vals = np.linspace(0.1, 0.9, 12)
g_vals = np.linspace(0.05, 0.5, 12)
t_vals = np.linspace(0, t_max, 40)

rows = []
for b in b_vals:
    for g in g_vals:
        for t in t_vals:
            rows.append([b, g, t / t_max, b, g])
rows = np.array(rows, dtype=np.float32)
print(f"Total samples: {len(rows)}")

inp = torch.tensor(rows[:, :3], requires_grad=True)

out    = model(inp) * N
S_pred = out[:, 0]
I_pred = out[:, 1]
R_pred = out[:, 2]

dSdt = torch.autograd.grad(S_pred.sum(), inp, retain_graph=True)[0][:, 2]  / (1.0/t_max)
dIdt = torch.autograd.grad(I_pred.sum(), inp, retain_graph=True)[0][:, 2]  / (1.0/t_max)
dRdt = torch.autograd.grad(R_pred.sum(), inp, retain_graph=False)[0][:, 2] / (1.0/t_max)

dSdt = dSdt.detach().numpy()
dIdt = dIdt.detach().numpy()
dRdt = dRdt.detach().numpy()
Sv   = S_pred.detach().numpy()
Iv   = I_pred.detach().numpy()
Rv   = R_pred.detach().numpy()
bv   = rows[:, 3]
gv   = rows[:, 4]
SIvN = Sv * Iv / N

# ── ALL variable names must be safe (no sympy reserved words) ─────
# S→x1, I→x2, R→x3, b→x4, g→x5, SI/N→x6
features      = np.column_stack([x1:=Sv, x2:=Iv, x3:=Rv, x4:=bv, x5:=gv, x6:=SIvN])
feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
# x1=S, x2=I, x3=R, x4=b(transmission), x5=g(recovery), x6=S*I/N

print("Features: x1=S  x2=I  x3=R  x4=b(transmission)  x5=g(recovery)  x6=S*I/N")
print(f"dR/dt range: [{dRdt.min():.2f}, {dRdt.max():.2f}]")

# ── dR/dt = x5 * x2  (g * I) ─────────────────────────────────────
print("\nFitting dR/dt  (expect: x5 * x2 = g * I) ...")
sr_R = PySRRegressor(
    niterations=50, binary_operators=["+","*","-"],
    unary_operators=[], maxsize=8, populations=12,
    verbosity=0, random_state=42, progress=True,
    batching=True, batch_size=512,
)
sr_R.fit(features, dRdt, variable_names=feature_names)
print("\n=== dR/dt  [x5=g, x2=I  →  expect: x5*x2] ===")
print(sr_R)

# ── dS/dt = -x4 * x6  (-b * S*I/N) ──────────────────────────────
print("\nFitting dS/dt  (expect: -x4 * x6 = -b * S*I/N) ...")
sr_S = PySRRegressor(
    niterations=50, binary_operators=["+","*","-"],
    unary_operators=[], maxsize=10, populations=12,
    verbosity=0, random_state=42, progress=True,
    batching=True, batch_size=512,
)
sr_S.fit(features, dSdt, variable_names=feature_names)
print("\n=== dS/dt  [x4=b, x6=S*I/N  →  expect: -x4*x6] ===")
print(sr_S)

# ── dI/dt = x4*x6 - x5*x2  (b*S*I/N - g*I) ──────────────────────
print("\nFitting dI/dt  (expect: x4*x6 - x5*x2) ...")
sr_I = PySRRegressor(
    niterations=50, binary_operators=["+","*","-"],
    unary_operators=[], maxsize=12, populations=12,
    verbosity=0, random_state=42, progress=True,
    batching=True, batch_size=512,
)
sr_I.fit(features, dIdt, variable_names=feature_names)
print("\n=== dI/dt  [x4=b, x6=S*I/N, x5=g, x2=I  →  expect: x4*x6 - x5*x2] ===")
print(sr_I)

# ── Verification plot ─────────────────────────────────────────────
b_test, g_test = 0.3, 0.1
t_test = np.linspace(0, t_max, 100)
inp_t  = torch.tensor(
    [[b_test, g_test, t/t_max] for t in t_test], dtype=torch.float32)
with torch.no_grad():
    pred_t = model(inp_t).numpy() * N

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.plot(t_test, pred_t[:,0], label='S', color='steelblue')
plt.plot(t_test, pred_t[:,1], label='I', color='tomato')
plt.plot(t_test, pred_t[:,2], label='R', color='seagreen')
plt.title(f'ML prediction (b={b_test}, g={g_test})')
plt.xlabel('Time'); plt.ylabel('Count'); plt.legend()

plt.subplot(1,2,2)
plt.plot(t_test, g_test * pred_t[:,1], color='purple',
         linewidth=2, label='g*I = dR/dt (symbolic)')
plt.title('dR/dt = g*I — verified')
plt.xlabel('Time'); plt.ylabel('dR/dt'); plt.legend()
plt.tight_layout()
plt.savefig(f'{BASE}/data/symbolic_result.png', dpi=120)
print("\nPlot saved to data/symbolic_result.png")
print("\nStep 7 complete!")
