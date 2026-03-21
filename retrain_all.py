import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint
from torch.utils.data import DataLoader, TensorDataset

BASE   = '/teamspace/studios/this_studio/sir_ml_project'
N      = 1000.0
t_max  = 160.0
t_grid = np.load(f'{BASE}/data/t_grid.npy').astype('float32')

def sir_ode(y, t, beta, gamma):
    S,I,R = y
    return [-beta*S*I/N, beta*S*I/N-gamma*I, gamma*I]

# ── Build dataset ─────────────────────────────────────────────────
print("Building dataset...")
betas  = np.linspace(0.10, 0.90, 40)
gammas = np.linspace(0.05, 0.50, 40)
X_list, Y_list = [], []
t_norm = t_grid / t_max
for b in betas:
    for g in gammas:
        sol = odeint(sir_ode, [N-1,1,0], t_grid, args=(float(b),float(g)))
        for j in range(len(t_grid)):
            X_list.append([b/0.9, g/0.5, t_norm[j]])
            Y_list.append([sol[j,0]/N, sol[j,1]/N, sol[j,2]/N])

X = torch.tensor(X_list, dtype=torch.float32)
Y = torch.tensor(Y_list, dtype=torch.float32)
idx = torch.randperm(len(X))
X, Y = X[idx], Y[idx]
n_val = int(0.1*len(X))
X_val,Y_val = X[:n_val], Y[:n_val]
X_tr, Y_tr  = X[n_val:], Y[n_val:]
print(f"Train={len(X_tr)}  Val={len(X_val)}")
tr_loader = DataLoader(TensorDataset(X_tr,Y_tr), batch_size=2048, shuffle=True)

# ── Architectures ─────────────────────────────────────────────────
class SIRMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,256),   nn.Tanh(),
            nn.Linear(256,512), nn.Tanh(),
            nn.Linear(512,512), nn.Tanh(),
            nn.Linear(512,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
            nn.Linear(128,3),
        )
    def forward(self,x):
        return torch.softmax(self.net(x), dim=-1)

class SIR_MCDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,256),   nn.Tanh(), nn.Dropout(p),
            nn.Linear(256,512), nn.Tanh(), nn.Dropout(p),
            nn.Linear(512,512), nn.Tanh(), nn.Dropout(p),
            nn.Linear(512,256), nn.Tanh(), nn.Dropout(p),
            nn.Linear(256,128), nn.Tanh(), nn.Dropout(p),
            nn.Linear(128,3),
        )
    def forward(self,x):
        return torch.softmax(self.net(x), dim=-1)

def train_model(model, name, epochs=300, lr=1e-3, pinn=False):
    print(f"\n=== Training {name} ({epochs} epochs) ===")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = torch.mean((pred - yb)**2)
            if pinn:
                # Physics loss on random batch
                xb_p = xb.clone().requires_grad_(True)
                out_p = model(xb_p) * N
                S_p,I_p,R_p = out_p[:,0],out_p[:,1],out_p[:,2]
                b_real = xb_p[:,0] * 0.9
                g_real = xb_p[:,1] * 0.5
                dSdt = torch.autograd.grad(S_p.sum(),xb_p,create_graph=True)[0][:,2]*t_max
                dIdt = torch.autograd.grad(I_p.sum(),xb_p,create_graph=True)[0][:,2]*t_max
                dRdt = torch.autograd.grad(R_p.sum(),xb_p,create_graph=True)[0][:,2]*t_max
                phy = (torch.mean((dSdt + b_real*S_p*I_p/N)**2) +
                       torch.mean((dIdt - b_real*S_p*I_p/N + g_real*I_p)**2) +
                       torch.mean((dRdt - g_real*I_p)**2))
                loss = loss + 0.01 * phy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sch.step()
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val = torch.mean((model(X_val)-Y_val)**2).item()
            if val < best_val:
                best_val = val
                torch.save(model.state_dict(), f'{BASE}/data/{name}.pt')
                print(f"  Epoch {epoch:3d} | val={val:.6f} <-- saved")
            else:
                print(f"  Epoch {epoch:3d} | val={val:.6f}")
    print(f"  Best val: {best_val:.6f}")
    return model

# ── Train all 3 models ────────────────────────────────────────────
mlp  = SIRMLP()
train_model(mlp, 'best_model', epochs=300)

mc   = SIR_MCDropout()
train_model(mc,  'mc_dropout_model', epochs=200)

pinn = SIRMLP()
train_model(pinn,'pinn_model', epochs=200, pinn=True)

# ── Final evaluation ──────────────────────────────────────────────
print("\n=== FINAL EVALUATION ===")
mlp.load_state_dict(torch.load(f'{BASE}/data/best_model.pt', map_location='cpu'))
mlp.eval()
test_cases = [(0.3,0.1),(0.5,0.2),(0.7,0.15),(0.2,0.1),(0.8,0.4)]
for b,g in test_cases:
    inp = torch.tensor([[b/0.9,g/0.5,t/t_max] for t in t_grid], dtype=torch.float32)
    with torch.no_grad():
        pred = mlp(inp).numpy()*N
    sol  = odeint(sir_ode,[N-1,1,0],t_grid,args=(b,g))
    r2   = [round(float(1-np.sum((pred[:,k]-sol[:,k])**2)/
            (np.sum((sol[:,k]-sol[:,k].mean())**2)+1e-12)),4) for k in range(3)]
    print(f"b={b} g={g} | R2={r2}")

print("\nAll models retrained and saved!")
