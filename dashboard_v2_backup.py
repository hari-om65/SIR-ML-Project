from pysr import PySRRegressor
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from scipy.integrate import odeint
import sys, io
from PIL import Image

sys.path.insert(0, "/teamspace/studios/this_studio/sir_ml_project/simulation")
from gillespie import run_gillespie_sir, mean_sir_trajectory

BASE   = "/teamspace/studios/this_studio/sir_ml_project"
N      = 1000.0
t_max  = 160.0
t_grid = np.load(f"{BASE}/data/t_grid.npy").astype(np.float32)

class SIRMLP(nn.Module):
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

class SIR_MCDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,128), nn.Tanh(), nn.Dropout(p),
            nn.Linear(128,256), nn.Tanh(), nn.Dropout(p),
            nn.Linear(256,256), nn.Tanh(), nn.Dropout(p),
            nn.Linear(256,128), nn.Tanh(), nn.Dropout(p),
            nn.Linear(128,3),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)
    def predict_with_uncertainty(self, x, n=100):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n):
                preds.append(self.forward(x).unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        return preds.mean(0), preds.std(0)

mlp_model = SIRMLP()
mlp_model.load_state_dict(torch.load(f"{BASE}/data/best_model.pt", map_location="cpu"))
mlp_model.eval()
mc_model = SIR_MCDropout()
mc_model.load_state_dict(torch.load(f"{BASE}/data/mc_dropout_model.pt", map_location="cpu"))
pinn_model = SIRMLP()
pinn_model.load_state_dict(torch.load(f"{BASE}/data/pinn_model.pt", map_location="cpu"))
pinn_model.eval()

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def sir_ode(y, t, beta, gamma):
    S,I,R = y
    return [-beta*S*I/N, beta*S*I/N-gamma*I, gamma*I]

def run_stochastic(beta, gamma, n_runs, show_mean):
    b, g = float(beta), float(gamma)
    n    = int(n_runs)
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    fig.suptitle(f"Stochastic SIR | beta={b:.2f} gamma={g:.2f} R0={b/g:.2f}", fontsize=12)
    ax = axes[0]
    for i in range(min(n, 20)):
        t_ev, S_ev, I_ev, R_ev = run_gillespie_sir(b, g, int(N), seed=i)
        ax.plot(t_ev, I_ev, color="tomato", alpha=0.25, lw=0.8)
    ax.set_title(f"{min(n,20)} stochastic runs - each is different!")
    ax.set_xlabel("Time"); ax.set_ylabel("Infected count")
    ax2 = axes[1]
    t_g, S_m, I_m, R_m = mean_sir_trajectory(b, g, int(N), n_runs=n, seed=0)
    sol = odeint(sir_ode, [N-1,1,0], t_g, args=(b,g))
    ax2.plot(t_g, sol[:,0],"b-", lw=2.5, alpha=0.5, label="S ODE")
    ax2.plot(t_g, sol[:,1],"r-", lw=2.5, alpha=0.5, label="I ODE")
    ax2.plot(t_g, sol[:,2],"g-", lw=2.5, alpha=0.5, label="R ODE")
    if show_mean:
        ax2.plot(t_g, S_m,"b--", lw=2, label="S stochastic mean")
        ax2.plot(t_g, I_m,"r--", lw=2, label="I stochastic mean")
        ax2.plot(t_g, R_m,"g--", lw=2, label="R stochastic mean")
    ax2.set_title("Mean converges to ODE as N increases")
    ax2.set_xlabel("Time"); ax2.set_ylabel("Count"); ax2.legend(fontsize=7)
    plt.tight_layout()
    stats = (f"Simulation Stats\n" + "-"*32 + "\n"
             f"beta={b:.2f}  gamma={g:.2f}  R0={b/g:.2f}\n"
             f"Runs={n}  Population={int(N)}\n\n"
             f"Stochastic mean peak I = {I_m.max():.1f}\n"
             f"Deterministic peak I   = {sol[:,1].max():.1f}\n"
             f"Difference             = {abs(I_m.max()-sol[:,1].max()):.1f}")
    return fig_to_img(fig), stats

def predict_epidemic(beta, gamma, model_choice, show_uncertainty):
    b, g  = float(beta), float(gamma)
    inp_t = torch.tensor([[b,g,t/t_max] for t in t_grid], dtype=torch.float32)
    sol   = odeint(sir_ode, [N-1,1,0], t_grid, args=(b,g))
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    fig.suptitle(f"ML Prediction | beta={b:.2f} gamma={g:.2f} R0={b/g:.2f}", fontsize=12)
    ax = axes[0]
    ax.plot(t_grid, sol[:,0],"b-", lw=2.5, alpha=0.4, label="S ODE truth")
    ax.plot(t_grid, sol[:,1],"r-", lw=2.5, alpha=0.4, label="I ODE truth")
    ax.plot(t_grid, sol[:,2],"g-", lw=2.5, alpha=0.4, label="R ODE truth")
    if model_choice == "MLP":
        with torch.no_grad(): pred = mlp_model(inp_t).numpy()*N
        ax.plot(t_grid,pred[:,0],"b--",lw=2,label="S MLP")
        ax.plot(t_grid,pred[:,1],"r--",lw=2,label="I MLP")
        ax.plot(t_grid,pred[:,2],"g--",lw=2,label="R MLP")
    elif model_choice == "PINN":
        with torch.no_grad(): pred = pinn_model(inp_t).numpy()*N
        ax.plot(t_grid,pred[:,0],"b--",lw=2,label="S PINN")
        ax.plot(t_grid,pred[:,1],"r--",lw=2,label="I PINN")
        ax.plot(t_grid,pred[:,2],"g--",lw=2,label="R PINN")
    elif model_choice == "MC Dropout":
        mean, std = mc_model.predict_with_uncertainty(inp_t, n=100)
        mean = mean.numpy()*N; std = std.numpy()*N
        for k,(c,lbl) in enumerate(zip(["steelblue","tomato","seagreen"],["S","I","R"])):
            ax.plot(t_grid,mean[:,k],color=c,lw=2,linestyle="--",label=f"{lbl} MC mean")
            if show_uncertainty:
                ax.fill_between(t_grid,mean[:,k]-2*std[:,k],mean[:,k]+2*std[:,k],color=c,alpha=0.15)
    ax.set_xlabel("Time"); ax.set_ylabel("Count"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax2 = axes[1]
    R0 = b/g
    for lo,hi,c in zip([0,1,2,3,4],[1,2,3,4,6],["green","yellowgreen","orange","orangered","red"]):
        ax2.barh(0,hi-lo,left=lo,height=0.4,color=c,alpha=0.4,edgecolor="white")
    ax2.axvline(R0,color="black",lw=3,label=f"R0={R0:.2f}")
    ax2.set_xlim(0,6); ax2.set_ylim(-0.5,1.5); ax2.set_yticks([])
    ax2.set_xlabel("R0"); ax2.legend(fontsize=10); ax2.grid(alpha=0.3,axis="x")
    plt.tight_layout()
    with torch.no_grad(): p = mlp_model(inp_t).numpy()*N
    stats = (f"R0={R0:.2f}\nPeak I={p[:,1].max():.0f}\n"
             f"Peak time=day {t_grid[p[:,1].argmax()]:.0f}\n"
             f"Final R={p[:,2][-1]:.0f}\nAttack rate={p[:,2][-1]/N*100:.1f}%")
    return fig_to_img(fig), stats
