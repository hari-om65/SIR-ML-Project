from pysr import PySRRegressor
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
from scipy.integrate import odeint
import io
from PIL import Image

BASE  = '/teamspace/studios/this_studio/sir_ml_project'
N     = 1000.0
t_max = 160.0
t_grid = np.load(f'{BASE}/data/t_grid.npy').astype(np.float32)

# ── Load all models ───────────────────────────────────────────────
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

class SIR_MCDropout(nn.Module):
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
    def predict_with_uncertainty(self, x, n_samples=100):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x).unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        return preds.mean(dim=0), preds.std(dim=0)

# Load models
mlp_model = SIRMLP()
mlp_model.load_state_dict(torch.load(
    f'{BASE}/data/best_model.pt', map_location='cpu'))
mlp_model.eval()

mc_model = SIR_MCDropout()
mc_model.load_state_dict(torch.load(
    f'{BASE}/data/mc_dropout_model.pt', map_location='cpu'))

pinn_model = SIRMLP()
pinn_model.load_state_dict(torch.load(
    f'{BASE}/data/pinn_model.pt', map_location='cpu'))
pinn_model.eval()

print("All models loaded!")

def sir_ode(y, t, beta, gamma):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

# ── Tab 1: Epidemic Predictor ─────────────────────────────────────
def predict_epidemic(beta, gamma, model_choice, show_uncertainty):
    b, g = float(beta), float(gamma)
    R0   = b / g

    inp_t = torch.tensor(
        [[b, g, t/t_max] for t in t_grid], dtype=torch.float32)

    # ODE ground truth
    sol = odeint(sir_ode, [N-1, 1, 0], t_grid, args=(b, g))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'SIR Epidemic  |  β={b:.2f}  γ={g:.2f}  R₀={R0:.2f}',
                 fontsize=13)

    ax = axes[0]
    ax.plot(t_grid, sol[:,0], 'b-',  lw=2.5, alpha=0.4, label='S (ODE truth)')
    ax.plot(t_grid, sol[:,1], 'r-',  lw=2.5, alpha=0.4, label='I (ODE truth)')
    ax.plot(t_grid, sol[:,2], 'g-',  lw=2.5, alpha=0.4, label='R (ODE truth)')

    if model_choice == 'MLP':
        with torch.no_grad():
            pred = mlp_model(inp_t).numpy() * N
        ax.plot(t_grid, pred[:,0], 'b--', lw=2, label='S (MLP)')
        ax.plot(t_grid, pred[:,1], 'r--', lw=2, label='I (MLP)')
        ax.plot(t_grid, pred[:,2], 'g--', lw=2, label='R (MLP)')

    elif model_choice == 'PINN':
        with torch.no_grad():
            pred = pinn_model(inp_t).numpy() * N
        ax.plot(t_grid, pred[:,0], 'b--', lw=2, label='S (PINN)')
        ax.plot(t_grid, pred[:,1], 'r--', lw=2, label='I (PINN)')
        ax.plot(t_grid, pred[:,2], 'g--', lw=2, label='R (PINN)')

    elif model_choice == 'MC Dropout':
        mean, std = mc_model.predict_with_uncertainty(inp_t, n_samples=100)
        mean = mean.numpy() * N
        std  = std.numpy()  * N
        colors = ['steelblue', 'tomato', 'seagreen']
        names  = ['S', 'I', 'R']
        for k in range(3):
            ax.plot(t_grid, mean[:,k], color=colors[k],
                    lw=2, linestyle='--', label=f'{names[k]} (MC mean)')
            if show_uncertainty:
                ax.fill_between(t_grid,
                    mean[:,k]-2*std[:,k],
                    mean[:,k]+2*std[:,k],
                    color=colors[k], alpha=0.2,
                    label=f'{names[k]} ±2σ')

    ax.set_xlabel('Time (days)'); ax.set_ylabel('Population count')
    ax.set_title('Epidemic curves'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # R0 gauge plot
    ax2 = axes[1]
    thresholds = [1, 2, 3, 4, 5]
    colors_bar = ['green','yellowgreen','orange','orangered','red']
    for i, (lo, hi, c) in enumerate(zip(
            [0,1,2,3,4], [1,2,3,4,6], colors_bar)):
        ax2.barh(0, hi-lo, left=lo, height=0.4,
                 color=c, alpha=0.4, edgecolor='white')
    ax2.axvline(R0, color='black', lw=3, label=f'R₀ = {R0:.2f}')
    ax2.set_xlim(0, 6); ax2.set_ylim(-0.5, 1.5)
    ax2.set_xlabel('R₀ value')
    ax2.set_yticks([])
    ax2.set_title(f'R₀ = {R0:.2f}  →  '
                  f'{"Epidemic spreads" if R0>1 else "Epidemic dies out"}')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='x')

    # Stats
    with torch.no_grad():
        p = mlp_model(inp_t).numpy() * N
    peak_I   = p[:,1].max()
    peak_t   = t_grid[p[:,1].argmax()]
    final_R  = p[:,2][-1]
    herd_thr = (1 - 1/R0) * N if R0 > 1 else 0

    stats = (f"📊 Key Statistics\n"
             f"{'─'*30}\n"
             f"R₀          = {R0:.2f}\n"
             f"Peak I      = {peak_I:.0f} people\n"
             f"Peak time   = day {peak_t:.0f}\n"
             f"Final R     = {final_R:.0f} people\n"
             f"Attack rate = {final_R/N*100:.1f}%\n"
             f"Herd imm.   = {herd_thr:.0f} people need immunity\n"
             f"Status      = {'🔴 Epidemic' if R0>1 else '🟢 Dies out'}")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img, stats

# ── Tab 2: Inverse Problem ────────────────────────────────────────
def solve_inverse(beta_true, gamma_true, noise_level, n_steps):
    b_true = float(beta_true)
    g_true = float(gamma_true)
    noise  = float(noise_level)
    steps  = int(n_steps)

    sol   = odeint(sir_ode, [N-1, 1, 0], t_grid, args=(b_true, g_true))
    rng   = np.random.RandomState(42)
    S_obs = np.clip(sol[:,0] + rng.normal(0,noise,len(t_grid)), 0, N).astype(np.float32)
    I_obs = np.clip(sol[:,1] + rng.normal(0,noise,len(t_grid)), 0, N).astype(np.float32)
    R_obs = np.clip(sol[:,2] + rng.normal(0,noise,len(t_grid)), 0, N).astype(np.float32)

    S_t = torch.tensor(S_obs/N, dtype=torch.float32)
    I_t = torch.tensor(I_obs/N, dtype=torch.float32)
    R_t = torch.tensor(R_obs/N, dtype=torch.float32)
    Y_t = torch.stack([S_t, I_t, R_t], dim=1)

    log_b = nn.Parameter(torch.tensor([np.log(0.3)], dtype=torch.float32))
    log_g = nn.Parameter(torch.tensor([np.log(0.15)], dtype=torch.float32))
    opt   = torch.optim.Adam([log_b, log_g], lr=0.05)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    b_hist, g_hist = [], []
    for _ in range(steps):
        opt.zero_grad()
        b_c  = torch.exp(log_b).clamp(0.05, 1.5)
        g_c  = torch.exp(log_g).clamp(0.01, 0.8)
        t_n  = torch.tensor(t_grid/t_max, dtype=torch.float32)
        inp  = torch.stack([b_c.expand(len(t_grid)),
                            g_c.expand(len(t_grid)), t_n], dim=1)
        loss = torch.mean((mlp_model(inp) - Y_t)**2)
        loss.backward(); opt.step(); sch.step()
        b_hist.append(b_c.item())
        g_hist.append(g_c.item())

    b_est = torch.exp(log_b).item()
    g_est = torch.exp(log_g).item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Inverse Problem — Inferring β and γ from data', fontsize=12)

    inp_t = torch.tensor([[b_est, g_est, t/t_max]
                          for t in t_grid], dtype=torch.float32)
    with torch.no_grad():
        pred = mlp_model(inp_t).numpy() * N

    ax = axes[0]
    ax.scatter(t_grid[::4], S_obs[::4], s=10, color='steelblue', alpha=0.5, label='S observed')
    ax.scatter(t_grid[::4], I_obs[::4], s=10, color='tomato',    alpha=0.5, label='I observed')
    ax.scatter(t_grid[::4], R_obs[::4], s=10, color='seagreen',  alpha=0.5, label='R observed')
    ax.plot(t_grid, pred[:,0], 'b-', lw=2, label='S fitted')
    ax.plot(t_grid, pred[:,1], 'r-', lw=2, label='I fitted')
    ax.plot(t_grid, pred[:,2], 'g-', lw=2, label='R fitted')
    ax.set_xlabel('Time'); ax.set_ylabel('Count')
    ax.set_title('Observed data vs fitted model'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.axhline(b_true, color='steelblue', lw=2.5,
                linestyle='--', alpha=0.7, label=f'True β={b_true}')
    ax2.axhline(g_true, color='tomato', lw=2.5,
                linestyle='--', alpha=0.7, label=f'True γ={g_true}')
    ax2.plot(b_hist, color='steelblue', lw=1.5, label='Est β')
    ax2.plot(g_hist, color='tomato',    lw=1.5, label='Est γ')
    ax2.set_xlabel('Optimisation step'); ax2.set_ylabel('Value')
    ax2.set_title('Parameter convergence'); ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    b_err = abs(b_est-b_true)/b_true*100
    g_err = abs(g_est-g_true)/g_true*100
    result = (f"🔍 Inference Results\n"
              f"{'─'*30}\n"
              f"True β      = {b_true:.3f}\n"
              f"Inferred β  = {b_est:.3f}  (error: {b_err:.1f}%)\n\n"
              f"True γ      = {g_true:.3f}\n"
              f"Inferred γ  = {g_est:.3f}  (error: {g_err:.1f}%)\n\n"
              f"True R₀     = {b_true/g_true:.2f}\n"
              f"Inferred R₀ = {b_est/g_est:.2f}\n\n"
              f"{'✅ Good fit!' if b_err<15 and g_err<15 else '⚠️ Try more steps'}")
    return img, result

# ── Build Gradio UI ───────────────────────────────────────────────
with gr.Blocks(title="SIR Epidemic ML Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🦠 SIR Epidemic ML Dashboard
    **Learning the SIR Model with Machine Learning**
    Predict epidemic curves · Quantify uncertainty · Infer hidden parameters
    """)

    with gr.Tabs():
        # Tab 1
        with gr.Tab("🔬 Epidemic Predictor"):
            gr.Markdown("Adjust β and γ to see how the epidemic changes in real time.")
            with gr.Row():
                with gr.Column(scale=1):
                    beta_sl   = gr.Slider(0.1, 0.9, value=0.3, step=0.01,
                                          label="β — Transmission rate")
                    gamma_sl  = gr.Slider(0.05, 0.5, value=0.1, step=0.01,
                                          label="γ — Recovery rate")
                    model_dd  = gr.Dropdown(
                        ['MLP', 'PINN', 'MC Dropout'],
                        value='MLP', label="Model")
                    unc_cb    = gr.Checkbox(value=True,
                                            label="Show uncertainty bands (MC Dropout only)")
                    run_btn   = gr.Button("▶ Run Prediction", variant="primary")
                with gr.Column(scale=2):
                    plot_out  = gr.Image(label="Epidemic curves")
                    stats_out = gr.Textbox(label="Statistics", lines=9)

            run_btn.click(
                predict_epidemic,
                inputs=[beta_sl, gamma_sl, model_dd, unc_cb],
                outputs=[plot_out, stats_out]
            )

        # Tab 2
        with gr.Tab("🔍 Inverse Problem"):
            gr.Markdown("Given an observed epidemic, infer the unknown β and γ.")
            with gr.Row():
                with gr.Column(scale=1):
                    bt_sl  = gr.Slider(0.1, 0.9, value=0.4, step=0.01,
                                       label="True β (hidden from model)")
                    gt_sl  = gr.Slider(0.05, 0.5, value=0.15, step=0.01,
                                       label="True γ (hidden from model)")
                    ns_sl  = gr.Slider(0, 30, value=10, step=1,
                                       label="Observation noise level")
                    st_sl  = gr.Slider(100, 800, value=400, step=50,
                                       label="Optimisation steps")
                    inv_btn = gr.Button("🔎 Infer Parameters", variant="primary")
                with gr.Column(scale=2):
                    inv_plot = gr.Image(label="Inference result")
                    inv_res  = gr.Textbox(label="Results", lines=12)

            inv_btn.click(
                solve_inverse,
                inputs=[bt_sl, gt_sl, ns_sl, st_sl],
                outputs=[inv_plot, inv_res]
            )

        # Tab 3: About
        with gr.Tab("📖 About"):
            gr.Markdown("""
            ## Project: Learning the SIR Model with ML

            ### What this dashboard shows
            | Feature | Description |
            |---|---|
            | **MLP** | Neural network trained on 400 stochastic epidemic simulations |
            | **PINN** | Same MLP but trained with physics (ODE) loss added |
            | **MC Dropout** | Shows uncertainty bands via Monte Carlo sampling |
            | **Inverse Problem** | Given data, finds β and γ via gradient descent |

            ### The SIR equations (what the ML rediscovered)
```
            dS/dt = -β · S · I / N
            dI/dt =  β · S · I / N  -  γ · I
            dR/dt =  γ · I
```

            ### Key results
            - Best validation loss: **0.000049**
            - Symbolic regression recovered all 3 ODE equations
            - Inverse problem error: **< 10%** for β and γ

            ### Tech stack
            Python · NumPy · SciPy · PyTorch · PySR · Gradio
            """)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
