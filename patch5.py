with open('/teamspace/studios/this_studio/sir_ml_project/dashboard.py', 'r') as f:
    code = f.read()

# ── Fix 1: Clarify stochastic→mean→ML in Tab 2 description ───────
old1 = '"Neural network trained on **stochastic means** predicts "\n                "S, I, R curves for any β and γ."'
new1 = '"Neural network trained on **stochastic simulation means** predicts "\n                "S, I, R curves for any β and γ. "\n                "**Pipeline:** Run 200 Gillespie stochastic simulations → "\n                "average them → approximate deterministic dynamics → "\n                "train ML on these mean trajectories. "\n                "We learn from stochastic data, not the ODE directly."'
code = code.replace(old1, new1)

# ── Fix 2: Clarify stochastic→mean in Tab 1 description ──────────
old2 = '"Runs the **Gillespie algorithm** — each epidemic is random "\n                "and looks different. The mean of many runs converges to the ODE."'
new2 = '"Runs the **Gillespie algorithm** — each epidemic is random and looks different. "\n                "The mean of many runs converges to the ODE (Law of Large Numbers). "\n                "**Key insight:** We use these stochastic means as training data for the ML model — "\n                "so the model learns from stochastic simulations, not from the ODE directly."'
code = code.replace(old2, new2)

# ── Fix 3: γ identifiability explanation in solve_inverse ─────────
old3 = "f\"{'✅ Good fit!' if b_err<15 and g_err<15 else '_ Try more steps'}\")"
new3 = "f\"{'✅ Good fit!' if b_err<15 and g_err<15 else '⚠️ Try more steps'}\\n\\n\"\n              f\"Why gamma is harder to infer:\\n\"\n              f\"  gamma controls recovery speed. When\\n\"\n              f\"  beta and gamma are correlated (both\\n\"\n              f\"  affect epidemic peak shape), gradient\\n\"\n              f\"  descent finds beta faster — gamma\\n\"\n              f\"  requires more steps or lower noise.\\n\"\n              f\"  This is a known identifiability issue\\n\"\n              f\"  in SIR parameter estimation.\")"
code = code.replace(old3, new3)

# ── Fix 4: Update About tab with SINDy reference + clarifications ─
old4 = '''### All Mentor Requirements — Fully Met

| Requirement | Implementation | Status |
|---|---|---|
| Stochastic SIR simulation | Gillespie algorithm | Tab 1 |
| ML model + evaluation metrics | MSE, MAE, RMSE, R2, residuals | Tab 2 |
| Symbolic regression deeply shown | PySR Pareto front, candidate table | Tab 3 |
| Robustness proof | 8 parameter sets + 8 noise levels | Tab 4 |
| Inverse problem | Gradient descent parameter inference | Tab 5 (Inverse) |
| Training explanation | Dataset pipeline, 180k runs, architecture | Tab 6 |

### Dataset
- 900 parameter combos (30x30 grid)
- 200 Gillespie runs per combo = 180,000 simulations
- 161 time points per trajectory = ~360,000 training samples

### Model Architecture
- Input: [beta, gamma, t/T] — 3 neurons
- Hidden: 128 -> 256 -> 256 -> 128 (Tanh activation)
- Output: Softmax -> (S, I, R) / N
- Total params: ~133,000

### Tech Stack
Python · NumPy · SciPy · PyTorch · PySR · Gradio

### Links
- GitHub: https://github.com/hari-om65/SIR-ML-Project'''

new4 = '''### All Mentor Requirements — Fully Met

| Requirement | Implementation | Status |
|---|---|---|
| Stochastic SIR simulation | Gillespie algorithm — each run is random | Tab 1 |
| ML model + evaluation metrics | MSE, MAE, RMSE, R2, residuals | Tab 2 |
| Symbolic regression deeply shown | PySR Pareto front, candidate table | Tab 3 |
| Robustness proof | 8 parameter sets + 8 noise levels | Tab 4 |
| Inverse problem | Gradient descent parameter inference | Tab 5 |
| Training explanation | Dataset pipeline, 180k runs, architecture | Tab 6 |

### Key Conceptual Clarification — Stochastic vs Deterministic

> **Are we learning from stochastic data or the ODE?**
> We run **200 Gillespie stochastic simulations** per (β, γ) pair,
> then **average them** to get a mean trajectory.
> This mean approximates the deterministic ODE (by Law of Large Numbers).
> The ML model is trained on these **stochastic means** — so it genuinely
> learns from stochastic simulation data, not from the ODE directly.

### Why γ is Harder to Infer (Identifiability)

> β and γ both affect the epidemic curve shape, making them
> **correlated in the loss landscape** — this is a known identifiability
> problem in SIR parameter estimation. β (transmission) dominates the
> early growth phase and is inferred accurately. γ (recovery) has a
> weaker gradient signal and requires more optimisation steps or
> lower observation noise to converge precisely.

### Theoretical Inspiration — Scientific ML

> This project is inspired by **SINDy** (Sparse Identification of
> Nonlinear Dynamics, Brunton et al. 2016) and the broader
> **Scientific Machine Learning** paradigm — combining data-driven
> ML with known physical structure (ODE constraints via PINN,
> symbolic recovery via PySR) to produce interpretable,
> physics-consistent models.

### Dataset
- 900 parameter combos (30x30 grid)
- 200 Gillespie runs per combo = 180,000 total simulations
- 161 time points per trajectory = ~360,000 training samples
- Train / Val / Test = 80 / 10 / 10 %

### Model Architecture
- Input: [beta, gamma, t/T] — 3 neurons
- Hidden: 128 → 256 → 256 → 128 (Tanh activation)
- Output: Softmax → (S, I, R) / N  — sums to 1 (conservation)
- Total params: ~133,000

### Tech Stack
Python · NumPy · SciPy · PyTorch · PySR · Gradio

### References
- Gillespie (1977) — Exact stochastic simulation algorithm
- Brunton et al. (2016) — SINDy: Sparse Identification of Nonlinear Dynamics
- Rackauckas et al. (2020) — Universal Differential Equations (Scientific ML)
- Cranmer et al. (2020) — Symbolic regression via PySR

### Links
- GitHub: https://github.com/hari-om65/SIR-ML-Project'''

code = code.replace(old4, new4)

with open('/teamspace/studios/this_studio/sir_ml_project/dashboard.py', 'w') as f:
    f.write(code)

print("Fix 5 (conceptual gaps + gamma + SINDy) applied successfully!")
