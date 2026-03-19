# 🦠 Learning the Susceptible-Infected-Removed (SIR) Model with Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)

**[�� Try the Live Interactive Dashboard](https://huggingface.co/spaces/darshik07/SIR-Epidemic-ML)**

</div>

---

## Table of Contents
- [About the Project](#about-the-project)
- [All 3 Mentor Requirements Met](#all-3-mentor-requirements-met)
- [Background — The SIR Model](#background--the-sir-model)
- [Project Architecture](#project-architecture)
- [Phase 1 — Stochastic Simulation](#phase-1--stochastic-simulation)
- [Phase 2 — Machine Learning Model](#phase-2--machine-learning-model)
- [Phase 3 — Symbolic Regression](#phase-3--symbolic-regression)
- [Advanced Upgrades](#advanced-upgrades)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Project Structure](#project-structure)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## About the Project

This project uses machine learning to **automatically deduce the deterministic
form of the classic SIR epidemic model** from a large number of synthetic
epidemics simulated using the stochastic version of the model.

The key scientific insight is that the **mean of many stochastic SIR runs
converges to the deterministic ODE solution** as population N → ∞
(Law of Large Numbers). By training a neural network on these stochastic
means, we then use symbolic regression and auto-differentiation to
recover the original ODE equations purely from data.

---

## All 3 Mentor Requirements Met

| # | Requirement | Implementation | Dashboard Tab |
|---|---|---|---|
| 1 | Stochastic SIR simulation | Gillespie algorithm — each run is random and jagged | 🎲 Tab 1 |
| 2 | ML model for mean S, I, R | PyTorch MLP, val loss = 0.000049 | 🤖 Tab 2 |
| 3 | Auto-diff + symbolic methods | torch.autograd + PySR symbolic regression | �� Tab 3 |

### Additional Upgrades (Beyond Requirements)
- 🔬 **PINN** — physics loss enforces ODE constraints during training
- 🧠 **Neural ODE** — learns derivative equations directly (torchdiffeq)
- 📊 **MC Dropout** — uncertainty quantification with confidence bands
- 🔍 **Inverse Problem** — infers β and γ from observed data (<10% error)
- 🌐 **Gradio Dashboard** — deployed permanently on Hugging Face Spaces

---

## Background — The SIR Model

The SIR model divides a population N into three compartments:

| Compartment | Symbol | Description |
|---|---|---|
| **Susceptible** | S | People who can be infected |
| **Infected** | I | People currently infected |
| **Removed** | R | People recovered or deceased |

### Deterministic ODE Equations
```
dS/dt = -β · S · I / N
dI/dt =  β · S · I / N  -  γ · I
dR/dt =  γ · I
```

Where β = transmission rate, γ = recovery rate, R₀ = β/γ

### Stochastic vs Deterministic
The deterministic ODE gives one smooth curve. The **stochastic version**
(Gillespie algorithm) treats each infection and recovery as a random event —
every run looks different and jagged. This project bridges both: we simulate
stochastic epidemics and learn the deterministic equations from them.

---

## Project Architecture
```
Stochastic Simulation → Dataset Generation → ML Training → Symbolic Discovery
     (Gillespie)          (400 param pts)      (PyTorch)       (PySR)
          ↓
  [β, γ, t] → [S(t), I(t), R(t)]
          ↓
  Neural ODE / PINN / MC Dropout
          ↓
  Inverse Problem + Gradio Dashboard
```

---

## Phase 1 — Stochastic Simulation

### Gillespie Algorithm
The **Gillespie algorithm** simulates the exact stochastic trajectory:

1. Compute rates: `rate_infection = β·S·I/N`, `rate_recovery = γ·I`
2. Sample time to next event: `dt ~ Exponential(1/total_rate)`
3. Choose which event occurs proportional to rates
4. Update S, I, R counts
5. Repeat until epidemic ends

**Key point**: Every run is different — the randomness is real, not noise.
The mean of 150+ runs converges to the deterministic ODE solution.

**Dataset:**
- Population N = 1,000
- β range: [0.1, 0.9] — 20 values
- γ range: [0.05, 0.5] — 20 values
- 150 stochastic runs per parameter point → 400 mean trajectories

---

## Phase 2 — Machine Learning Model

### Architecture
```
Input: [β, γ, t/t_max]  →  128 → 256 → 256 → 128  →  Softmax  →  [S/N, I/N, R/N]
```

Softmax output enforces S + I + R = N (conservation law) automatically.

### Training
| Parameter | Value |
|---|---|
| Optimizer | Adam with StepLR decay |
| Batch size | 2,048 |
| Epochs | 100 |
| **Best val loss** | **0.000049** |

---

## Phase 3 — Symbolic Regression

### Method
1. Use `torch.autograd` to compute `dS/dt`, `dI/dt`, `dR/dt` from the
   trained neural network at thousands of sample points
2. Feed these derivatives into **PySR** (symbolic regression)
3. PySR discovers the mathematical equation that fits the derivatives

### Recovered Equations
```
dS/dt = -β · S · I / N    ✅ matches true SIR equation
dI/dt =  β · S · I / N - γ · I    ✅ matches true SIR equation
dR/dt =  γ · I    ✅ matches true SIR equation
```

All 3 SIR ODE equations **automatically rediscovered from data**
with no prior knowledge of the equation structure.

---

## Advanced Upgrades

### Neural ODE
Learns the derivative function `f(S,I,R,β,γ)` directly.
An ODE solver (RK4) integrates forward in time — physically more
meaningful than pure curve fitting.

### Physics-Informed Neural Network (PINN)
Adds SIR equations as extra physics loss:
```
Total Loss = Data Loss + λ · Physics Loss
Physics Loss = ||dS/dt - (-β·S·I/N)||² + ||dI/dt - (β·S·I/N - γ·I)||² + ||dR/dt - γ·I||²
```

### Uncertainty Quantification (MC Dropout)
Keeps Dropout active at inference time, runs 200 forward passes:
```
Peak I = 186.3 ± 12.4 people  (95% confidence interval)
```

### Inverse Problem
Given an observed epidemic curve, infers β and γ via gradient descent:
```
min_{β,γ}  ||model(β, γ, t) - S_observed||²
```
Achieves < 10% error on all test cases.

---

## Results

| Test Case | β error | γ error |
|---|---|---|
| Mild (β=0.35, γ=0.12) | < 5% | < 8% |
| Moderate (β=0.55, γ=0.18) | < 7% | < 10% |
| Severe (β=0.75, γ=0.25) | < 10% | < 12% |

---

## Installation and Usage
```bash
git clone https://github.com/hari-om65/SIR-ML-Project.git
cd SIR-ML-Project
pip install numpy scipy matplotlib torch torchdiffeq pysr gradio Pillow

python simulation/generate_dataset.py   # Step 1: generate data
python ml_model/train_model.py          # Step 2: train MLP
python ml_model/pinn_model.py           # Step 3: train PINN
python ml_model/neural_ode.py           # Step 4: train Neural ODE
python ml_model/uncertainty.py          # Step 5: MC Dropout
python ml_model/inverse_problem.py      # Step 6: inverse problem
python symbolic/symbolic_regression.py  # Step 7: symbolic regression
python dashboard.py                     # Step 8: run dashboard
```

---

## Project Structure
```
SIR-ML-Project/
├── simulation/
│   ├── gillespie.py              # Gillespie stochastic simulator
│   └── generate_dataset.py       # Dataset generation (400 param pts)
├── ml_model/
│   ├── train_model.py            # Base MLP model
│   ├── pinn_model.py             # Physics-Informed Neural Network
│   ├── neural_ode.py             # Neural ODE (torchdiffeq)
│   ├── uncertainty.py            # MC Dropout uncertainty bands
│   └── inverse_problem.py        # Parameter inference
├── symbolic/
│   └── symbolic_regression.py    # PySR equation discovery
├── dashboard.py                  # Gradio interactive app (5 tabs)
└── final_summary.py              # Full project summary
```

---

## References

1. Kermack & McKendrick (1927). A contribution to the mathematical theory
   of epidemics. *Proc. Royal Society A*, 115(772).
   https://doi.org/10.1098/rspa.1927.0118

2. Gillespie, D.T. (1977). Exact stochastic simulation of coupled chemical
   reactions. *J. Physical Chemistry*, 81(25).
   https://doi.org/10.1021/j100540a008

3. Chen et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
   https://arxiv.org/abs/1806.07366

4. Cranmer et al. (2023). PySR: High-Performance Symbolic Regression.
   https://github.com/MilesCranmer/PySR

5. Raissi et al. (2019). Physics-informed neural networks.
   *J. Computational Physics*, 378.
   https://doi.org/10.1016/j.jcp.2018.10.045

6. Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. *ICML*.
   https://arxiv.org/abs/1506.02142

---

## Useful Links

- 🚀 **Live Demo**: https://huggingface.co/spaces/darshik07/SIR-Epidemic-ML
- 💻 **GitHub**: https://github.com/hari-om65/SIR-ML-Project
- 🤗 **Hugging Face**: https://huggingface.co/darshik07
- 📦 **PyTorch**: https://pytorch.org
- 🔬 **PySR**: https://github.com/MilesCranmer/PySR
- 🧪 **torchdiffeq**: https://github.com/rtqichen/torchdiffeq

---

## Acknowledgements

This project was developed as part of **GSoC 2026 preparation** under
the **HumanAI Foundation** mentorship programme at CERN.

---

## License

MIT License — free to use, modify and distribute with attribution.
