import sys
sys.path.insert(0, '/root/sir_ml_project/simulation')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gillespie import run_gillespie_sir, mean_sir_trajectory

N = 1000
beta = 0.3
gamma = 0.1

# Single stochastic run
t, S, I, R = run_gillespie_sir(beta, gamma, N, seed=42)
print(f"Single run: {len(t)} events | peak I = {I.max()} | final R = {R[-1]}")

# Mean over 200 runs
t_grid, S_mean, I_mean, R_mean = mean_sir_trajectory(beta, gamma, N, n_runs=200, seed=0)
print(f"Mean trajectory: peak mean I = {I_mean.max():.1f} at t = {t_grid[I_mean.argmax()]:.1f}")
print(f"Conservation check S+I+R = {(S_mean+I_mean+R_mean).mean():.1f} (should be {N})")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t, S, label='S', color='steelblue')
plt.plot(t, I, label='I', color='tomato')
plt.plot(t, R, label='R', color='seagreen')
plt.title('Single stochastic run')
plt.xlabel('Time'); plt.ylabel('Count'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_grid, S_mean, label='S mean', color='steelblue')
plt.plot(t_grid, I_mean, label='I mean', color='tomato')
plt.plot(t_grid, R_mean, label='R mean', color='seagreen')
plt.title('Mean over 200 runs')
plt.xlabel('Time'); plt.ylabel('Count'); plt.legend()

plt.tight_layout()
plt.savefig('/root/sir_ml_project/data/gillespie_test.png', dpi=120)
print("Plot saved to data/gillespie_test.png")
print("ALL DONE - Step 4 complete!")
