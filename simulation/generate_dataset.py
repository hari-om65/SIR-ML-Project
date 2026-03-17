import sys
sys.path.insert(0, '/teamspace/studios/this_studio/sir_ml_project/simulation')
import numpy as np
from gillespie import mean_sir_trajectory
from tqdm import tqdm

BASE = '/teamspace/studios/this_studio/sir_ml_project'

N        = 1000
n_runs   = 150
t_max    = 160
n_points = 161

# Grid of beta and gamma values — 20x20 = 400 parameter combinations
beta_values  = np.linspace(0.1, 0.9, 20)
gamma_values = np.linspace(0.05, 0.5, 20)

all_beta, all_gamma, all_S, all_I, all_R = [], [], [], [], []

total = len(beta_values) * len(gamma_values)
print(f"Generating dataset: {total} parameter combinations...")

with tqdm(total=total) as pbar:
    for beta in beta_values:
        for gamma in gamma_values:
            t_grid, S_mean, I_mean, R_mean = mean_sir_trajectory(
                beta, gamma, N,
                n_runs=n_runs,
                t_max=t_max,
                n_points=n_points,
                seed=42
            )
            all_beta.append(beta)
            all_gamma.append(gamma)
            all_S.append(S_mean)
            all_I.append(I_mean)
            all_R.append(R_mean)
            pbar.update(1)

# Save everything
np.save(f'{BASE}/data/t_grid.npy',    t_grid)
np.save(f'{BASE}/data/all_beta.npy',  np.array(all_beta))
np.save(f'{BASE}/data/all_gamma.npy', np.array(all_gamma))
np.save(f'{BASE}/data/all_S.npy',     np.array(all_S))
np.save(f'{BASE}/data/all_I.npy',     np.array(all_I))
np.save(f'{BASE}/data/all_R.npy',     np.array(all_R))

print(f"\nDataset saved!")
print(f"Shape — S: {np.array(all_S).shape}  means (400 param points x 161 time steps)")
print(f"Beta range:  {min(all_beta):.2f} to {max(all_beta):.2f}")
print(f"Gamma range: {min(all_gamma):.2f} to {max(all_gamma):.2f}")
print("Step 5 complete!")
