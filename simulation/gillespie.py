import numpy as np

def run_gillespie_sir(beta, gamma, N, I0=1, t_max=160, seed=None):
    if seed is not None:
        np.random.seed(seed)
    S = N - I0
    I = I0
    R = 0
    t = 0.0
    t_events = [t]; S_traj = [S]; I_traj = [I]; R_traj = [R]
    while t < t_max and I > 0:
        rate_infection = beta * S * I / N
        rate_recovery  = gamma * I
        total_rate     = rate_infection + rate_recovery
        if total_rate == 0:
            break
        dt = np.random.exponential(1.0 / total_rate)
        t  = t + dt
        if t > t_max:
            break
        if np.random.uniform() < rate_infection / total_rate:
            S -= 1; I += 1
        else:
            I -= 1; R += 1
        t_events.append(t); S_traj.append(S)
        I_traj.append(I); R_traj.append(R)
    return np.array(t_events), np.array(S_traj), np.array(I_traj), np.array(R_traj)

def interpolate_trajectory(t_events, S_traj, I_traj, R_traj, t_grid):
    S_out = np.interp(t_grid, t_events, S_traj)
    I_out = np.interp(t_grid, t_events, I_traj)
    R_out = np.interp(t_grid, t_events, R_traj)
    return S_out, I_out, R_out

def mean_sir_trajectory(beta, gamma, N, n_runs=200, t_max=160, n_points=161, seed=None):
    t_grid = np.linspace(0, t_max, n_points)
    S_runs = np.zeros((n_runs, n_points))
    I_runs = np.zeros((n_runs, n_points))
    R_runs = np.zeros((n_runs, n_points))
    for i in range(n_runs):
        run_seed = None if seed is None else seed + i
        t_ev, S_ev, I_ev, R_ev = run_gillespie_sir(
            beta, gamma, N, t_max=t_max, seed=run_seed)
        S_runs[i], I_runs[i], R_runs[i] = interpolate_trajectory(
            t_ev, S_ev, I_ev, R_ev, t_grid)
    return t_grid, S_runs.mean(axis=0), I_runs.mean(axis=0), R_runs.mean(axis=0)
