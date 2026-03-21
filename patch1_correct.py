with open('/teamspace/studios/this_studio/sir_ml_project/dashboard.py', 'r') as f:
    code = f.read()

with open('/teamspace/studios/this_studio/sir_ml_project/dashboard_backup.py', 'w') as f:
    f.write(code)

NEW_PREDICT = '''def predict_epidemic(beta, gamma, model_choice, show_uncertainty):
    b, g  = float(beta), float(gamma)
    inp_t = torch.tensor([[b,g,t/t_max] for t in t_grid], dtype=torch.float32)
    sol   = odeint(sir_ode, [N-1,1,0], t_grid, args=(b,g))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"ML Prediction | beta={b:.2f}  gamma={g:.2f}  R0={b/g:.2f}",
        fontsize=13, fontweight="bold")

    # ── Panel 1: Prediction vs Ground Truth ──────────────────────
    ax = axes[0, 0]
    ax.plot(t_grid, sol[:,0], "b-", lw=2.5, alpha=0.4, label="S ODE truth")
    ax.plot(t_grid, sol[:,1], "r-", lw=2.5, alpha=0.4, label="I ODE truth")
    ax.plot(t_grid, sol[:,2], "g-", lw=2.5, alpha=0.4, label="R ODE truth")

    if model_choice == "MLP":
        with torch.no_grad():
            pred = mlp_model(inp_t).numpy() * N
        lbl = "MLP"
    elif model_choice == "PINN":
        with torch.no_grad():
            pred = pinn_model(inp_t).numpy() * N
        lbl = "PINN"
    else:
        mean_p, std_t = mc_model.predict_with_uncertainty(inp_t, n=100)
        pred   = mean_p.numpy() * N
        std_np = std_t.numpy() * N
        lbl = "MC Dropout"
        if show_uncertainty:
            for k, c in enumerate(["steelblue","tomato","seagreen"]):
                ax.fill_between(t_grid,
                    pred[:,k] - 2*std_np[:,k],
                    pred[:,k] + 2*std_np[:,k],
                    color=c, alpha=0.15)

    for k, (s, lname) in enumerate(zip(["b--","r--","g--"], ["S","I","R"])):
        ax.plot(t_grid, pred[:,k], s, lw=2, label=f"{lname} {lbl}")

    ax.set_xlabel("Time (days)"); ax.set_ylabel("Count")
    ax.set_title("ML Prediction vs ODE Ground Truth")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Panel 2: Residuals ────────────────────────────────────────
    ax_res = axes[0, 1]
    residuals = pred - sol
    ax_res.plot(t_grid, residuals[:,0], "b-", lw=1.8, label="S error")
    ax_res.plot(t_grid, residuals[:,1], "r-", lw=1.8, label="I error")
    ax_res.plot(t_grid, residuals[:,2], "g-", lw=1.8, label="R error")
    ax_res.axhline(0, color="k", lw=1.0, linestyle="--", alpha=0.6)
    ax_res.fill_between(t_grid, residuals[:,1], 0,
                        where=(residuals[:,1] > 0),
                        color="red", alpha=0.10, label="Over-predict I")
    ax_res.fill_between(t_grid, residuals[:,1], 0,
                        where=(residuals[:,1] < 0),
                        color="blue", alpha=0.10, label="Under-predict I")
    ax_res.set_xlabel("Time (days)"); ax_res.set_ylabel("Prediction Error (count)")
    ax_res.set_title("Residuals = Prediction minus ODE Truth")
    ax_res.legend(fontsize=7); ax_res.grid(alpha=0.3)

    # ── Panel 3: MSE bar chart ────────────────────────────────────
    ax_mse = axes[1, 0]
    mse_vals = [float(np.mean((pred[:,k] - sol[:,k])**2)) for k in range(3)]
    mae_vals = [float(np.mean(np.abs(pred[:,k] - sol[:,k]))) for k in range(3)]

    def r2score(p, t):
        ss_res = np.sum((p - t)**2)
        ss_tot = np.sum((t - t.mean())**2) + 1e-12
        return float(1 - ss_res / ss_tot)

    r2_vals = [r2score(pred[:,k], sol[:,k]) for k in range(3)]

    bar_colors = ["steelblue","tomato","seagreen"]
    bar_labels = ["S  Susceptible","I  Infected","R  Recovered"]
    bars = ax_mse.bar(bar_labels, mse_vals,
                      color=bar_colors, alpha=0.80,
                      edgecolor="k", linewidth=0.8)
    for bar, v, r in zip(bars, mse_vals, r2_vals):
        ax_mse.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.03,
                    f"MSE={v:.2f}\\nR2={r:.4f}",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
    ax_mse.set_ylabel("MSE  (count squared)")
    ax_mse.set_title("Mean Squared Error per Compartment")
    ax_mse.grid(alpha=0.3, axis="y")
    ax_mse.tick_params(axis="x", labelsize=8)

    # ── Panel 4: Full metrics table ───────────────────────────────
    ax_tbl = axes[1, 1]
    ax_tbl.axis("off")
    table_rows = [
        ["MSE",
         f"{mse_vals[0]:.2f}", f"{mse_vals[1]:.2f}", f"{mse_vals[2]:.2f}"],
        ["MAE",
         f"{mae_vals[0]:.2f}", f"{mae_vals[1]:.2f}", f"{mae_vals[2]:.2f}"],
        ["RMSE",
         f"{float(np.sqrt(mse_vals[0])):.2f}",
         f"{float(np.sqrt(mse_vals[1])):.2f}",
         f"{float(np.sqrt(mse_vals[2])):.2f}"],
        ["R2",
         f"{r2_vals[0]:.4f}", f"{r2_vals[1]:.4f}", f"{r2_vals[2]:.4f}"],
        ["Max|err|",
         f"{float(np.abs(residuals[:,0]).max()):.1f}",
         f"{float(np.abs(residuals[:,1]).max()):.1f}",
         f"{float(np.abs(residuals[:,2]).max()):.1f}"],
    ]
    col_hdr = ["Metric", "S", "I", "R"]
    tbl = ax_tbl.table(cellText=table_rows, colLabels=col_hdr,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    tbl.scale(1.35, 2.2)
    for j in range(4):
        tbl[0, j].set_facecolor("#1a3a5c")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_rows)+1, 2):
        for j in range(4):
            tbl[i, j].set_facecolor("#eaf4fb")
    ax_tbl.set_title("Quantitative Evaluation Metrics",
                     fontweight="bold", fontsize=11, pad=10)

    plt.tight_layout(pad=2.0)

    R0 = b / g
    stats = (
        f"Evaluation Metrics ({lbl})\\n"
        f"{'─'*36}\\n"
        f"MSE   S={mse_vals[0]:.2f}   I={mse_vals[1]:.2f}   R={mse_vals[2]:.2f}\\n"
        f"MAE   S={mae_vals[0]:.2f}   I={mae_vals[1]:.2f}   R={mae_vals[2]:.2f}\\n"
        f"RMSE  S={float(np.sqrt(mse_vals[0])):.2f}  "
        f"I={float(np.sqrt(mse_vals[1])):.2f}  "
        f"R={float(np.sqrt(mse_vals[2])):.2f}\\n"
        f"R2    S={r2_vals[0]:.4f}  I={r2_vals[1]:.4f}  R={r2_vals[2]:.4f}\\n\\n"
        f"Epidemic Summary\\n"
        f"{'─'*36}\\n"
        f"R0       = {R0:.2f}\\n"
        f"Peak I   = {pred[:,1].max():.0f} people\\n"
        f"Peak day = {t_grid[pred[:,1].argmax()]:.0f}\\n"
        f"Final R  = {pred[:,2][-1]:.0f}  ({pred[:,2][-1]/N*100:.1f}% attacked)\\n"
        f"Status   = {'Epidemic spreads' if R0>1 else 'Dies out R0<1'}"
    )
    return fig_to_img(fig), stats

'''

start_marker = "def predict_epidemic("
end_marker   = "def show_symbolic("

start_idx = code.index(start_marker)
end_idx   = code.index(end_marker)

new_code = code[:start_idx] + NEW_PREDICT + "\n" + code[end_idx:]

with open('/teamspace/studios/this_studio/sir_ml_project/dashboard.py', 'w') as f:
    f.write(new_code)

print("Fix 1 applied to dashboard.py successfully!")
