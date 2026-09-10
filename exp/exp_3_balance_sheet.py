"""
Exp 3: Balance-sheet simulation with fixed-profile Meyer risk measures.

Model:
  Asset A ~ LogNormal(mu=2.323, sigma=0.200)
  Three alternative liabilities:
    L_lognormal ~ LogNormal(mu=0.837, sigma=0.693)
    L_gamma     ~ Gamma(shape=2.0, scale=1.333)
    L_pareto    ~ Pareto(shape=3.75), support [1, inf), scipy: pareto(b=3.75, scale=1, loc=0)

  Equity X = A - L

Step profile: p1=0.975, p2=0.9, c=1
  H_lam(X) = max(rho_{lam,p1}(X) - c, rho_{lam,p2}(X))

Run: python -m exp.exp_3_balance_sheet
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from pathlib import Path

from src.meyer import fixed_profile_meyer_risk, adjusted_es, adjusted_lvar

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

N_MC = 200_000
SEED = 42
P1 = 0.975
P2 = 0.9
C = 1.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_equity_samples(n: int = N_MC, seed: int = SEED) -> dict:
    """Simulate equity X = A - L for three liability models.

    Asset A ~ LogNormal(mu=2.323, sigma=0.200)
      scipy: lognorm(s=0.200, scale=exp(2.323))

    Liability models:
      lognormal: lognorm(s=0.693, scale=exp(0.837))
      gamma:     gamma(a=2.0, scale=1.333)
      pareto:    pareto(b=3.75, scale=1.0, loc=0)
        -> support x >= 1; corresponds to shape=3.75, location=1.0 in the paper.

    Parameters
    ----------
    n    : number of Monte Carlo draws
    seed : random seed for reproducibility

    Returns
    -------
    dict with keys 'lognormal', 'gamma', 'pareto', each an np.ndarray of shape (n,)
    """
    rng = np.random.RandomState(seed)

    # Common asset sample
    A = st.lognorm(s=0.200, scale=np.exp(2.323)).rvs(n, random_state=rng)

    # Liability models
    L_ln = st.lognorm(s=0.693, scale=np.exp(0.837)).rvs(n, random_state=rng)
    L_gamma = st.gamma(a=2.0, scale=1.333).rvs(n, random_state=rng)
    # scipy.stats.pareto(b, loc, scale): support x >= loc + scale
    # For shape=3.75, location=1.0 (paper): set b=3.75, loc=0, scale=1.0 => x >= 1
    L_pareto = st.pareto(b=3.75, scale=1.0, loc=0).rvs(n, random_state=rng)

    return {
        "lognormal": A - L_ln,
        "gamma": A - L_gamma,
        "pareto": A - L_pareto,
    }


# ---------------------------------------------------------------------------
# Risk curves computation
# ---------------------------------------------------------------------------

def compute_risk_curves(
    samples_dict: dict,
    lam_grid: np.ndarray,
    p1: float = P1,
    p2: float = P2,
    c: float = C,
) -> pd.DataFrame:
    """Compute risk measure values for each liability model and lambda.

    Returns a DataFrame with columns:
      liability_model, lambda, H_lam, adj_ES, adj_LVaR
    """
    rows = []
    for model_name, sample in samples_dict.items():
        adj_es_val = adjusted_es(sample, p1, p2, c)
        adj_lvar_val = adjusted_lvar(sample, p1, p2, c)
        for lam in lam_grid:
            H = fixed_profile_meyer_risk(sample, float(lam), p1, p2, c)
            rows.append({
                "liability_model": model_name,
                "lambda": float(lam),
                "H_lam": H,
                "adj_ES": adj_es_val,
                "adj_LVaR": adj_lvar_val,
            })
    return pd.DataFrame(rows)


def robustness_analysis(
    samples_dict: dict,
    p2_values: list,
    c_values: list,
    p1: float,
    lam_grid: np.ndarray,
) -> pd.DataFrame:
    """Robustness w.r.t. p2 and c: compute H_lam at the mid lambda and edges.

    Parameters
    ----------
    samples_dict : dict of equity samples
    p2_values    : list of alternative p2 levels
    c_values     : list of alternative c values
    p1           : fixed upper level
    lam_grid     : lambda grid

    Returns
    -------
    pd.DataFrame with columns: liability_model, p2, c, lambda, H_lam
    """
    rows = []
    lam_subset = lam_grid[::10]  # sparse grid for speed
    for model_name, sample in samples_dict.items():
        for p2 in p2_values:
            for c in c_values:
                for lam in lam_subset:
                    H = fixed_profile_meyer_risk(sample, float(lam), p1, p2, c)
                    rows.append({
                        "liability_model": model_name,
                        "p2": p2, "c": c,
                        "lambda": float(lam),
                        "H_lam": H,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_equity_distributions(samples_dict: dict, save_path: Path):
    """Violin plots of equity capital distributions."""
    data_long = []
    for model, arr in samples_dict.items():
        # Clip for display
        clipped = np.clip(arr, np.percentile(arr, 0.5), np.percentile(arr, 99.5))
        data_long.extend([{"Liability model": model, "Equity X = A-L": v} for v in clipped])
    df_plot = pd.DataFrame(data_long)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df_plot, x="Liability model", y="Equity X = A-L", ax=ax,
                   palette="muted", inner="quartile")
    ax.set_title("Equity Capital Distributions by Liability Model (Figure 2 style)")
    ax.set_xlabel("Liability distribution")
    ax.set_ylabel("Equity X = A − L")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_risk_curves(results_df: pd.DataFrame, save_path: Path):
    """Plot H_lam vs lambda with comparator lines for each liability model."""
    models = results_df["liability_model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes = [axes]
    colors = {"lognormal": "steelblue", "gamma": "darkorange", "pareto": "darkgreen"}
    for ax, model in zip(axes, models):
        sub = results_df[results_df["liability_model"] == model]
        ax.plot(sub["lambda"], sub["H_lam"],
                label="H_λ (Meyer)", color=colors.get(model, "black"), linewidth=2)
        ax.axhline(sub["adj_ES"].iloc[0], color="tomato", linestyle="--",
                   linewidth=1.5, label="Adj. ES")
        ax.axhline(sub["adj_LVaR"].iloc[0], color="goldenrod", linestyle=":",
                   linewidth=1.5, label="Adj. LVaR")
        ax.set_title(f"Liability: {model}")
        ax.set_xlabel("λ"); ax.set_ylabel("Risk measure")
        ax.legend(fontsize=8)
        ax.axvline(0, color="gray", linewidth=0.5)
    fig.suptitle("Fixed-profile Meyer Risk vs λ (Figure 3 style)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Simulating equity samples (N_MC=200,000)...")
    samples = simulate_equity_samples(N_MC, SEED)

    # Print descriptive stats
    for model, x in samples.items():
        print(f"  {model}: mean={x.mean():.2f}, std={x.std():.2f}, "
              f"5%={np.percentile(x,5):.2f}, 95%={np.percentile(x,95):.2f}")

    lam_grid = np.linspace(-3.0, 3.0, 61)

    print("\nComputing risk curves...")
    results_df = compute_risk_curves(samples, lam_grid)

    # Check: for lam > 0, pareto should have highest H_lam
    high_lam_df = results_df[results_df["lambda"] > 1.5].groupby("liability_model")["H_lam"].mean()
    print("\nMean H_lam for lambda > 1.5:")
    print(high_lam_df)
    pareto_highest = high_lam_df.idxmax() == "pareto"
    print(f"[{'PASS' if pareto_highest else 'NOTE'}] Pareto has highest H_lam for lam>1.5: {pareto_highest}")

    # Robustness analysis
    print("\nRunning robustness analysis...")
    rob_df = robustness_analysis(
        samples,
        p2_values=[0.85, 0.90, 0.95],
        c_values=[0.5, 1.0, 1.5],
        p1=P1,
        lam_grid=lam_grid,
    )
    rob_df.to_csv(RESULTS / "exp3_robustness.csv", index=False)
    print(f"Saved: {RESULTS}/exp3_robustness.csv")

    # Plots
    plot_equity_distributions(samples, RESULTS / "exp3_equity_distributions.png")
    plot_risk_curves(results_df, RESULTS / "exp3_risk_curves.png")

    # Append to RESULTS.md
    summary = results_df.groupby(["liability_model", "lambda"])[["H_lam", "adj_ES", "adj_LVaR"]].first()
    with open(RESULTS / "RESULTS.md", "a") as f:
        f.write("\n## Exp 3: Balance-Sheet Simulation with Fixed-Profile Meyer Risk\n\n")
        f.write(f"**N_MC = {N_MC}, seed = {SEED}**\n\n")
        f.write("### Step profile: p1=0.975, p2=0.9, c=1\n\n")
        f.write(f"Pareto has highest H_lam at lambda>1.5: **{'PASS' if pareto_highest else 'NOTE'}**\n\n")
        f.write("Summary (lambda=0, 1, 2):\n")
        sub = results_df[results_df["lambda"].isin([-2.0, 0.0, 2.0])]
        f.write(sub.to_markdown(index=False))
        f.write("\n\n")
        f.write("### Scipy Pareto parameterization\n")
        f.write("scipy.stats.pareto(b=3.75, scale=1.0, loc=0) has support x >= 1,\n")
        f.write("consistent with shape=3.75 and location=1.0 in the paper.\n\n")

    print(f"\nResults appended to {RESULTS}/RESULTS.md")


if __name__ == "__main__":
    main()
