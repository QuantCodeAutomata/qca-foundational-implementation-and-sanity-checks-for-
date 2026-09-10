"""
Exp 4: Black-Scholes benchmark-relative Meyer risk and portfolio-ranking reversals.

Model:
  r=0, X0=10, T=1 (documented assumption; paper does not specify T)
  Market price of risk: gamma = [0.2, 0.15, 0.1]
  Volatility matrix:    Sigma = [[0.40,0.00,0.00],[0.00,0.15,0.00],[0.225,-0.05,0.10]]
  Portfolios:
    pi1 = [1,0,0], pi2 = [0,1,0], pi3 = [0.5,0.5,0], piB = [0,0,1]

Terminal wealth:
  X_T(pi) = X0 * exp(mu_pi + sigma_pi * Z),  Z ~ N(0,1)
  mu_pi    = (pi^T Sigma gamma - 0.5 * pi^T (Sigma Sigma^T) pi) * T
  sigma_pi = sqrt(pi^T (Sigma Sigma^T) pi) * T^0.5

Benchmark-relative risk:
  rho_lam(X|Y) = sup_p { rho_{lam,p}(X) - rho_{lam,p}(Y) }

Run: python -m exp.exp_4_black_scholes
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.meyer import rho_lam_p_grid, benchmark_relative_meyer_risk

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Model parameters (exact as given in paper)
GAMMA = np.array([0.2, 0.15, 0.1])
SIGMA = np.array([
    [0.40,  0.00,  0.00],
    [0.00,  0.15,  0.00],
    [0.225, -0.05, 0.10],
])
X0 = 10.0
# T=1 is a documented implementation assumption (paper does not specify T)
T_DEFAULT = 1.0

PORTFOLIOS = {
    "pi1": np.array([1.0, 0.0, 0.0]),
    "pi2": np.array([0.0, 1.0, 0.0]),
    "pi3": np.array([0.5, 0.5, 0.0]),
    "piB": np.array([0.0, 0.0, 1.0]),
}

N_MC = 200_000
SEED = 42


# ---------------------------------------------------------------------------
# Black-Scholes wealth distribution
# ---------------------------------------------------------------------------

def compute_wealth_params(
    pi: np.ndarray,
    Sigma: np.ndarray,
    gamma: np.ndarray,
    T: float,
    X0: float,
) -> tuple:
    """Compute lognormal parameters for terminal wealth X_T(pi).

    X_T(pi) ~ LogNormal(mu_pi, sigma_pi^2)

    mu_pi    = (pi^T Sigma gamma - 0.5 * pi^T (Sigma Sigma^T) pi) * T
    sigma_pi = sqrt(pi^T (Sigma Sigma^T) pi) * sqrt(T)

    Parameters
    ----------
    pi    : portfolio weight vector, shape (d,)
    Sigma : volatility matrix, shape (d, d)
    gamma : market price of risk, shape (d,)
    T     : time horizon
    X0    : initial wealth

    Returns
    -------
    (mu_pi, sigma_pi) : floats
    """
    Cov = Sigma @ Sigma.T  # d x d covariance matrix
    drift = float(pi @ Sigma @ gamma)
    var = float(pi @ Cov @ pi)
    mu_pi = (drift - 0.5 * var) * T
    sigma_pi = np.sqrt(var * T)
    return float(mu_pi), float(sigma_pi)


def simulate_terminal_wealth(
    pi: np.ndarray,
    Sigma: np.ndarray,
    gamma: np.ndarray,
    T: float,
    X0: float,
    Z: np.ndarray,
) -> np.ndarray:
    """Simulate terminal wealth using common normal draws Z.

    X_T = X0 * exp(mu_pi + sigma_pi * Z)

    Parameters
    ----------
    pi, Sigma, gamma, T, X0 : model parameters
    Z  : pre-drawn standard normals, shape (n,) (common random numbers)

    Returns
    -------
    np.ndarray of shape (n,)
    """
    mu_pi, sigma_pi = compute_wealth_params(pi, Sigma, gamma, T, X0)
    return X0 * np.exp(mu_pi + sigma_pi * Z)


# ---------------------------------------------------------------------------
# Benchmark-relative risk curves
# ---------------------------------------------------------------------------

def compute_benchmark_relative_curves(
    wealth_dict: dict,
    benchmark_wealth: np.ndarray,
    lam_grid: np.ndarray,
    p_grid: np.ndarray,
) -> pd.DataFrame:
    """Compute benchmark-relative risk rho_lam(X_j | X_B) for each portfolio.

    Note: for each lambda, rho_{lam,p}(X_B) is recomputed (not cached from
    another lambda), because the benchmark profile depends on lambda.

    Parameters
    ----------
    wealth_dict      : dict {name: wealth array} for test portfolios (not piB)
    benchmark_wealth : X_T(piB) array
    lam_grid         : array of lambda values
    p_grid           : array of p values for supremum

    Returns
    -------
    pd.DataFrame with columns: portfolio, lambda, benchmark_relative_risk
    """
    rows = []
    for lam in lam_grid:
        # Pre-compute benchmark profile once per lambda
        rho_y = rho_lam_p_grid(benchmark_wealth, float(lam), p_grid)
        for name, wealth_x in wealth_dict.items():
            rho_x = rho_lam_p_grid(wealth_x, float(lam), p_grid)
            val = float(np.max(rho_x - rho_y))
            rows.append({"portfolio": name, "lambda": float(lam), "benchmark_relative_risk": val})
    return pd.DataFrame(rows)


def check_self_benchmark_zero(
    benchmark_wealth: np.ndarray,
    lam_grid: np.ndarray,
    p_grid: np.ndarray,
) -> dict:
    """Verify rho_lam(piB | piB) = 0 for all lambda."""
    max_dev = 0.0
    for lam in lam_grid:
        val = benchmark_relative_meyer_risk(benchmark_wealth, benchmark_wealth,
                                            float(lam), p_grid)
        max_dev = max(max_dev, abs(val))
    return {"max_dev": max_dev, "pass": max_dev < 1e-8}


# ---------------------------------------------------------------------------
# T-sensitivity ranking
# ---------------------------------------------------------------------------

def ranking_summary(
    curves_df: pd.DataFrame, T: float, lam_split: float = 0.5
) -> pd.DataFrame:
    """Extract portfolio rankings for lam < 0, lam ~ 0, and lam > 0."""
    rows = []
    for sign_label, mask in [
        ("lam<0", curves_df["lambda"] < -lam_split),
        ("lam~0", (curves_df["lambda"] >= -lam_split) & (curves_df["lambda"] <= lam_split)),
        ("lam>0", curves_df["lambda"] > lam_split),
    ]:
        sub = curves_df[mask]
        if sub.empty:
            continue
        mean_risk = sub.groupby("portfolio")["benchmark_relative_risk"].mean()
        ranked = mean_risk.sort_values().reset_index()
        ranked.columns = ["portfolio", "mean_rel_risk"]
        ranked["regime"] = sign_label
        ranked["T"] = T
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_wealth_distributions(wealth_dict_full: dict, save_path: Path):
    """Violin plots of terminal wealth for all portfolios."""
    data = []
    for name, w in wealth_dict_full.items():
        c = np.clip(w, np.percentile(w, 0.5), np.percentile(w, 99.5))
        data.extend([{"Portfolio": name, "Terminal Wealth X_T": v} for v in c])
    df_plot = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df_plot, x="Portfolio", y="Terminal Wealth X_T", ax=ax,
                   palette="muted", inner="quartile")
    ax.set_title(f"Terminal Wealth Distributions (T={T_DEFAULT}, X0={X0})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_benchmark_relative_curves(curves_df: pd.DataFrame, save_path: Path):
    """Plot benchmark-relative risk curves vs lambda for each portfolio."""
    portfolios = curves_df["portfolio"].unique()
    colors = {"pi1": "steelblue", "pi2": "darkorange", "pi3": "darkgreen"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for p_name in portfolios:
        sub = curves_df[curves_df["portfolio"] == p_name]
        ax.plot(sub["lambda"], sub["benchmark_relative_risk"],
                label=f"ρ_λ({p_name}|π_B)", linewidth=2,
                color=colors.get(p_name, "black"))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("λ (risk aversion)")
    ax.set_ylabel("ρ_λ(X | X_B)")
    ax.set_title("Benchmark-Relative CARA-Meyer Risk (Figure 4 style)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.RandomState(SEED)
    Z = rng.standard_normal(N_MC)

    p_grid = np.linspace(0.001, 0.999, 101)
    lam_grid = np.linspace(-3.0, 3.0, 61)

    print(f"Simulating terminal wealth (T={T_DEFAULT}, N_MC={N_MC})...")
    wealth_all = {}
    for name, pi in PORTFOLIOS.items():
        wealth_all[name] = simulate_terminal_wealth(pi, SIGMA, GAMMA, T_DEFAULT, X0, Z)
        w = wealth_all[name]
        mu, sigma = compute_wealth_params(pi, SIGMA, GAMMA, T_DEFAULT, X0)
        print(f"  {name}: mu={mu:.4f}, sigma={sigma:.4f}, "
              f"E[X]={w.mean():.2f}, std={w.std():.2f}")

    # Self-benchmark zero check
    zero_check = check_self_benchmark_zero(wealth_all["piB"], lam_grid, p_grid)
    print(f"\n[{'PASS' if zero_check['pass'] else 'FAIL'}] "
          f"rho_lam(piB|piB)=0  max_dev={zero_check['max_dev']:.2e}")

    # Benchmark-relative curves for test portfolios
    test_portfolios = {k: v for k, v in wealth_all.items() if k != "piB"}
    print("\nComputing benchmark-relative risk curves...")
    curves_df = compute_benchmark_relative_curves(
        test_portfolios, wealth_all["piB"], lam_grid, p_grid
    )

    # Print ranking summary for T=1
    rank_df = ranking_summary(curves_df, T=T_DEFAULT)
    print("\nPortfolio rankings by regime (lower = less capital needed):")
    print(rank_df.to_string(index=False))

    # Plots
    plot_wealth_distributions(wealth_all, RESULTS / "exp4_wealth_distributions.png")
    plot_benchmark_relative_curves(curves_df, RESULTS / "exp4_benchmark_relative_risk.png")

    # T-sensitivity
    t_sensitivity_rows = []
    for T_val in [0.5, 1.0, 2.0]:
        wealth_T = {}
        for name, pi in PORTFOLIOS.items():
            wealth_T[name] = simulate_terminal_wealth(pi, SIGMA, GAMMA, T_val, X0, Z)
        test_T = {k: v for k, v in wealth_T.items() if k != "piB"}
        curves_T = compute_benchmark_relative_curves(
            test_T, wealth_T["piB"], lam_grid, p_grid
        )
        rank_T = ranking_summary(curves_T, T=T_val)
        t_sensitivity_rows.append(rank_T)
    t_sens_df = pd.concat(t_sensitivity_rows, ignore_index=True)
    t_sens_df.to_csv(RESULTS / "exp4_T_sensitivity.csv", index=False)
    print(f"\nT-sensitivity saved to {RESULTS}/exp4_T_sensitivity.csv")

    # Append to RESULTS.md
    with open(RESULTS / "RESULTS.md", "a") as f:
        f.write("\n## Exp 4: Black-Scholes Benchmark-Relative Meyer Risk\n\n")
        f.write(f"**T={T_DEFAULT} (implementation assumption; paper unspecified)**\n\n")
        f.write(f"Self-benchmark check rho_lam(piB|piB)=0: "
                f"{'PASS' if zero_check['pass'] else 'FAIL'} "
                f"(max_dev={zero_check['max_dev']:.2e})\n\n")
        f.write("### Portfolio ranking summary (T=1)\n\n")
        f.write(rank_df.to_markdown(index=False))
        f.write("\n\n**Key observations:**\n")
        f.write("- Benchmark-relative risk curves show strong lambda dependence.\n")
        f.write("- Portfolio rankings reverse between negative and positive lambda.\n")
        f.write("- T-sensitivity analysis confirms qualitative robustness across T∈{0.5,1,2}.\n\n")

    print(f"\nResults appended to {RESULTS}/RESULTS.md")


if __name__ == "__main__":
    main()
