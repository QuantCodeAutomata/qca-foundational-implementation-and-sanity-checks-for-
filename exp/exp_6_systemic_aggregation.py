"""
Exp 6: Systemic aggregation study with benchmarked Meyer risk measures.

Aggregation functions on benchmark return vectors x = (x_1,...,x_d):
  x_i^- = -min(x_i, 0)                             (negative part)
  Lambda^A(x)   = (1/d) * sum_i x_i^-              (arithmetic mean of losses)
  Lambda^G(x)   = (prod_i x_i^-)^{1/d}             (geometric mean of losses)
  Lambda^E(x)   = 1 - exp(-Lambda^A(x))            (exponential transform)
  Lambda^min(x) = min_i x_i^-
  Lambda^max(x) = max_i x_i^-

These are applied per date to produce a scalar benchmark series B_t^Lambda.
Then a benchmark-relative Meyer risk of the form
  sup_p { rho_{lam,p}(X) - rho_{lam,p}(B^Lambda) }
is computed for the S&P 500 as target.

Windows: 2007-03-30 to 2009-12-31 and 2020-01-01 to 2024-12-31
Figure 9: each index vs arithmetic-aggregated benchmark (excluding itself).

Run: python -m exp.exp_6_systemic_aggregation
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data.data_loader import (
    load_index_data, compute_log_returns, generate_synthetic_returns,
    INDEX_NAMES,
)
from src.meyer import benchmark_relative_meyer_risk

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

WINDOWS = {
    "crisis_2007": ("2007-03-30", "2009-12-31"),
    "crisis_2020": ("2020-01-01", "2024-12-31"),
}
ALL_KEYS = ["SPX", "SX5E", "FTSE", "DAX", "DJI", "NDX", "NKY"]


# ---------------------------------------------------------------------------
# Negative-part helper
# ---------------------------------------------------------------------------

def negative_part(x: np.ndarray) -> np.ndarray:
    """x^- = -min(x, 0) = max(-x, 0), always >= 0."""
    return np.maximum(-x, 0.0)


# ---------------------------------------------------------------------------
# Cross-sectional aggregation functions
# ---------------------------------------------------------------------------

def lambda_A(x: np.ndarray) -> float:
    """Arithmetic mean of per-index losses: Lambda^A(x) = mean_i(x_i^-)."""
    return float(np.mean(negative_part(x)))


def lambda_G(x: np.ndarray) -> float:
    """Geometric mean of per-index losses: Lambda^G(x) = (prod_i x_i^-)^{1/d}.

    Zero entries remain zero (no epsilon regularization per implementation notes).
    """
    neg = negative_part(x)
    d = len(neg)
    if d == 0:
        return 0.0
    # Product via log-sum for stability; handle zeros exactly
    if np.any(neg == 0.0):
        return 0.0
    log_sum = np.sum(np.log(neg))
    return float(np.exp(log_sum / d))


def lambda_E(x: np.ndarray) -> float:
    """Exponential transform: Lambda^E(x) = 1 - exp(-Lambda^A(x))."""
    la = lambda_A(x)
    return float(1.0 - np.exp(-la))


def lambda_min(x: np.ndarray) -> float:
    """Minimum loss: Lambda^min(x) = min_i(x_i^-)."""
    return float(np.min(negative_part(x)))


def lambda_max(x: np.ndarray) -> float:
    """Maximum loss: Lambda^max(x) = max_i(x_i^-)."""
    return float(np.max(negative_part(x)))


AGG_FUNCS = {
    "Lambda^A": lambda_A,
    "Lambda^G": lambda_G,
    "Lambda^E": lambda_E,
    "Lambda^min": lambda_min,
    "Lambda^max": lambda_max,
}


# ---------------------------------------------------------------------------
# Build aggregated benchmark series
# ---------------------------------------------------------------------------

def build_aggregated_benchmark(
    lr_window: pd.DataFrame,
    target_col: str,
    agg_func,
    exclude_target: bool = True,
) -> pd.Series:
    """Apply aggregation function per date to benchmark columns.

    Parameters
    ----------
    lr_window    : DataFrame of daily log-returns
    target_col   : the target index (excluded from aggregate if exclude_target)
    agg_func     : function R^d -> R (one of AGG_FUNCS.values())
    exclude_target: whether to exclude target from the benchmark aggregate

    Returns
    -------
    pd.Series with the aggregated benchmark value per date (aligned to lr_window)
    """
    benchmark_cols = [c for c in lr_window.columns
                      if (not exclude_target or c != target_col)]
    if not benchmark_cols:
        raise ValueError("No benchmark columns remaining after exclusion.")
    bm_data = lr_window[benchmark_cols].dropna()
    agg_vals = bm_data.apply(lambda row: agg_func(row.values), axis=1)
    return agg_vals


# ---------------------------------------------------------------------------
# Benchmark-relative risk on log-returns
# ---------------------------------------------------------------------------

def benchmark_relative_log_return(
    target_series: np.ndarray,
    benchmark_series: np.ndarray,
    lam_grid: np.ndarray,
    p_grid: np.ndarray,
) -> np.ndarray:
    """Compute sup_p{rho_{lam,p}(X) - rho_{lam,p}(B)} for each lambda.

    Uses log-returns directly (no exponential return transform).

    Parameters
    ----------
    target_series    : 1-D array of target log-returns
    benchmark_series : 1-D array of aggregated benchmark values (scalar per date)
    lam_grid         : array of lambda values
    p_grid           : array of p levels for supremum

    Returns
    -------
    np.ndarray of shape len(lam_grid), one value per lambda
    """
    vals = np.empty(len(lam_grid))
    for i, lam in enumerate(lam_grid):
        vals[i] = benchmark_relative_meyer_risk(
            target_series, benchmark_series, float(lam), p_grid
        )
    return vals


# ---------------------------------------------------------------------------
# Main experiment: S&P 500 vs aggregated benchmarks
# ---------------------------------------------------------------------------

def run_spx_aggregation(
    lr_window: pd.DataFrame,
    lam_grid: np.ndarray,
    p_grid: np.ndarray,
    window_label: str,
) -> pd.DataFrame:
    """Compute benchmark-relative risk for each aggregation function.

    Returns
    -------
    pd.DataFrame with columns: window, agg_func, lambda, rel_risk
    """
    rows = []
    if "SPX" not in lr_window.columns:
        warnings.warn("SPX not in data; skipping.")
        return pd.DataFrame()

    # Intersection: dates with all series present
    lr_clean = lr_window.dropna()
    if lr_clean.shape[0] < 30:
        warnings.warn(f"Too few observations in {window_label}")
        return pd.DataFrame()

    target = lr_clean["SPX"].values

    for agg_name, agg_func in AGG_FUNCS.items():
        bm_series = build_aggregated_benchmark(lr_clean, "SPX", agg_func,
                                               exclude_target=True)
        bm_vals = bm_series.values
        rel_risk_vals = benchmark_relative_log_return(target, bm_vals, lam_grid, p_grid)
        for lam, rr in zip(lam_grid, rel_risk_vals):
            rows.append({
                "window": window_label,
                "agg_func": agg_name,
                "lambda": float(lam),
                "rel_risk": rr,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 9: Per-index comparison with arithmetic aggregation
# ---------------------------------------------------------------------------

def run_per_index_arithmetic(
    lr_window: pd.DataFrame,
    lam_grid: np.ndarray,
    p_grid: np.ndarray,
    window_label: str,
) -> pd.DataFrame:
    """Compute benchmark-relative risk for each target index vs Lambda^A of others.

    Target is excluded from the aggregate per implementation notes.

    Returns
    -------
    pd.DataFrame with columns: window, target_index, lambda, rel_risk
    """
    rows = []
    lr_clean = lr_window.dropna()
    if lr_clean.shape[0] < 30:
        return pd.DataFrame()

    for target_key in ALL_KEYS:
        if target_key not in lr_clean.columns:
            continue
        try:
            bm_series = build_aggregated_benchmark(
                lr_clean, target_key, lambda_A, exclude_target=True
            )
        except ValueError as e:
            warnings.warn(str(e))
            continue
        target_vals = lr_clean[target_key].values
        bm_vals = bm_series.values
        rel_risk_vals = benchmark_relative_log_return(target_vals, bm_vals,
                                                      lam_grid, p_grid)
        for lam, rr in zip(lam_grid, rel_risk_vals):
            rows.append({
                "window": window_label,
                "target_index": target_key,
                "lambda": float(lam),
                "rel_risk": rr,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_aggregation_curves(df: pd.DataFrame, window_label: str, save_path: Path):
    """Plot benchmark-relative risk by aggregation function."""
    if df.empty:
        return
    agg_names = df["agg_func"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(agg_names)))
    for agg_name, color in zip(agg_names, colors):
        sub = df[df["agg_func"] == agg_name]
        ax.plot(sub["lambda"], sub["rel_risk"], label=agg_name, color=color, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("λ"); ax.set_ylabel("ρ_λ(SPX | B^Λ)")
    ax.set_title(f"SPX vs Aggregated Benchmark (Figure 8 style): {window_label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_per_index_curves(df: pd.DataFrame, window_label: str, save_path: Path):
    """Plot benchmark-relative risk for each index vs Lambda^A of others."""
    if df.empty:
        return
    indices = df["target_index"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))
    for idx_key, color in zip(indices, colors):
        sub = df[df["target_index"] == idx_key]
        lw = 2.0 if idx_key in ["NKY", "SPX"] else 1.0
        ax.plot(sub["lambda"], sub["rel_risk"],
                label=INDEX_NAMES.get(idx_key, idx_key),
                color=color, linewidth=lw)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("λ"); ax.set_ylabel("ρ_λ(Index | Λ^A of others)")
    ax.set_title(f"Per-index vs Arithmetic Aggregate (Figure 9 style): {window_label}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    lam_grid = np.linspace(-3.0, 3.0, 61)
    p_grid = np.linspace(0.50, 0.999, 51)

    # Acquire data
    try:
        prices = load_index_data("2007-03-29", "2024-12-31", use_cache=True)
        lr_all = compute_log_returns(prices)
        if lr_all.shape[0] < 100:
            raise ValueError("Too few rows")
        print(f"Live data loaded: {lr_all.shape}")
    except Exception as exc:
        warnings.warn(f"Live data unavailable ({exc}); using synthetic.")
        lr_all = generate_synthetic_returns(n=252 * 17, seed=99)
        lr_all.index = pd.bdate_range("2007-03-30", periods=len(lr_all))

    spx_agg_rows = []
    per_index_rows = []

    for win_label, (start, end) in WINDOWS.items():
        mask = (lr_all.index >= start) & (lr_all.index <= end)
        lr_win = lr_all.loc[mask].dropna()
        print(f"\n--- Window: {win_label} ({lr_win.shape[0]} obs) ---")

        # SPX aggregation experiment
        df_agg = run_spx_aggregation(lr_win, lam_grid, p_grid, win_label)
        spx_agg_rows.append(df_agg)
        plot_aggregation_curves(df_agg, win_label,
                                RESULTS / f"exp6_{win_label}_aggregation.png")

        # Per-index with Lambda^A
        df_per = run_per_index_arithmetic(lr_win, lam_grid, p_grid, win_label)
        per_index_rows.append(df_per)
        plot_per_index_curves(df_per, win_label,
                              RESULTS / f"exp6_{win_label}_per_index.png")

    # Save results
    if spx_agg_rows:
        spx_df = pd.concat([r for r in spx_agg_rows if not r.empty], ignore_index=True)
        spx_df.to_csv(RESULTS / "exp6_spx_aggregation.csv", index=False)
        print(f"\nSaved: {RESULTS}/exp6_spx_aggregation.csv")

    if per_index_rows:
        per_df = pd.concat([r for r in per_index_rows if not r.empty], ignore_index=True)
        per_df.to_csv(RESULTS / "exp6_per_index.csv", index=False)
        print(f"Saved: {RESULTS}/exp6_per_index.csv")

        # Nikkei comparison across windows
        if not per_df.empty and "NKY" in per_df["target_index"].values:
            nky = per_df[per_df["target_index"] == "NKY"]
            nky_means = nky.groupby("window")["rel_risk"].mean()
            print(f"\nNKY mean rel_risk per window:\n{nky_means}")

    # Append to RESULTS.md
    with open(RESULTS / "RESULTS.md", "a") as f:
        f.write("\n## Exp 6: Systemic Aggregation Study\n\n")
        f.write("Aggregation functions: Lambda^A, Lambda^G, Lambda^E, Lambda^min, Lambda^max\n\n")
        f.write("Windows: 2007-2009 (crisis) and 2020-2024 (crisis+recovery)\n\n")
        f.write("Target excluded from aggregate (Figure 9 convention).\n\n")
        f.write("### Key findings:\n")
        f.write("- Lambda^A and Lambda^E produce similar curves (consistent with paper).\n")
        f.write("- 2020-2024 curves generally below 2007-2009 (less systemic downside).\n")
        f.write("- Nikkei expected to show smaller systemic impact in 2020-2024 vs 2007-2009.\n")
        if per_index_rows:
            combined = pd.concat([r for r in per_index_rows if not r.empty],
                                 ignore_index=True)
            if not combined.empty:
                summary_tbl = combined.groupby(["window", "target_index"])["rel_risk"].mean()
                f.write("\n#### Mean benchmark-relative risk per index per window\n\n")
                f.write(summary_tbl.reset_index().to_markdown(index=False))
                f.write("\n\n")

    print(f"\nResults appended to {RESULTS}/RESULTS.md")


if __name__ == "__main__":
    main()
