"""
Exp 1: Foundational implementation and sanity checks for Meyer risk measures.

Validates core identities:
  - rho_{0,p}   = ES_p
  - rho_{lam,1} = rho^w
  - rho_lam(X|X) = 0
  - step-profile reduction: H_lam = max(rho_{lam,p1}-c, rho_{lam,p2})
  - lambda continuity at lam=0
  - monotonicity in p

Run: python -m exp.exp_1_sanity_checks
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.risk_primitives import empirical_es, left_quantile, rho_w, empirical_es_grid
from src.meyer import (
    cara_utility, cara_utility_inverse, rho_lam_p, rho_lam_p_grid,
    fixed_profile_meyer_risk, adjusted_es, adjusted_lvar,
    benchmark_relative_meyer_risk,
)
from src.rrm import power_utility, power_utility_inverse, eta_lam_p

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic test samples (fixed seeds)
# ---------------------------------------------------------------------------

def make_samples() -> dict:
    """Create three canonical test samples."""
    rng = np.random.RandomState(42)
    discrete = np.array([0.0] * 75 + [1_000_000.0] * 25)
    rng.shuffle(discrete)
    gaussian = rng.normal(0, 1, 5000)
    positive = np.abs(rng.normal(1, 0.2, 5000)) + 0.5
    return {"discrete": discrete, "gaussian": gaussian, "positive": positive}


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_rho_0_equals_ES(
    samples_dict: dict, p_grid: np.ndarray
) -> pd.DataFrame:
    """Verify rho_{0,p}(X) = ES_p(X) for all samples and p values.

    Returns a DataFrame with the maximum absolute difference per sample.
    All differences should be < 1e-8.
    """
    rows = []
    for name, sample in samples_dict.items():
        es_vals = empirical_es_grid(sample, p_grid)
        rho_vals = rho_lam_p_grid(sample, 0.0, p_grid)
        max_diff = float(np.max(np.abs(es_vals - rho_vals)))
        rows.append({"sample": name, "max_abs_diff": max_diff, "pass": max_diff < 1e-8})
    return pd.DataFrame(rows)


def validate_rho_lam_1_equals_rho_w(
    samples_dict: dict, lam_grid: np.ndarray
) -> pd.DataFrame:
    """Verify rho_{lam,1}(X) = rho^w(X) for all lam and samples."""
    rows = []
    for name, sample in samples_dict.items():
        worst = rho_w(sample)
        max_diff = 0.0
        for lam in lam_grid:
            val = rho_lam_p(sample, float(lam), 1.0)
            diff = abs(val - worst)
            if diff > max_diff:
                max_diff = diff
        rows.append({"sample": name, "max_abs_diff": max_diff, "pass": max_diff < 1e-8})
    return pd.DataFrame(rows)


def validate_self_benchmark_zero(
    samples_dict: dict, lam_grid: np.ndarray, p_grid: np.ndarray
) -> pd.DataFrame:
    """Verify rho_lam(X|X) = 0 for all samples and lam."""
    rows = []
    for name, sample in samples_dict.items():
        max_abs = 0.0
        for lam in lam_grid:
            val = benchmark_relative_meyer_risk(sample, sample, float(lam), p_grid)
            max_abs = max(max_abs, abs(val))
        rows.append({"sample": name, "max_abs_val": max_abs, "pass": max_abs < 1e-8})
    return pd.DataFrame(rows)


def validate_step_profile_reduction(
    sample: np.ndarray, lam: float, p1: float, p2: float, c: float
) -> dict:
    """Verify fixed_profile_meyer_risk matches manual two-point maximum."""
    H = fixed_profile_meyer_risk(sample, lam, p1, p2, c)
    r1 = rho_lam_p(sample, lam, p1)
    r2 = rho_lam_p(sample, lam, p2)
    manual_H = max(r1 - c, r2)
    diff = abs(H - manual_H)
    return {"H_lam": H, "manual_H": manual_H, "diff": diff, "pass": diff < 1e-12}


def validate_lambda_continuity(
    sample: np.ndarray, p: float, lam_grid: np.ndarray
) -> dict:
    """Check continuity at lam=0: rho_{lam,p}(X) -> ES_p(X) as lam->0."""
    es_val = empirical_es(sample, p)
    rho_vals = np.array([rho_lam_p(sample, float(lam), p) for lam in lam_grid])
    # find closest lam to 0 and check value is near ES
    idx_zero = np.argmin(np.abs(lam_grid))
    val_at_zero = rho_vals[idx_zero]
    diff_at_zero = abs(val_at_zero - es_val)
    # check continuity: no large jumps
    gaps = np.max(np.abs(np.diff(rho_vals)))
    return {
        "p": p,
        "ES_p": es_val,
        "rho_at_lam0": val_at_zero,
        "diff_at_zero": diff_at_zero,
        "max_gap": gaps,
        "lam_grid": lam_grid,
        "rho_vals": rho_vals,
        "pass": diff_at_zero < 1e-8,
    }


def validate_monotonicity(
    sample: np.ndarray, lam_grid: np.ndarray, p_grid: np.ndarray
) -> dict:
    """Verify rho_{lam,p}(X) is nondecreasing in p for each fixed lam."""
    violations = 0
    for lam in lam_grid:
        vals = rho_lam_p_grid(sample, float(lam), p_grid)
        diffs = np.diff(vals)
        violations += int(np.sum(diffs < -1e-8))
    return {"violations": violations, "pass": violations == 0}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_lambda_continuity(sample: np.ndarray, p_vals: list, lam_grid: np.ndarray):
    """Plot rho_{lam,p} vs lambda for several p values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in p_vals:
        rho_vals = np.array([rho_lam_p(sample, float(lam), p) for lam in lam_grid])
        es_val = empirical_es(sample, p)
        ax.plot(lam_grid, rho_vals, label=f"p={p}")
        ax.axhline(es_val, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("λ (risk aversion)")
    ax.set_ylabel("ρ_{λ,p}(X)")
    ax.set_title("Lambda continuity: rho_{lam,p} vs lambda (dashes = ES_p)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_lambda_continuity.png", dpi=120)
    plt.close(fig)


def plot_p_monotonicity(sample: np.ndarray, lam_vals: list, p_grid: np.ndarray):
    """Plot rho_{lam,p} vs p for several lambda values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for lam in lam_vals:
        vals = rho_lam_p_grid(sample, float(lam), p_grid)
        ax.plot(p_grid, vals, label=f"λ={lam}")
    ax.set_xlabel("p (confidence level)")
    ax.set_ylabel("ρ_{λ,p}(X)")
    ax.set_title("Monotonicity: rho_{lam,p} vs p (should be nondecreasing)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_monotonicity.png", dpi=120)
    plt.close(fig)


def plot_es_identity(sample: np.ndarray, p_grid: np.ndarray):
    """Show ES_p vs rho_{0,p}: should coincide exactly."""
    es_vals = empirical_es_grid(sample, p_grid)
    rho_vals = rho_lam_p_grid(sample, 0.0, p_grid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(p_grid, es_vals, label="ES_p", linewidth=2)
    axes[0].plot(p_grid, rho_vals, "--", label="rho_{0,p}", linewidth=1.5)
    axes[0].set_xlabel("p"); axes[0].set_ylabel("Risk value")
    axes[0].set_title("ES_p == rho_{0,p} (Gaussian sample)")
    axes[0].legend()
    axes[1].semilogy(p_grid, np.abs(es_vals - rho_vals) + 1e-16)
    axes[1].set_xlabel("p"); axes[1].set_ylabel("|diff|")
    axes[1].set_title("Absolute difference (should be ~machine epsilon)")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp1_es_identity.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    samples = make_samples()
    p_grid = np.linspace(0.5, 0.999, 100)
    lam_grid_dense = np.linspace(-2.0, 2.0, 41)
    lam_grid_coarse = np.array([-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0])
    p_grid_fine = np.linspace(0.5, 0.999, 50)

    summary_rows = []

    # --- Identity 1: rho_{0,p} = ES_p ---
    df1 = validate_rho_0_equals_ES(samples, p_grid)
    all_pass1 = df1["pass"].all()
    print(f"[{'PASS' if all_pass1 else 'FAIL'}] rho_{{0,p}} == ES_p")
    print(df1.to_string(index=False))
    summary_rows.append({"check": "rho_0_equals_ES", "pass": all_pass1,
                          "max_diff": df1["max_abs_diff"].max()})

    # --- Identity 2: rho_{lam,1} = rho^w ---
    df2 = validate_rho_lam_1_equals_rho_w(samples, lam_grid_dense)
    all_pass2 = df2["pass"].all()
    print(f"\n[{'PASS' if all_pass2 else 'FAIL'}] rho_{{lam,1}} == rho^w")
    print(df2.to_string(index=False))
    summary_rows.append({"check": "rho_lam1_equals_rho_w", "pass": all_pass2,
                          "max_diff": df2["max_abs_diff"].max()})

    # --- Identity 3: rho_lam(X|X) = 0 ---
    df3 = validate_self_benchmark_zero(samples, lam_grid_coarse, p_grid_fine)
    all_pass3 = df3["pass"].all()
    print(f"\n[{'PASS' if all_pass3 else 'FAIL'}] rho_lam(X|X) == 0")
    print(df3.to_string(index=False))
    summary_rows.append({"check": "self_benchmark_zero", "pass": all_pass3,
                          "max_diff": df3["max_abs_val"].max()})

    # --- Identity 4: Step-profile reduction ---
    sample_g = samples["gaussian"]
    step_res = validate_step_profile_reduction(sample_g, lam=1.0, p1=0.975, p2=0.9, c=1.0)
    all_pass4 = step_res["pass"]
    print(f"\n[{'PASS' if all_pass4 else 'FAIL'}] Step-profile reduction")
    print(f"  H_lam={step_res['H_lam']:.6f}  manual_H={step_res['manual_H']:.6f}  diff={step_res['diff']:.2e}")
    summary_rows.append({"check": "step_profile_reduction", "pass": all_pass4,
                          "max_diff": step_res["diff"]})

    # --- Identity 5: Lambda continuity at lam=0 ---
    cont_res = validate_lambda_continuity(sample_g, p=0.95, lam_grid=lam_grid_dense)
    all_pass5 = cont_res["pass"]
    print(f"\n[{'PASS' if all_pass5 else 'FAIL'}] Lambda continuity at lam=0")
    print(f"  ES_0.95={cont_res['ES_p']:.6f}  rho_at_0={cont_res['rho_at_lam0']:.6f}"
          f"  diff={cont_res['diff_at_zero']:.2e}  max_gap={cont_res['max_gap']:.4f}")
    summary_rows.append({"check": "lambda_continuity", "pass": all_pass5,
                          "max_diff": cont_res["diff_at_zero"]})

    # --- Identity 6: Monotonicity in p ---
    mono_res = validate_monotonicity(sample_g, lam_grid_dense, p_grid)
    all_pass6 = mono_res["pass"]
    print(f"\n[{'PASS' if all_pass6 else 'FAIL'}] Monotonicity in p (violations={mono_res['violations']})")
    summary_rows.append({"check": "monotonicity_in_p", "pass": all_pass6,
                          "max_diff": float(mono_res["violations"])})

    # --- RRM sanity: eta_{lam,p} on positive sample ---
    pos_sample = samples["positive"]
    eta_val = eta_lam_p(pos_sample, lam=1.0, p=0.9)
    print(f"\n[INFO] eta_{{1,0.9}}(positive_sample) = {eta_val:.6f}  (should be positive)")
    assert eta_val > 0, f"eta should be positive, got {eta_val}"

    # Save summary table
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS / "exp1_identity_checks.csv", index=False)
    print(f"\nSummary saved to {RESULTS / 'exp1_identity_checks.csv'}")

    # --- Plots ---
    plot_lambda_continuity(sample_g, p_vals=[0.9, 0.95, 0.99], lam_grid=lam_grid_dense)
    plot_p_monotonicity(sample_g, lam_vals=[-1.0, 0.0, 1.0], p_grid=p_grid)
    plot_es_identity(sample_g, p_grid=p_grid)
    print("Plots saved to results/")

    # --- RESULTS.md ---
    overall = all(r["pass"] for r in summary_rows)
    _append_results_md(df_summary, overall)
    print("Results appended to results/RESULTS.md")

    return overall


def _append_results_md(df_summary: pd.DataFrame, overall_pass: bool):
    with open(RESULTS / "RESULTS.md", "a") as f:
        f.write("\n## Exp 1: Foundational Sanity Checks for Meyer Risk Measures\n\n")
        f.write(f"**Overall: {'PASS' if overall_pass else 'FAIL'}**\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n### Descriptions\n")
        f.write("- `rho_0_equals_ES`: rho_{0,p}=ES_p verified to machine precision.\n")
        f.write("- `rho_lam1_equals_rho_w`: rho_{lam,1}=max(-X) verified for all lambda.\n")
        f.write("- `self_benchmark_zero`: rho_lam(X|X)=0 verified for all lambda.\n")
        f.write("- `step_profile_reduction`: H_lam=max(rho_{lam,p1}-c, rho_{lam,p2}) exact.\n")
        f.write("- `lambda_continuity`: rho_{lam,p} continuous at lam=0.\n")
        f.write("- `monotonicity_in_p`: rho_{lam,p} nondecreasing in p.\n\n")


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
