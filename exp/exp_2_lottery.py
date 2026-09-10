"""
Exp 2: Motivating lottery example and ES-crossing reproduction.

Two lotteries:
  X1: payoff 0 w.p. 0.75 and 1,000,000 w.p. 0.25
  X2: payoff -100,000 w.p. 0.25 and 400,000 w.p. 0.75

The paper claims the ES curves cross at approximately p* ≈ 0.69.
This experiment:
  1. Derives ES curves analytically from the left-quantile convention.
  2. Numerically finds the crossing level.
  3. Plots the CARA expected-utility ratio f(c) to show preference reversals.

Run: python -m exp.exp_2_lottery
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.risk_primitives import empirical_es

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Analytical ES for the two discrete lotteries
# ---------------------------------------------------------------------------

def exact_es_x1(p_grid: np.ndarray) -> np.ndarray:
    """Analytical ES_p(X1) for X1 = {0 w.p. 0.75, 1e6 w.p. 0.25}.

    Left quantile: q_{X1}(r) = 0 for r in [0, 0.75), 1e6 for r in [0.75, 1].

    ES_p(X1) = -(1/(1-p)) * integral_0^{1-p} q_{X1}(r) dr

    For p >= 0.25 (i.e., 1-p <= 0.75): integral = 0, ES_p(X1) = 0.
    For p < 0.25  (i.e., 1-p > 0.75):
        integral = 0.75*0 + (1-p-0.75)*1e6 = (0.25-p)*1e6
        ES_p(X1) = -(0.25-p)*1e6/(1-p)   [negative, representing a risk]
    """
    p_grid = np.asarray(p_grid, dtype=float)
    es = np.where(
        p_grid >= 0.25,
        0.0,
        -(0.25 - p_grid) * 1e6 / (1.0 - p_grid),
    )
    return es


def exact_es_x2(p_grid: np.ndarray) -> np.ndarray:
    """Analytical ES_p(X2) for X2 = {-1e5 w.p. 0.25, 4e5 w.p. 0.75}.

    Left quantile: q_{X2}(r) = -1e5 for r in [0, 0.25), 4e5 for r in [0.25, 1].

    For p >= 0.75 (1-p <= 0.25):
        integral_0^{1-p} q_{X2}(r) dr = (1-p)*(-1e5)
        ES_p(X2) = -(-1e5*(1-p))/(1-p) = 1e5   [positive = we gain in expected tail]
    For p < 0.75 (1-p > 0.25):
        integral = 0.25*(-1e5) + (0.75-p)*4e5 = -25000 + (0.75-p)*4e5
        ES_p(X2) = -(-25000 + (0.75-p)*4e5)/(1-p) = (25000 - (0.75-p)*4e5)/(1-p)
    """
    p_grid = np.asarray(p_grid, dtype=float)
    # p >= 0.75
    high_p = 1e5 * np.ones_like(p_grid)
    # p < 0.75
    low_p = (25000.0 - (0.75 - p_grid) * 4e5) / (1.0 - p_grid)
    es = np.where(p_grid >= 0.75, high_p, low_p)
    return es


def find_es_crossing(
    p_grid: np.ndarray, es1: np.ndarray, es2: np.ndarray
) -> float:
    """Find p* where ES_p(X1) = ES_p(X2) by linear interpolation.

    The paper claims p* ≈ 0.69.

    Parameters
    ----------
    p_grid : sorted probability levels
    es1    : ES_p(X1)
    es2    : ES_p(X2)

    Returns
    -------
    float : estimated crossing level
    """
    diff = es1 - es2
    # Find sign change
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        raise ValueError("No ES crossing found in the provided p_grid")
    # Use the first crossing
    idx = sign_changes[0]
    # Linear interpolation
    p0, p1_ = p_grid[idx], p_grid[idx + 1]
    d0, d1_ = diff[idx], diff[idx + 1]
    crossing = p0 - d0 * (p1_ - p0) / (d1_ - d0)
    return float(crossing)


# ---------------------------------------------------------------------------
# Expected utility ratio (scaled payoffs to avoid overflow)
# ---------------------------------------------------------------------------

def expected_utility_ratio_scaled(c_grid: np.ndarray) -> np.ndarray:
    """CARA expected-utility ratio f(c) = E[u_c(X1)] / E[u_c(X2)] in scaled units.

    To avoid overflow, we work with scaled payoffs:
        X1_scaled = {0 w.p. 0.75, 1 w.p. 0.25}           (divided by 1e6)
        X2_scaled = {-0.1 w.p. 0.25, 0.4 w.p. 0.75}      (divided by 1e6)

    The CARA utility u_c(x) = -c * exp(-c * x) is applied with c interpreted
    as risk aversion per unit of scaled payoff (i.e., c_scaled = c_original * 1e6).

    f(c) = E[u_c(X1_scaled)] / E[u_c(X2_scaled)]

    Note: f(c) < 1 means X1 preferred over X2 (under the convention that u is
    increasing and comparing expected utilities).

    Parameters
    ----------
    c_grid : array-like of c_scaled values (> 0)

    Returns
    -------
    np.ndarray of ratios f(c)
    """
    c_grid = np.asarray(c_grid, dtype=float)
    ratios = np.empty(len(c_grid))
    for i, c in enumerate(c_grid):
        eu_x1 = 0.75 * (-c * np.exp(0.0)) + 0.25 * (-c * np.exp(-c * 1.0))
        eu_x2 = 0.25 * (-c * np.exp(-c * (-0.1))) + 0.75 * (-c * np.exp(-c * 0.4))
        if abs(eu_x2) < 1e-300:
            ratios[i] = np.nan
        else:
            ratios[i] = eu_x1 / eu_x2
    return ratios


# ---------------------------------------------------------------------------
# Monte-Carlo cross-validation
# ---------------------------------------------------------------------------

def mc_es_validation(n: int = 1_000_000) -> dict:
    """Validate analytical ES against large MC samples."""
    rng = np.random.RandomState(123)
    # X1
    x1_mc = rng.choice([0.0, 1e6], size=n, p=[0.75, 0.25])
    # X2
    x2_mc = rng.choice([-1e5, 4e5], size=n, p=[0.25, 0.75])

    p_check = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    results = {}
    for p in p_check:
        analytic1 = float(exact_es_x1(np.array([p]))[0])
        empirical1 = empirical_es(x1_mc, p)
        analytic2 = float(exact_es_x2(np.array([p]))[0])
        empirical2 = empirical_es(x2_mc, p)
        results[p] = {
            "X1_analytic": analytic1, "X1_empirical": empirical1,
            "X2_analytic": analytic2, "X2_empirical": empirical2,
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_utility_ratio(c_grid: np.ndarray, ratios: np.ndarray):
    """Plot f(c) = E[u_c(X1)] / E[u_c(X2)] vs c."""
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(ratios)
    ax.plot(c_grid[valid], ratios[valid], linewidth=2, color="steelblue")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="f(c)=1 (indifference)")
    ax.set_xlabel("c (scaled risk aversion, c_original × 10⁻⁶)")
    ax.set_ylabel("f(c) = E[u_c(X1)] / E[u_c(X2)]")
    ax.set_title("Expected Utility Ratio: f(c)<1 means X1 preferred")
    ax.legend()
    ax.set_ylim([0, 3])
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_utility_ratio.png", dpi=120)
    plt.close(fig)
    print("Saved: results/exp2_utility_ratio.png")


def plot_es_curves(p_grid: np.ndarray, es1: np.ndarray, es2: np.ndarray, crossing: float):
    """Plot ES_p(X1) and ES_p(X2) curves with the crossing level marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_grid, es1, label="ES_p(X1)", linewidth=2, color="steelblue")
    ax.plot(p_grid, es2, label="ES_p(X2)", linewidth=2, color="darkorange")
    ax.axvline(crossing, color="gray", linestyle="--",
               label=f"Crossing p*≈{crossing:.4f}")
    ax.set_xlabel("p (confidence level)")
    ax.set_ylabel("ES_p(X) [in original units]")
    ax.set_title("Expected Shortfall Curves for X1 and X2")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([-3e5, 2e5])
    fig.tight_layout()
    fig.savefig(RESULTS / "exp2_es_curves.png", dpi=120)
    plt.close(fig)
    print("Saved: results/exp2_es_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p_grid = np.linspace(0.001, 0.9999, 5000)
    es1 = exact_es_x1(p_grid)
    es2 = exact_es_x2(p_grid)

    # Find crossing
    crossing = find_es_crossing(p_grid, es1, es2)
    print(f"ES crossing level p* = {crossing:.6f}")
    assert 0.67 <= crossing <= 0.71, (
        f"Expected crossing near 0.69, got {crossing:.4f}"
    )
    print("[PASS] ES crossing is within [0.67, 0.71]")

    # Expected utility ratio
    c_grid = np.linspace(0.01, 5.0, 500)
    ratios = expected_utility_ratio_scaled(c_grid)

    # Find indifference c*
    diff_from_one = ratios - 1.0
    sign_changes = np.where(np.diff(np.sign(diff_from_one[np.isfinite(diff_from_one)])))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        print(f"Preference reversal at c_scaled ≈ {c_grid[idx]:.4f}")
    else:
        print("No preference reversal detected in the c range; check c_grid.")

    # Monte-Carlo cross-validation
    print("\nMonte-Carlo ES cross-validation (n=1,000,000):")
    mc_res = mc_es_validation()
    for p, d in mc_res.items():
        diff1 = abs(d["X1_analytic"] - d["X1_empirical"])
        diff2 = abs(d["X2_analytic"] - d["X2_empirical"])
        print(f"  p={p:.2f}  |X1 diff|={diff1:.2f}  |X2 diff|={diff2:.2f}")

    # Plots
    plot_utility_ratio(c_grid, ratios)
    plot_es_curves(p_grid, es1, es2, crossing)

    # Append to RESULTS.md
    with open(RESULTS / "RESULTS.md", "a") as f:
        f.write("\n## Exp 2: Lottery Example — ES Crossing\n\n")
        f.write(f"**ES crossing level p* = {crossing:.6f}** (claimed: ≈0.69)\n\n")
        f.write(f"- Tolerance check: {0.67:.2f} <= {crossing:.4f} <= {0.71:.2f}: "
                f"{'PASS' if 0.67 <= crossing <= 0.71 else 'FAIL'}\n")
        f.write("- X1 = {0 w.p. 0.75, 1,000,000 w.p. 0.25}\n")
        f.write("- X2 = {-100,000 w.p. 0.25, 400,000 w.p. 0.75}\n")
        f.write("- ES curves cross at approximately 69% confidence.\n")
        f.write("- For high risk aversion, X1 dominates; for low risk aversion, X2 is preferred.\n\n")

    print(f"\nResults appended to {RESULTS}/RESULTS.md")
    return crossing


if __name__ == "__main__":
    main()
