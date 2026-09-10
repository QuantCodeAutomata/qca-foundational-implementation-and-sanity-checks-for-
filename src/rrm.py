"""
Return Risk Measures (RRM) based on power utility transforms.

Custom implementation — Context7 confirms no standard library provides
the paper's power-utility-based RRM family eta_{lam,p}.

Key formulas:
    p_a(x) = x^a / a    for a != 0   (power utility, x > 0)
    p_0(x) = log(x)     for a == 0

    eta_{lam,p}(X) = p_lam^{-1}(-1 / ES_p(p_lam(X)))

    benchmarked RRM:
    eta_lam(X) = inf_{i in K} sup_p {eta_{-lam,p}(Y_i) / eta_{-lam,p}(X)}

All sample-based functions require strictly positive inputs (gross returns).
"""
import numpy as np
from src.risk_primitives import empirical_es, empirical_es_grid


# ---------------------------------------------------------------------------
# Power utility and inverse
# ---------------------------------------------------------------------------

def power_utility(x: np.ndarray, a: float) -> np.ndarray:
    """Power utility p_a(x).

    p_a(x) = x^a / a   for a != 0
    p_0(x) = log(x)    for a == 0

    Defined on strictly positive reals (x > 0).

    Parameters
    ----------
    x : array-like of strictly positive floats
    a : utility exponent

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("power_utility requires strictly positive inputs (x > 0)")
    if abs(a) < 1e-14:
        return np.log(x)
    return x ** a / a


def power_utility_inverse(y: np.ndarray, a: float) -> np.ndarray:
    """Inverse power utility p_a^{-1}(y).

    For a != 0: p_a^{-1}(y) = (a * y)^{1/a}  (requires a*y > 0)
    For a == 0: p_0^{-1}(y) = exp(y)

    Parameters
    ----------
    y : array-like
    a : utility exponent

    Returns
    -------
    np.ndarray of strictly positive floats (for valid inputs)
    """
    y = np.asarray(y, dtype=float)
    if abs(a) < 1e-14:
        return np.exp(np.clip(y, -500.0, 500.0))
    arg = a * y
    # Clip to avoid negative base for fractional exponent
    arg_safe = np.maximum(arg, 1e-300)
    return arg_safe ** (1.0 / a)


# ---------------------------------------------------------------------------
# eta_{lam,p}: single-level return risk measure
# ---------------------------------------------------------------------------

def eta_lam_p(sample: np.ndarray, lam: float, p: float) -> float:
    """Single-level return risk measure eta_{lam,p}(X).

    eta_{lam,p}(X) = p_lam^{-1}(-1 / ES_p(p_lam(X)))

    Steps:
      1. z = p_lam(X)                  (power-utility transform)
      2. es_z = ES_p(z)                 (expected shortfall of transformed series)
      3. inner = -1 / es_z             (must be in domain of p_lam^{-1})
      4. return p_lam^{-1}(inner)

    Notes on signs:
      - For lam > 0 and positive X: z = X^lam/lam > 0; ES_p(z) < 0 (neg avg of lower tail);
        inner = -1/negative = positive; p_lam^{-1}(positive) OK.
      - For lam < 0 and positive X: z = X^lam/lam < 0; ES_p(z) >= 0 (neg avg of neg values);
        inner = -1/positive < 0; lam*inner = neg*neg > 0; OK.

    Parameters
    ----------
    sample : array-like of strictly positive floats (gross returns), shape (n,)
    lam    : power utility exponent
    p      : tail probability level in [0, 1)

    Returns
    -------
    float : eta_{lam,p}(X)
    """
    sample = np.asarray(sample, dtype=float)
    z = power_utility(sample, lam)
    es_z = empirical_es(z, p)
    if abs(es_z) < 1e-300:
        raise ValueError(f"ES_p is effectively zero at p={p}; cannot invert.")
    inner = -1.0 / es_z
    return float(power_utility_inverse(np.array([inner]), lam)[0])


def eta_lam_p_grid(sample: np.ndarray, lam: float, p_grid: np.ndarray) -> np.ndarray:
    """Vectorised eta_{lam,p} over a probability grid.

    Uses a single sort of the power-utility-transformed sample (prefix sums)
    to compute ES for all p values in O(n log n + m) rather than O(m n log n).

    Parameters
    ----------
    sample : array-like of strictly positive floats, shape (n,)
    lam    : power utility exponent
    p_grid : array-like of levels in [0, 1), shape (m,)

    Returns
    -------
    np.ndarray of shape (m,).  Entries may be NaN for degenerate p values.
    """
    sample = np.asarray(sample, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)
    z = power_utility(sample, lam)                    # single transform

    es_z_arr = empirical_es_grid(z, p_grid)           # single sort, all p

    valid = np.abs(es_z_arr) > 1e-300
    inner = np.where(valid, -1.0 / es_z_arr, np.nan)

    # Vectorised inverse over all valid entries at once
    results = np.full(len(p_grid), np.nan)
    if np.any(valid):
        results[valid] = power_utility_inverse(inner[valid], lam)
    return results


# ---------------------------------------------------------------------------
# Benchmarked Return Risk Measure
# ---------------------------------------------------------------------------

def benchmarked_rrm(
    sample_x: np.ndarray,
    benchmark_samples: list,
    lam: float,
    p_grid: np.ndarray,
) -> float:
    """Empirical benchmarked RRM eta_lam(X).

    eta_lam(X) = inf_{i in K} sup_p {eta_{-lam,p}(Y_i) / eta_{-lam,p}(X)}

    Parameters
    ----------
    sample_x          : array-like of strictly positive floats (target), shape (n,)
    benchmark_samples : list of arrays (each positive), shape (m_i,)
    lam               : risk-aversion parameter (note: uses -lam inside)
    p_grid            : array-like of levels in [0, 1)

    Returns
    -------
    float : eta_lam(X)
    """
    sample_x = np.asarray(sample_x, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)
    neg_lam = -lam

    eta_x = eta_lam_p_grid(sample_x, neg_lam, p_grid)

    sup_per_benchmark = []
    for yi in benchmark_samples:
        yi = np.asarray(yi, dtype=float)
        eta_yi = eta_lam_p_grid(yi, neg_lam, p_grid)
        # Compute ratio where both are finite and eta_x > 0
        valid = np.isfinite(eta_x) & np.isfinite(eta_yi) & (np.abs(eta_x) > 1e-300)
        if not np.any(valid):
            sup_per_benchmark.append(np.nan)
            continue
        ratios = np.where(valid, eta_yi / eta_x, np.nan)
        sup_per_benchmark.append(float(np.nanmax(ratios)))

    valid_sups = [s for s in sup_per_benchmark if np.isfinite(s)]
    if not valid_sups:
        return np.nan
    return float(min(valid_sups))
