"""
Risk primitives: empirical left quantile and Expected Shortfall.

Convention (matching the paper):
  q_X(p) = inf{x : P(X <= x) >= p}   (left/lower quantile)
  ES_p(X) = -(1/(1-p)) * integral_0^{1-p} q_X(r) dr,  for p < 1
  ES_1(X) = rho^w(X) = ess_sup(-X) = max(-X) on finite samples

All functions operate on 1-D numpy arrays representing i.i.d. sample realisations.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Empirical left quantile
# ---------------------------------------------------------------------------

def left_quantile(sample: np.ndarray, p: float) -> float:
    """Empirical left quantile q_X(p) = inf{x : F_X(x) >= p}.

    Uses a left-continuous empirical CDF convention: F_n(x) = #{x_i <= x}/n.
    For p in (0, 1], this is the ceil(p*n)-th order statistic (1-indexed).

    Parameters
    ----------
    sample : array-like of floats, shape (n,)
    p      : probability level in [0, 1]

    Returns
    -------
    float : q_X(p)
    """
    sample = np.asarray(sample, dtype=float)
    n = len(sample)
    if n == 0:
        raise ValueError("sample must be non-empty")
    x_sorted = np.sort(sample)
    if p <= 0.0:
        return float(x_sorted[0])
    if p >= 1.0:
        return float(x_sorted[-1])
    # F_n(x_{(k)}) = k/n; we need first k such that k/n >= p -> k = ceil(p*n)
    idx = int(np.ceil(p * n)) - 1  # 0-indexed
    idx = min(max(idx, 0), n - 1)
    return float(x_sorted[idx])


def quantile_grid(sample: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
    """Vectorised left quantile over a grid of probability levels.

    Parameters
    ----------
    sample : array-like, shape (n,)
    p_grid : array-like of probability levels in [0, 1], shape (m,)

    Returns
    -------
    np.ndarray of shape (m,) with q_X(p) for each p in p_grid.
    """
    sample = np.asarray(sample, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)
    n = len(sample)
    x_sorted = np.sort(sample)
    # For each p, idx = ceil(p*n) - 1 in 0-indexed form
    raw_idx = np.ceil(p_grid * n).astype(int) - 1
    idx = np.clip(raw_idx, 0, n - 1)
    return x_sorted[idx]


# ---------------------------------------------------------------------------
# Empirical Expected Shortfall
# ---------------------------------------------------------------------------

def empirical_es(sample: np.ndarray, p: float) -> float:
    """Empirical Expected Shortfall ES_p(X).

    For p < 1:
        ES_p(X) = -(1/((1-p)*n)) * sum_{i=1}^{floor((1-p)*n)} x_{(i)}
                  - frac * x_{(floor((1-p)*n)+1)} / ((1-p)*n)
    where x_{(1)} <= ... <= x_{(n)} are order statistics and
          frac = (1-p)*n - floor((1-p)*n) (fractional weight for the last bin).

    For p == 1:
        ES_1(X) = rho^w(X) = max(-X) = -min(X)  (worst-case loss).

    Parameters
    ----------
    sample : array-like, shape (n,)
    p      : probability level in [0, 1)

    Returns
    -------
    float : ES_p(X)
    """
    sample = np.asarray(sample, dtype=float)
    n = len(sample)
    if n == 0:
        raise ValueError("sample must be non-empty")
    if p >= 1.0:
        return float(np.max(-sample))  # worst-case loss
    if p < 0.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")

    x_sorted = np.sort(sample)
    threshold = (1.0 - p) * n  # number of tail observations
    k = int(np.floor(threshold))  # full observations
    frac = threshold - k           # fractional weight

    if k == 0:
        # Only fractional part, pointing at x_{(1)}
        return -float(x_sorted[0])

    tail_sum = np.sum(x_sorted[:k])
    if frac > 0.0 and k < n:
        tail_sum += frac * x_sorted[k]
        denom = threshold
    else:
        denom = float(k)

    return -tail_sum / denom


def empirical_es_grid(sample: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
    """Vectorised empirical ES over a grid of probability levels.

    Sorts the sample once and evaluates all p-levels in a single pass using
    prefix sums, giving O(n log n + m) complexity versus O(m * n log n) for
    repeated calls to :func:`empirical_es`.

    Parameters
    ----------
    sample : array-like, shape (n,)
    p_grid : array-like of probability levels in [0, 1], shape (m,)

    Returns
    -------
    np.ndarray of shape (m,)
    """
    sample = np.asarray(sample, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)
    n = len(sample)
    if n == 0:
        raise ValueError("sample must be non-empty")

    x_sorted = np.sort(sample)
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(x_sorted)  # prefix[k] = sum of first k order stats

    results = np.empty(len(p_grid))
    for i, p in enumerate(p_grid):
        if p >= 1.0:
            results[i] = float(-x_sorted[0])  # max(-X)
        else:
            threshold = (1.0 - p) * n
            k = int(np.floor(threshold))
            frac = threshold - k
            if k == 0:
                results[i] = -x_sorted[0]
            else:
                tail_sum = prefix[k]
                if frac > 0.0 and k < n:
                    tail_sum += frac * x_sorted[k]
                    denom = threshold
                else:
                    denom = float(k)
                results[i] = -tail_sum / denom
    return results


# ---------------------------------------------------------------------------
# Worst-case (essential supremum of loss)
# ---------------------------------------------------------------------------

def rho_w(sample: np.ndarray) -> float:
    """Worst-case risk measure: rho^w(X) = ess_sup(-X) = max(-X).

    Parameters
    ----------
    sample : array-like, shape (n,)

    Returns
    -------
    float
    """
    sample = np.asarray(sample, dtype=float)
    return float(np.max(-sample))
