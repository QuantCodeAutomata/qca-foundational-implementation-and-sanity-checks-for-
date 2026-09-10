"""
Meyer risk measures: CARA threshold-utility transforms and rho_{lambda,p}.

All functions are custom implementations — Context7 confirms no standard
quantitative library provides Meyer / CARA-entropic tail risk measures.

Key formula (paper convention):
    rho_{lam,p}(X) = -e_lam^{-1}(-ES_p(e_lam(X)))
where
    e_lam(x) = -lam * exp(-lam * x)  for lam != 0
    e_0(x)   = x

Inverse:
    e_lam^{-1}(y) = -(1/lam) * log(-y/lam)  for lam != 0
    e_0^{-1}(y)   = y

Special cases (implemented exactly, not numerically):
    rho_{0,p}(X)   = ES_p(X)
    rho_{lam,1}(X) = rho^w(X) = max(-X)
"""
import numpy as np
from src.risk_primitives import empirical_es, empirical_es_grid, left_quantile, rho_w
from src.utils import safe_log

# threshold for treating lambda as zero
_LAM_EPS = 1e-10
# clipping bound for exponent to prevent overflow
_EXP_CLIP = 500.0


# ---------------------------------------------------------------------------
# CARA threshold utility and inverse
# ---------------------------------------------------------------------------

def cara_utility(x: np.ndarray, lam: float) -> np.ndarray:
    """CARA threshold utility e_lam(x).

    e_lam(x) = -lam * exp(-lam * x)  for lam != 0
    e_0(x)   = x                       for lam == 0

    Note: sign convention matches the paper so that e_lam is strictly
    increasing in x for all lam.

    Parameters
    ----------
    x   : array-like of floats
    lam : risk-aversion parameter (any real)

    Returns
    -------
    np.ndarray, same shape as x
    """
    x = np.asarray(x, dtype=float)
    if abs(lam) < _LAM_EPS:
        return x.copy()
    exponent = np.clip(-lam * x, -_EXP_CLIP, _EXP_CLIP)
    return -lam * np.exp(exponent)


def cara_utility_inverse(y: np.ndarray, lam: float) -> np.ndarray:
    """Inverse CARA threshold utility e_lam^{-1}(y).

    e_lam^{-1}(y) = -(1/lam) * log(-y/lam)  for lam != 0
    e_0^{-1}(y)   = y                          for lam == 0

    Domain requirements:
      - lam > 0: y must be < 0, so -y/lam > 0.
      - lam < 0: y must be > 0, so -y/lam = -y/negative > 0.
    A small epsilon clip is applied to the log argument for stability.

    Parameters
    ----------
    y   : array-like of floats (must lie in the range of e_lam)
    lam : risk-aversion parameter

    Returns
    -------
    np.ndarray, same shape as y
    """
    y = np.asarray(y, dtype=float)
    if abs(lam) < _LAM_EPS:
        return y.copy()
    # argument: -y / lam
    arg = -y / lam
    # clip from below to avoid log(0)
    arg_safe = np.maximum(arg, 1e-300)
    return -(1.0 / lam) * np.log(arg_safe)


# ---------------------------------------------------------------------------
# CARA-Meyer tail risk measure rho_{lam,p}
# ---------------------------------------------------------------------------

def rho_lam_p(sample: np.ndarray, lam: float, p: float) -> float:
    """CARA-Meyer tail risk measure rho_{lam,p}(X).

    rho_{lam,p}(X) = -e_lam^{-1}(-ES_p(e_lam(X)))

    Special cases (exact, not approximated):
      - lam ~ 0   -> ES_p(X)
      - p  == 1.0 -> rho^w(X) = max(-X)

    Parameters
    ----------
    sample : array-like, shape (n,)
    lam    : risk-aversion parameter
    p      : tail probability level in [0, 1]

    Returns
    -------
    float : rho_{lam,p}(X)
    """
    sample = np.asarray(sample, dtype=float)
    # Special case p == 1
    if p >= 1.0:
        return rho_w(sample)
    # Special case lam ~ 0
    if abs(lam) < _LAM_EPS:
        return empirical_es(sample, p)
    # General case
    z = cara_utility(sample, lam)          # e_lam(X)
    es_z = empirical_es(z, p)              # ES_p(e_lam(X))
    inner = -es_z                          # -ES_p(e_lam(X))
    # Apply e_lam^{-1}; wrap as array for inverse
    inv_val = cara_utility_inverse(np.array([inner]), lam)[0]
    return float(-inv_val)


def rho_lam_p_grid(sample: np.ndarray, lam: float, p_grid: np.ndarray) -> np.ndarray:
    """Vectorised rho_{lam,p} over a probability grid.

    Pre-transforms the sample once with the CARA utility, then computes ES for
    all p values in a single-sort pass (O(n log n + m) instead of
    O(m * n log n)).

    Parameters
    ----------
    sample : array-like, shape (n,)
    lam    : risk-aversion parameter
    p_grid : array-like of levels in [0, 1], shape (m,)

    Returns
    -------
    np.ndarray of shape (m,)
    """
    p_grid = np.asarray(p_grid, dtype=float)
    sample = np.asarray(sample, dtype=float)

    if abs(lam) < _LAM_EPS:
        return empirical_es_grid(sample, p_grid)

    rho_w_val = rho_w(sample)
    z = cara_utility(sample, lam)       # e_lam(X) — transformed once

    # All p < 1 ES values computed with a single sort via prefix sums
    non_one = p_grid < 1.0
    results = np.empty(len(p_grid))
    results[~non_one] = rho_w_val

    if np.any(non_one):
        es_z = empirical_es_grid(z, p_grid[non_one])   # single sort of z
        inner = -es_z                                    # -ES_p(e_lam(X))
        results[non_one] = -cara_utility_inverse(inner, lam)
    return results


# ---------------------------------------------------------------------------
# Fixed-profile (step-profile) Meyer risk
# ---------------------------------------------------------------------------

def fixed_profile_meyer_risk(
    sample: np.ndarray, lam: float, p1: float, p2: float, c: float
) -> float:
    """Fixed-profile Meyer risk under a step target profile.

    Step profile g(p) = c * 1_{[p2,p1)}(p) + inf * 1_{[p1,1]}(p)

    Under this profile:
        H_lam(X) = sup_p {rho_{lam,p}(X) - g(p)}
                 = max(rho_{lam,p1}(X) - c, rho_{lam,p2}(X))

    Parameters
    ----------
    sample : array-like, shape (n,)
    lam    : risk-aversion parameter
    p1     : upper confidence level (constraint level), e.g. 0.975
    p2     : lower confidence level, e.g. 0.9
    c      : capital buffer cost (step height)

    Returns
    -------
    float : H_lam(X)
    """
    r1 = rho_lam_p(sample, lam, p1)
    r2 = rho_lam_p(sample, lam, p2)
    return max(r1 - c, r2)


# ---------------------------------------------------------------------------
# Adjusted ES and adjusted Loss VaR (lambda-independent comparators)
# ---------------------------------------------------------------------------

def adjusted_es(sample: np.ndarray, p1: float, p2: float, c: float) -> float:
    """Adjusted ES under the step profile (lambda=0 special case of H_lam).

    ES^g(X) = max(ES_{p1}(X) - c, ES_{p2}(X))

    Parameters
    ----------
    sample : array-like, shape (n,)
    p1, p2 : confidence levels (p1 > p2)
    c      : cost parameter

    Returns
    -------
    float
    """
    es1 = empirical_es(sample, p1)
    es2 = empirical_es(sample, p2)
    return max(es1 - c, es2)


def adjusted_lvar(sample: np.ndarray, p1: float, p2: float, c: float) -> float:
    """Adjusted Loss VaR under the step profile.

    LVaR^g(X) = sup_p {-q_X(1-p) - g(p)}
              = max(-q_X(1-p1) - c, -q_X(1-p2))

    under the step profile with g(p)=c for p in [p2,p1) and g(p)=0 for p<p2.

    Parameters
    ----------
    sample : array-like, shape (n,)
    p1, p2 : confidence levels
    c      : cost parameter

    Returns
    -------
    float
    """
    q1 = left_quantile(sample, 1.0 - p1)
    q2 = left_quantile(sample, 1.0 - p2)
    return max(-q1 - c, -q2)


# ---------------------------------------------------------------------------
# Benchmark-relative Meyer risk
# ---------------------------------------------------------------------------

def benchmark_relative_meyer_risk(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    lam: float,
    p_grid: np.ndarray,
) -> float:
    """Benchmark-relative CARA-Meyer risk rho_lam(X|Y).

    rho_lam(X|Y) = sup_p {rho_{lam,p}(X) - rho_{lam,p}(Y)}

    The supremum is evaluated over the discrete p_grid.

    Parameters
    ----------
    sample_x : array-like, shape (n,)   target portfolio
    sample_y : array-like, shape (m,)   benchmark portfolio
    lam      : risk-aversion parameter
    p_grid   : array-like of levels in [0, 1]

    Returns
    -------
    float : benchmark-relative risk
    """
    sample_x = np.asarray(sample_x, dtype=float)
    sample_y = np.asarray(sample_y, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)

    rho_x = rho_lam_p_grid(sample_x, lam, p_grid)
    rho_y = rho_lam_p_grid(sample_y, lam, p_grid)
    return float(np.max(rho_x - rho_y))
