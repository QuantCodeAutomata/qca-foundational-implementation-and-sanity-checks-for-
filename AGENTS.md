# Repository Knowledge — Meyer Risk Measures

## Project Purpose
Full implementation and numerical reproduction of Meyer CARA-transformed tail risk measures,
benchmark-relative risk, and power-utility return risk measures (RRM) from the underlying
theoretical paper. Six experiments are implemented and validated.

## Module Map
| File | Role |
|------|------|
| `src/risk_primitives.py` | `left_quantile`, `empirical_es`, `empirical_es_grid`, `rho_w` |
| `src/meyer.py` | `cara_utility`, `cara_inverse`, `rho_lam_p`, `fixed_profile_meyer`, `adjusted_es`, `adjusted_lvar`, `benchmark_relative_risk` |
| `src/rrm.py` | `power_utility`, `power_utility_inverse`, `eta_lam_p`, `eta_lam_p_grid`, `benchmarked_rrm` |
| `src/utils.py` | `safe_log`, `safe_exp`, numeric helpers |
| `data/data_loader.py` | `load_index_prices()` — stooq loader with synthetic fallback |
| `exp/exp_1_sanity_checks.py` | Core identity verification (rho_0=ES, etc.) |
| `exp/exp_2_lottery.py` | Two-lottery ES-crossing illustration |
| `exp/exp_3_balance_sheet.py` | Lognormal/gamma/Pareto liability simulation |
| `exp/exp_4_black_scholes.py` | Black-Scholes benchmark-relative Meyer risk |
| `exp/exp_5_historical.py` | Historical 7-index study with RRM curves |
| `exp/exp_6_systemic_aggregation.py` | Systemic aggregation with 5 aggregation functions |

## Important Conventions
- **Left quantile**: `q_X(p) = inf{x: F_X(x) >= p}`, left-continuous CDF. Implemented as `np.sort(x)[ceil(p*n)-1]`.
- **ES at p=1**: hard-coded as `max(-X)` (worst-case / rho^w), not numerically integrated.
- **Lambda threshold**: `_LAM_EPS = 1e-10` in `meyer.py`. Branch to `ES_p` when `|lambda| < _LAM_EPS`.
- **Exp clipping**: `_EXP_CLIP = 500` prevents overflow in `cara_utility`.
- **Positive inputs for RRM**: `eta_lam_p` and `benchmarked_rrm` require strictly positive samples (gross returns).
- **Step profile**: `g(p) = c * 1_{[p2,p1)}(p) + inf * 1_{[p1,1]}(p)` collapses supremum to `max(rho_{lam,p1}(X)-c, rho_{lam,p2}(X))`.

## Test Coverage
- `tests/test_primitives.py`: 49 unit tests for quantile, ES, CARA, rho, adjusted measures, power utility, eta, edge cases.
- `tests/test_experiments.py`: 15 integration tests for exp_2 through exp_6.
- Run with: `python -m pytest tests/ -v` (64 total, all passing in ~1 s).

## Data Notes
- Market data for exp_5 and exp_6 is fetched from stooq via `pandas-datareader`.
- If stooq is unavailable, `data_loader.py` falls back to synthetic calibrated data.
- **Do not commit `data/cache/`** — cached CSV files from stooq are excluded via `.gitignore`.

## Common Parameterizations
- **Scipy Pareto**: `scipy.stats.pareto(b=3.75, scale=1.0, loc=0)` → support x ≥ 1 (paper: shape=3.75, location=1).
- **Scipy lognormal**: `scipy.stats.lognorm(s=sigma, scale=np.exp(mu))`.
- **Scipy gamma**: `scipy.stats.gamma(a=shape, scale=scale)`.
- **Black-Scholes T**: T=1.0 (paper unspecified). Sensitivity checked for T ∈ {0.5, 1, 2}.

## Results Location
All output is in `results/`: PNGs named `expN_*.png`, CSVs named `expN_*.csv`, and `RESULTS.md`.

## Dependency Stack
numpy, scipy, pandas, matplotlib, seaborn, statsmodels, pandas-datareader.
No specialized quant library (skfolio, vectorbt, etc.) is needed — all risk measures are custom.
