# Meyer Risk Measures — Foundational Implementation and Experiments

Validated Python implementation of CARA-transformed tail risk measures, adjusted Expected
Shortfall / Loss VaR, benchmark-relative Meyer risk, and power-utility-based return risk
measures (RRM), together with six reproduced numerical experiments from the underlying
theoretical paper.

---

## Repository Structure

```
.
├── src/                        # Core mathematical primitives
│   ├── risk_primitives.py      # Left quantile, empirical ES, rho^w
│   ├── meyer.py                # e_lambda, rho_{lambda,p}, adjusted ES/LVaR, benchmark-relative risk
│   ├── rrm.py                  # Power utility, eta_{lambda,p}, benchmarked RRM
│   └── utils.py                # Numerical helpers (safe_log, etc.)
│
├── data/
│   ├── data_loader.py          # Daily index price loader (stooq / synthetic fallback)
│   └── .gitkeep
│
├── exp/                        # Experiment scripts
│   ├── exp_1_sanity_checks.py  # Core identity verification
│   ├── exp_2_lottery.py        # Motivating lottery + ES crossing
│   ├── exp_3_balance_sheet.py  # Balance-sheet simulation
│   ├── exp_4_black_scholes.py  # Black-Scholes benchmark-relative risk
│   ├── exp_5_historical.py     # Historical index study + RRM curves
│   └── exp_6_systemic_aggregation.py  # Systemic aggregation analysis
│
├── tests/
│   ├── test_primitives.py      # 55 unit tests for core primitives
│   └── test_experiments.py     # 15 experiment-level integration tests
│
├── results/                    # Generated figures, tables, and RESULTS.md
├── pyproject.toml
└── README.md
```

---

## Mathematical Framework

### Left Quantile
```
q_X(p) = inf{x : P(X ≤ x) ≥ p}
```

### Expected Shortfall
```
ES_p(X) = -(1/(1-p)) ∫₀^{1-p} q_X(r) dr,   p < 1
ES_1(X) = ρ^w(X) = ess_sup(-X)
```

### CARA Threshold Utility
```
e_λ(x) = -λ · exp(-λx),   λ ≠ 0
e_0(x) = x
```

### Meyer Risk Measure ρ_{λ,p}
```
ρ_{λ,p}(X) = -e_λ^{-1}(-ES_p(e_λ(X)))
```
Special cases (exact):
- `ρ_{0,p}(X) = ES_p(X)`
- `ρ_{λ,1}(X) = ρ^w(X) = max(-X)`

### Adjusted ES and Loss VaR
```
ES^g(X)   = sup_p { ES_p(X) - g(p) }
LVaR^g(X) = sup_p { -q_X(1-p) - g(p) }
```

### Benchmark-Relative Meyer Risk
```
ρ_λ(X|Y) = sup_p { ρ_{λ,p}(X) - ρ_{λ,p}(Y) }
```

### Return Risk Measure η_{λ,p}
```
η_{λ,p}(X) = p_λ^{-1}(-1 / ES_p(p_λ(X)))
```
where `p_a(x) = x^a/a` (power utility, `x > 0`).

---

## Experiments

| ID | Title | Key Result |
|----|-------|------------|
| exp_1 | Foundational sanity checks | All 6 core identities hold to machine precision |
| exp_2 | Lottery example — ES crossing | Crossing confirmed at p* ≈ 0.6875 (claimed ≈ 0.69) |
| exp_3 | Balance-sheet simulation | Positive λ amplifies Pareto tail danger vs adjusted ES |
| exp_4 | Black-Scholes benchmark risk | Portfolio rankings reverse between λ < 0 and λ > 0 |
| exp_5 | Historical index study | RRM curves nonincreasing in λ; crisis > stable period |
| exp_6 | Systemic aggregation | Λ^A ≈ Λ^E; 2020-2024 less severe than 2007-2009 |

---

## Installation

```bash
pip install -e .
pip install pandas-datareader seaborn statsmodels
```

---

## Running Experiments

```bash
# Run all experiments sequentially
python exp/exp_1_sanity_checks.py
python exp/exp_2_lottery.py
python exp/exp_3_balance_sheet.py
python exp/exp_4_black_scholes.py
python exp/exp_5_historical.py
python exp/exp_6_systemic_aggregation.py
```

Results (PNG figures and CSV tables) are written to `results/`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

**64 tests, all passing.**

---

## Key Design Decisions

- **Custom implementations only**: No standard quant library implements Meyer risk measures, adjusted entropic tail risk, or the paper's RRM family. All are implemented from scratch.
- **Exact order-statistic ES**: Empirical ES is computed from order statistics, not a coarse p-grid.
- **Branch switching at λ → 0**: `_LAM_EPS = 1e-10` threshold guards against overflow/underflow in `exp(-λx)`.
- **p = 1 exact handling**: `ρ_{λ,1}(X) = max(-X)` is hard-coded, not evaluated numerically.
- **Common random numbers**: Experiments using Monte Carlo simulation fix seeds and share draws across λ values.
- **Data fallback**: Market data (exp_5, exp_6) uses stooq via `pandas-datareader`; if unavailable, a synthetic calibrated substitute is used.

---

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full experiment outputs.
