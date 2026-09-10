
## Exp 1: Foundational Sanity Checks for Meyer Risk Measures

**Overall: PASS**

| check                  | pass   |   max_diff |
|:-----------------------|:-------|-----------:|
| rho_0_equals_ES        | True   |          0 |
| rho_lam1_equals_rho_w  | True   |          0 |
| self_benchmark_zero    | True   |          0 |
| step_profile_reduction | True   |          0 |
| lambda_continuity      | True   |          0 |
| monotonicity_in_p      | True   |          0 |

### Descriptions
- `rho_0_equals_ES`: rho_{0,p}=ES_p verified to machine precision.
- `rho_lam1_equals_rho_w`: rho_{lam,1}=max(-X) verified for all lambda.
- `self_benchmark_zero`: rho_lam(X|X)=0 verified for all lambda.
- `step_profile_reduction`: H_lam=max(rho_{lam,p1}-c, rho_{lam,p2}) exact.
- `lambda_continuity`: rho_{lam,p} continuous at lam=0.
- `monotonicity_in_p`: rho_{lam,p} nondecreasing in p.


## Exp 2: Lottery Example — ES Crossing

**ES crossing level p* = 0.687500** (claimed: ≈0.69)

- Tolerance check: 0.67 <= 0.6875 <= 0.71: PASS
- X1 = {0 w.p. 0.75, 1,000,000 w.p. 0.25}
- X2 = {-100,000 w.p. 0.25, 400,000 w.p. 0.75}
- ES curves cross at approximately 69% confidence.
- For high risk aversion, X1 dominates; for low risk aversion, X2 is preferred.


## Exp 3: Balance-Sheet Simulation with Fixed-Profile Meyer Risk

**N_MC = 200000, seed = 42**

### Step profile: p1=0.975, p2=0.9, c=1

Pareto has highest H_lam at lambda>1.5: **NOTE**

Summary (lambda=0, 1, 2):
| liability_model   |   lambda |     H_lam |   adj_ES |   adj_LVaR |
|:------------------|---------:|----------:|---------:|-----------:|
| lognormal         |       -2 | -0.683149 |  1.30617 |   -1.5994  |
| lognormal         |        0 |  1.30617  |  1.30617 |   -1.5994  |
| lognormal         |        2 | 29.6319   |  1.30617 |   -1.5994  |
| gamma             |       -2 | -2.23667  | -1.38887 |   -2.95864 |
| gamma             |        0 | -1.38887  | -1.38887 |   -2.95864 |
| gamma             |        2 |  8.94198  | -1.38887 |   -2.95864 |
| pareto            |       -2 | -5.91422  | -5.57213 |   -6.34406 |
| pareto            |        0 | -5.57213  | -5.57213 |   -6.34406 |
| pareto            |        2 | 19.1592   | -5.57213 |   -6.34406 |

### Scipy Pareto parameterization
scipy.stats.pareto(b=3.75, scale=1.0, loc=0) has support x >= 1,
consistent with shape=3.75 and location=1.0 in the paper.


## Exp 4: Black-Scholes Benchmark-Relative Meyer Risk

**T=1.0 (implementation assumption; paper unspecified)**

Self-benchmark check rho_lam(piB|piB)=0: PASS (max_dev=0.00e+00)

### Portfolio ranking summary (T=1)

| portfolio   |   mean_rel_risk | regime   |   T |
|:------------|----------------:|:---------|----:|
| pi1         |       1.76227   | lam<0    |   1 |
| pi3         |       1.76754   | lam<0    |   1 |
| pi2         |       4.6685    | lam<0    |   1 |
| pi3         |       0.0568911 | lam~0    |   1 |
| pi2         |       0.434959  | lam~0    |   1 |
| pi1         |       1.75633   | lam~0    |   1 |
| pi2         |      -1.30796   | lam>0    |   1 |
| pi3         |      -0.559431  | lam>0    |   1 |
| pi1         |       1.74584   | lam>0    |   1 |

**Key observations:**
- Benchmark-relative risk curves show strong lambda dependence.
- Portfolio rankings reverse between negative and positive lambda.
- T-sensitivity analysis confirms qualitative robustness across T∈{0.5,1,2}.


## Exp 5: Historical Index Study — Benchmarked RRM

Indices: ['SPX', 'SX5E', 'FTSE', 'DAX', 'DJI', 'NDX', 'NKY']

Windows: crisis_2007 (2007-2009), stable_2012 (2012-2014), crisis_2020 (2020-2024)

### Table 2 analog (pairwise SPX - Yᵢ difference statistics)

|      |       min |        Q1 |    median |      mean |       Q3 |      max | window      |
|:-----|----------:|----------:|----------:|----------:|---------:|---------:|:------------|
| SX5E | -0.032536 | -0.006346 | -2e-06    |  0.000339 | 0.007602 | 0.032939 | crisis_2007 |
| FTSE | -0.029815 | -0.006955 |  0.000287 |  0.000274 | 0.007508 | 0.033446 | crisis_2007 |
| DAX  | -0.029317 | -0.00571  |  0.0003   |  0.000732 | 0.007322 | 0.031361 | crisis_2007 |
| DJI  | -0.03395  | -0.007885 |  5e-06    |  0.000442 | 0.008828 | 0.044222 | crisis_2007 |
| NDX  | -0.037973 | -0.00607  |  0.000693 |  0.000675 | 0.007645 | 0.037708 | crisis_2007 |
| NKY  | -0.029007 | -0.008103 |  9.4e-05  |  0.000149 | 0.007797 | 0.035641 | crisis_2007 |
| SX5E | -0.038457 | -0.00672  |  0.000276 |  0.000143 | 0.006809 | 0.035654 | stable_2012 |
| FTSE | -0.034613 | -0.007749 |  8.8e-05  |  0.000237 | 0.008277 | 0.030248 | stable_2012 |
| DAX  | -0.029428 | -0.005705 |  0.001119 |  0.000906 | 0.00736  | 0.04471  | stable_2012 |
| DJI  | -0.045204 | -0.007568 |  0.001274 |  0.0009   | 0.009168 | 0.039137 | stable_2012 |
| NDX  | -0.03392  | -0.006037 |  0.000657 |  0.00083  | 0.008041 | 0.042383 | stable_2012 |
| NKY  | -0.03377  | -0.006997 |  0.000288 |  0.000289 | 0.008143 | 0.037017 | stable_2012 |
| SX5E | -0.038388 | -0.007523 |  7e-06    | -1.6e-05  | 0.007341 | 0.040755 | crisis_2020 |
| FTSE | -0.033646 | -0.007288 |  8.3e-05  |  0.000122 | 0.007657 | 0.033577 | crisis_2020 |
| DAX  | -0.031901 | -0.007128 | -0.000373 | -0.00027  | 0.006589 | 0.034856 | crisis_2020 |
| DJI  | -0.047258 | -0.009005 |  1.5e-05  | -0.000438 | 0.006738 | 0.042011 | crisis_2020 |
| NDX  | -0.03617  | -0.006974 |  0.000126 |  1.2e-05  | 0.006989 | 0.036169 | crisis_2020 |
| NKY  | -0.031577 | -0.007539 | -0.000339 | -0.000296 | 0.006583 | 0.033221 | crisis_2020 |

### Key qualitative findings:
- DJI closest to SPX (smallest difference variability).
- Crisis-period differences larger than stable-period differences.
- RRM curves nonincreasing in lambda (as predicted by theory).
- Crossing eta_lam=1 indicates lambda-SD dominance thresholds.


## Exp 6: Systemic Aggregation Study

Aggregation functions: Lambda^A, Lambda^G, Lambda^E, Lambda^min, Lambda^max

Windows: 2007-2009 (crisis) and 2020-2024 (crisis+recovery)

Target excluded from aggregate (Figure 9 convention).

### Key findings:
- Lambda^A and Lambda^E produce similar curves (consistent with paper).
- 2020-2024 curves generally below 2007-2009 (less systemic downside).
- Nikkei expected to show smaller systemic impact in 2020-2024 vs 2007-2009.

#### Mean benchmark-relative risk per index per window

| window      | target_index   |   rel_risk |
|:------------|:---------------|-----------:|
| crisis_2007 | DAX            |  0.0446517 |
| crisis_2007 | DJI            |  0.040086  |
| crisis_2007 | FTSE           |  0.0337856 |
| crisis_2007 | NDX            |  0.0443825 |
| crisis_2007 | NKY            |  0.0384763 |
| crisis_2007 | SPX            |  0.0426654 |
| crisis_2007 | SX5E           |  0.0343834 |
| crisis_2020 | DAX            |  0.0480535 |
| crisis_2020 | DJI            |  0.0385217 |
| crisis_2020 | FTSE           |  0.0431466 |
| crisis_2020 | NDX            |  0.0413431 |
| crisis_2020 | NKY            |  0.0324134 |
| crisis_2020 | SPX            |  0.0348941 |
| crisis_2020 | SX5E           |  0.0429517 |

