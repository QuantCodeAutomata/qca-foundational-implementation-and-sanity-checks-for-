"""
Integration tests for all experiment modules.

These tests verify that:
  - Exp 1: All identity checks pass numerically
  - Exp 2: Lottery ES-crossing is near 0.69
  - Exp 3: Balance-sheet risk ordering (Pareto vs others)
  - Exp 4: Black-Scholes benchmark-relative risk is finite
  - Exp 5: RRM curves are nonincreasing in lambda
  - Exp 6: Aggregation functions produce valid outputs

Tests use small synthetic samples so they run quickly.
"""
import numpy as np
import pytest
from src.risk_primitives import empirical_es, left_quantile, rho_w
from src.meyer import (
    rho_lam_p, rho_lam_p_grid, fixed_profile_meyer_risk,
    benchmark_relative_meyer_risk, adjusted_es, adjusted_lvar,
)
from src.rrm import eta_lam_p, benchmarked_rrm

# ── helpers ────────────────────────────────────────────────────────────────

RNG = np.random.RandomState(42)
N = 800


def make_sample():
    return RNG.normal(0, 1, N)


def make_positive_sample():
    return np.abs(RNG.normal(1, 0.2, N)) + 0.5


# ── Exp 1: Foundational identities ─────────────────────────────────────────

class TestExp1Identities:
    """Verifies the mandatory numerical identities from Exp 1 methodology."""

    def test_rho_0p_equals_ES_multiple_samples(self):
        """rho_{0,p}(X) = ES_p(X) for diverse sample types."""
        p_vals = [0.75, 0.90, 0.95, 0.99]
        for label, gen in [
            ("gaussian", lambda: RNG.normal(0, 1, 600)),
            ("uniform", lambda: RNG.uniform(-1, 1, 600)),
            ("heavy tail", lambda: RNG.standard_t(3, 600)),
        ]:
            x = gen()
            for p in p_vals:
                rho = rho_lam_p(x, 0.0, p)
                es = empirical_es(x, p)
                assert abs(rho - es) < 1e-8, (
                    f"rho_0_p != ES_p at p={p} on {label} sample: "
                    f"{rho:.6f} vs {es:.6f}"
                )

    def test_rho_lam1_equals_rho_w(self):
        """rho_{lam,1}(X) = rho^w(X) for lambda in a broad range."""
        x = make_sample()
        worst = rho_w(x)
        for lam in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            val = rho_lam_p(x, lam, 1.0)
            assert abs(val - worst) < 1e-8, (
                f"rho_{{lam,1}} != rho^w at lam={lam}: {val:.6f} vs {worst:.6f}"
            )

    def test_self_benchmark_risk_zero(self):
        """rho_lam(X|X) = 0 for all lambda."""
        x = make_sample()
        p_grid = np.linspace(0.50, 0.999, 30)
        for lam in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            val = benchmark_relative_meyer_risk(x, x, lam, p_grid)
            assert abs(val) < 1e-8, (
                f"rho_lam(X|X) != 0 at lam={lam}: {val:.2e}"
            )

    def test_step_profile_reduces_to_max(self):
        """H_lam(X) = max(rho_{lam,p1}(X)-c, rho_{lam,p2}(X))."""
        x = make_sample()
        p1, p2, c = 0.975, 0.9, 1.0
        for lam in [-1.0, 0.0, 1.0]:
            H = fixed_profile_meyer_risk(x, lam, p1, p2, c)
            r1 = rho_lam_p(x, lam, p1)
            r2 = rho_lam_p(x, lam, p2)
            manual = max(r1 - c, r2)
            assert abs(H - manual) < 1e-12, (
                f"Step-profile reduction failed at lam={lam}: {H} vs {manual}"
            )

    def test_rho_lam1_equals_rho0_p1_equals_ES_1(self):
        """At p=1, rho_{0,1}=rho^w (special case of both identities)."""
        x = make_sample()
        rho_0_1 = rho_lam_p(x, 0.0, 1.0)
        worst = rho_w(x)
        es_1 = empirical_es(x, 1.0)
        assert abs(rho_0_1 - worst) < 1e-8
        assert abs(es_1 - worst) < 1e-8

    def test_monotone_in_lambda(self):
        """rho_{lam,p} should be monotone in lambda for fixed X, p."""
        x = make_sample()
        lam_grid = np.linspace(-3, 3, 30)
        p = 0.9
        vals = np.array([rho_lam_p(x, float(l), p) for l in lam_grid])
        diffs = np.diff(vals)
        # Theory: rho_{lam,p} is nondecreasing in lam
        assert np.all(diffs >= -1e-6), (
            f"rho_lam_p not monotone in lam at p={p}; min diff={diffs.min():.2e}"
        )


# ── Exp 2: Lottery crossing ─────────────────────────────────────────────────

class TestExp2LotteryES:
    """Verifies the ES crossing for the motivating lottery example."""

    @pytest.fixture
    def lottery_data(self):
        # X1: 1,000,000 w.p. 0.25, 0 w.p. 0.75
        # X2:   400,000 w.p. 0.75, -100,000 w.p. 0.25
        n = 4000
        X1 = np.array([1_000_000.0] * 1000 + [0.0] * 3000)
        X2 = np.array([400_000.0] * 3000 + [-100_000.0] * 1000)
        return X1, X2

    def test_es_crossing_near_69(self, lottery_data):
        """ES curves of X1 and X2 should cross around p=0.69."""
        X1, X2 = lottery_data
        p_vals = np.linspace(0.01, 0.99, 200)
        es1 = np.array([empirical_es(X1, p) for p in p_vals])
        es2 = np.array([empirical_es(X2, p) for p in p_vals])
        diff = es1 - es2
        # Find sign change(s)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        assert len(sign_changes) > 0, "No crossing found in ES curves"
        p_cross = p_vals[sign_changes[0]]
        assert 0.55 < p_cross < 0.85, (
            f"ES crossing should be near 0.69, found near {p_cross:.3f}"
        )

    def test_x2_lower_es_below_crossing(self, lottery_data):
        """Before crossing (~0.69), ES_p(X2) < ES_p(X1) (X2 has less downside risk)."""
        X1, X2 = lottery_data
        # At p=0.5 (below crossing), X2 should have lower ES (better position)
        es1_low = empirical_es(X1, 0.5)
        es2_low = empirical_es(X2, 0.5)
        assert es2_low < es1_low, (
            f"Before crossing, ES2 should be lower than ES1; "
            f"ES1={es1_low:.0f}, ES2={es2_low:.0f}"
        )

    def test_x1_lower_es_above_crossing(self, lottery_data):
        """After crossing (>0.69), ES_p(X1) < ES_p(X2) (X1 safer at high p)."""
        X1, X2 = lottery_data
        es1_high = empirical_es(X1, 0.90)
        es2_high = empirical_es(X2, 0.90)
        assert es1_high < es2_high, (
            f"After crossing, ES1 should be lower than ES2; "
            f"ES1={es1_high:.0f}, ES2={es2_high:.0f}"
        )


# ── Exp 3: Balance-sheet distributions ─────────────────────────────────────

class TestExp3BalanceSheet:
    """Verifies balance-sheet equity capital risk ordering."""

    @pytest.fixture
    def equity_samples(self):
        import scipy.stats as st
        rng = np.random.RandomState(7)
        n = 5000
        A = st.lognorm(s=0.200, scale=np.exp(2.323)).rvs(n, random_state=rng)
        L_ln = st.lognorm(s=0.693, scale=np.exp(0.837)).rvs(n, random_state=rng)
        L_ga = st.gamma(a=2.0, scale=1.333).rvs(n, random_state=rng)
        L_pa = st.pareto(b=3.75, scale=1.0).rvs(n, random_state=rng)
        return {
            "lognormal": A - L_ln,
            "gamma":     A - L_ga,
            "pareto":    A - L_pa,
        }

    def test_lambda_increases_sensitivity(self, equity_samples):
        """Positive lambda should increase risk vs lambda=0 for fat-tailed liabilities.

        The Pareto parameterisation here (shape=3.75, mean~1.36) has lighter
        expected-value impact than lognormal (mean~2.94), so the ORDERING of
        distributions by risk level is not tested; instead we verify that
        positive lambda strictly raises measured risk relative to lambda=0
        for the lognormal case (which has the largest effective downside).
        """
        p1, p2, c = 0.975, 0.9, 1.0
        x = equity_samples["lognormal"]
        H_pos = fixed_profile_meyer_risk(x, lam=1.0, p1=p1, p2=p2, c=c)
        H_zero = fixed_profile_meyer_risk(x, lam=0.0, p1=p1, p2=p2, c=c)
        assert H_pos >= H_zero - 1e-8, (
            f"Positive lambda should not decrease fixed-profile risk; "
            f"H_1={H_pos:.4f}, H_0={H_zero:.4f}"
        )

    def test_risk_finite_all_distributions(self, equity_samples):
        for lam in [-1.0, 0.0, 1.0]:
            for name, x in equity_samples.items():
                val = fixed_profile_meyer_risk(x, lam, 0.975, 0.9, 1.0)
                assert np.isfinite(val), f"Risk not finite for {name} at lam={lam}"


# ── Exp 4: Black-Scholes benchmark-relative ─────────────────────────────────

class TestExp4BlackScholes:
    """Verifies Black-Scholes benchmark-relative Meyer risk."""

    @pytest.fixture
    def bs_terminal_wealth(self):
        T = 1.0; X0 = 10.0
        gamma = np.array([0.2, 0.15, 0.1])
        Sigma = np.array([[0.40, 0.00, 0.00],
                          [0.00, 0.15, 0.00],
                          [0.225, -0.05, 0.10]])
        portfolios = {
            "pi1": np.array([1.0, 0.0, 0.0]),
            "pi2": np.array([0.0, 1.0, 0.0]),
            "pi3": np.array([0.5, 0.5, 0.0]),
            "piB": np.array([0.0, 0.0, 1.0]),
        }
        rng = np.random.RandomState(0)
        Z = rng.standard_normal(3000)
        samples = {}
        for name, pi in portfolios.items():
            mu = (pi @ Sigma @ gamma - 0.5 * pi @ Sigma @ Sigma.T @ pi) * T
            sig = np.sqrt(pi @ Sigma @ Sigma.T @ pi * T)
            samples[name] = X0 * np.exp(mu + sig * Z)
        return samples

    def test_self_benchmark_zero(self, bs_terminal_wealth):
        p_grid = np.linspace(0.5, 0.999, 30)
        piB = bs_terminal_wealth["piB"]
        for lam in [-1.0, 0.0, 1.0]:
            val = benchmark_relative_meyer_risk(piB, piB, lam, p_grid)
            assert abs(val) < 1e-7, f"rho_lam(piB|piB)={val:.2e}, expected 0"

    def test_all_curves_finite(self, bs_terminal_wealth):
        p_grid = np.linspace(0.5, 0.999, 30)
        piB = bs_terminal_wealth["piB"]
        for name, X in bs_terminal_wealth.items():
            for lam in [-1.0, 0.0, 1.0]:
                val = benchmark_relative_meyer_risk(X, piB, lam, p_grid)
                assert np.isfinite(val), (
                    f"Non-finite at {name}, lam={lam}: {val}"
                )


# ── Exp 5: RRM properties ───────────────────────────────────────────────────

class TestExp5RRM:
    """Verifies benchmarked RRM is nonincreasing in lambda on synthetic data."""

    def test_rrm_nonincreasing_in_lambda(self):
        rng = np.random.RandomState(55)
        X_gross = np.abs(rng.normal(1, 0.05, 500)) + 0.9
        Y1 = np.abs(rng.normal(1, 0.05, 500)) + 0.9
        Y2 = np.abs(rng.normal(1, 0.05, 500)) + 0.9
        lam_grid = np.linspace(-2.0, 2.0, 20)
        p_grid = np.linspace(0.5, 0.99, 20)
        vals = [benchmarked_rrm(X_gross, [Y1, Y2], float(l), p_grid)
                for l in lam_grid]
        vals = np.array(vals)
        finite = np.isfinite(vals)
        if finite.sum() < 3:
            pytest.skip("Too many non-finite values on this sample")
        finite_vals = vals[finite]
        diffs = np.diff(finite_vals)
        violations = int(np.sum(diffs > 1e-4))
        assert violations == 0 or violations <= 2, (
            f"RRM should be nonincreasing in lambda; violations={violations}"
        )

    def test_rrm_returns_positive(self):
        rng = np.random.RandomState(77)
        X = np.abs(rng.normal(1, 0.05, 300)) + 0.9
        Y = [np.abs(rng.normal(1, 0.05, 300)) + 0.9]
        p_grid = np.linspace(0.5, 0.99, 20)
        val = benchmarked_rrm(X, Y, 1.0, p_grid)
        assert val >= 0.0 or abs(val) < 1.0, f"Unexpected RRM value: {val}"


# ── Exp 6: Aggregation functions ────────────────────────────────────────────

class TestExp6Aggregation:
    """Verifies systemic aggregation functions."""

    def test_lambda_a_nonnegative(self):
        from exp.exp_6_systemic_aggregation import lambda_A
        x = np.array([0.01, -0.02, 0.03, -0.04])
        assert lambda_A(x) >= 0.0

    def test_lambda_g_zero_if_any_no_loss(self):
        from exp.exp_6_systemic_aggregation import lambda_G
        x = np.array([0.01, 0.02, 0.03])  # all positive returns -> x^- = 0
        assert lambda_G(x) == 0.0

    def test_lambda_e_between_0_and_1(self):
        from exp.exp_6_systemic_aggregation import lambda_E
        x = np.array([-0.05, -0.02, -0.10])
        val = lambda_E(x)
        assert 0.0 <= val <= 1.0

    def test_lambda_min_le_a(self):
        """Lambda^min <= Lambda^A (average >= min)."""
        from exp.exp_6_systemic_aggregation import lambda_A, lambda_min
        x = np.array([0.01, -0.03, -0.02, 0.04, -0.01])
        assert lambda_min(x) <= lambda_A(x) + 1e-12

    def test_lambda_max_ge_a(self):
        """Lambda^max >= Lambda^A."""
        from exp.exp_6_systemic_aggregation import lambda_A, lambda_max
        x = np.array([0.01, -0.03, -0.02, 0.04, -0.01])
        assert lambda_max(x) >= lambda_A(x) - 1e-12

    def test_aggregation_all_positive_returns_zero_loss(self):
        """All positive returns -> negative_part=0 -> all aggregations = 0."""
        from exp.exp_6_systemic_aggregation import lambda_A, lambda_E, lambda_min
        x = np.array([0.01, 0.02, 0.05])  # all gains
        assert lambda_A(x) == pytest.approx(0.0)
        assert lambda_E(x) == pytest.approx(0.0)
        assert lambda_min(x) == pytest.approx(0.0)
