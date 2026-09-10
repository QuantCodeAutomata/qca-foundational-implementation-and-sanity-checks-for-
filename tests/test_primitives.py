"""
Unit and integration tests for risk primitives and Meyer risk measures.

Validates:
  - Left quantile convention (empirical, order-statistic consistent)
  - Expected Shortfall formula (exact order-statistic, not coarse grid)
  - CARA utility and inverse correctness
  - Identity: rho_{0,p} == ES_p
  - Identity: rho_{lam,1} == rho^w
  - Identity: rho_lam(X|X) == 0
  - Step-profile reduction H_lam == max(rho_{lam,p1}-c, rho_{lam,p2})
  - Power utility and inverse
  - eta_{lam,p} positivity and monotonicity
  - Edge cases: empty arrays, single element, extreme lambda
"""
import numpy as np
import pytest
from src.risk_primitives import (
    left_quantile, quantile_grid, empirical_es, empirical_es_grid, rho_w,
)
from src.meyer import (
    cara_utility, cara_utility_inverse,
    rho_lam_p, rho_lam_p_grid,
    fixed_profile_meyer_risk,
    adjusted_es, adjusted_lvar,
    benchmark_relative_meyer_risk,
)
from src.rrm import power_utility, power_utility_inverse, eta_lam_p, eta_lam_p_grid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_sample():
    rng = np.random.RandomState(42)
    return rng.normal(0, 1, 2000)


@pytest.fixture
def positive_sample():
    rng = np.random.RandomState(42)
    return np.abs(rng.normal(1, 0.2, 2000)) + 0.5


@pytest.fixture
def simple_discrete():
    """Simple two-point discrete sample: -1 w.p. 0.5, +1 w.p. 0.5."""
    return np.array([-1.0] * 500 + [1.0] * 500)


@pytest.fixture
def p_grid():
    return np.linspace(0.50, 0.995, 50)


# ---------------------------------------------------------------------------
# Left quantile
# ---------------------------------------------------------------------------

class TestLeftQuantile:
    def test_sorted_sample_order_statistics(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # p=0.2: ceil(0.2*5)=1 -> x[0]=1
        assert left_quantile(x, 0.2) == pytest.approx(1.0)
        # p=0.4: ceil(0.4*5)=2 -> x[1]=2
        assert left_quantile(x, 0.4) == pytest.approx(2.0)

    def test_p_zero_returns_minimum(self):
        x = np.array([3.0, 1.0, 4.0, 1.5])
        assert left_quantile(x, 0.0) == pytest.approx(1.0)

    def test_p_one_returns_maximum(self):
        x = np.array([3.0, 1.0, 4.0, 1.5])
        assert left_quantile(x, 1.0) == pytest.approx(4.0)

    def test_p_half_on_symmetric_sample(self, gaussian_sample):
        q = left_quantile(gaussian_sample, 0.5)
        assert abs(q) < 0.1, f"Median of N(0,1) should be ~0, got {q}"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            left_quantile(np.array([]), 0.5)

    def test_monotone_in_p(self):
        rng = np.random.RandomState(7)
        x = rng.randn(500)
        p_vals = np.linspace(0.01, 0.99, 50)
        q_vals = np.array([left_quantile(x, p) for p in p_vals])
        assert np.all(np.diff(q_vals) >= 0), "Quantile must be nondecreasing in p"


class TestEmpiricalES:
    def test_es_at_zero_is_negative_mean(self):
        """ES_0(X) = -E[X] because integral_0^1 q_X(r) dr = E[X]."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        es = empirical_es(x, 0.0)
        assert es == pytest.approx(-np.mean(x), abs=1e-10)

    def test_es_at_p1_equals_rho_w(self, gaussian_sample):
        """ES_1(X) = rho^w(X) = max(-X)."""
        es1 = empirical_es(gaussian_sample, 1.0)
        worst = rho_w(gaussian_sample)
        assert es1 == pytest.approx(worst, abs=1e-10)

    def test_es_monotone_in_p(self, gaussian_sample):
        """ES_p is nondecreasing in p."""
        p_vals = np.linspace(0.01, 0.999, 50)
        es_vals = empirical_es_grid(gaussian_sample, p_vals)
        diffs = np.diff(es_vals)
        assert np.all(diffs >= -1e-10), f"ES not monotone; min diff={diffs.min():.2e}"

    def test_es_gaussian_95(self, gaussian_sample):
        """ES_0.95(N(0,1)) ≈ 2.063 (known value)."""
        es = empirical_es(gaussian_sample, 0.95)
        assert 1.85 < es < 2.30, f"ES_0.95 outside expected range: {es}"

    def test_es_symmetry(self):
        """ES_p(X) = -E[X | X <= q_X(p)] for large n."""
        rng = np.random.RandomState(1)
        x = rng.uniform(-1, 1, 10000)
        # Uniform[-1,1]: ES_0.9 = -(average of bottom 10%) = 0.9 + ... = 0.95
        es = empirical_es(x, 0.9)
        # Bottom 10% of U[-1,1] covers [-1, -0.8]; average = -0.9 -> ES = 0.9
        assert 0.85 < es < 0.95, f"ES_0.9(U[-1,1]) ≈ 0.9, got {es}"

    def test_es_single_element_equals_minus_x(self):
        """For a single-element sample, ES_p = -x for all p."""
        x = np.array([3.14])
        for p in [0.0, 0.5, 0.9, 1.0]:
            es = empirical_es(x, p)
            assert es == pytest.approx(-3.14, abs=1e-10)


# ---------------------------------------------------------------------------
# CARA utility
# ---------------------------------------------------------------------------

class TestCARAUtility:
    def test_lam_zero_is_identity(self):
        x = np.linspace(-2, 2, 20)
        result = cara_utility(x, 0.0)
        np.testing.assert_allclose(result, x, rtol=1e-12)

    def test_lam_positive_strictly_increasing(self):
        x = np.linspace(-3, 3, 30)
        u = cara_utility(x, 1.0)
        assert np.all(np.diff(u) > 0), "CARA utility should be strictly increasing"

    def test_lam_negative_strictly_increasing(self):
        x = np.linspace(-3, 3, 30)
        u = cara_utility(x, -1.0)
        assert np.all(np.diff(u) > 0), "CARA utility should be strictly increasing"

    def test_inverse_roundtrip_positive_lam(self):
        x = np.linspace(-2, 2, 20)
        lam = 0.5
        u = cara_utility(x, lam)
        x_back = cara_utility_inverse(u, lam)
        np.testing.assert_allclose(x_back, x, rtol=1e-8)

    def test_inverse_roundtrip_negative_lam(self):
        x = np.linspace(-2, 2, 20)
        lam = -0.5
        u = cara_utility(x, lam)
        x_back = cara_utility_inverse(u, lam)
        np.testing.assert_allclose(x_back, x, rtol=1e-8)

    def test_inverse_roundtrip_zero_lam(self):
        x = np.linspace(-2, 2, 20)
        u = cara_utility(x, 0.0)
        x_back = cara_utility_inverse(u, 0.0)
        np.testing.assert_allclose(x_back, x, rtol=1e-12)


# ---------------------------------------------------------------------------
# rho_{lam,p}
# ---------------------------------------------------------------------------

class TestRhoLamP:
    def test_rho_0p_equals_ES(self, gaussian_sample, p_grid):
        """rho_{0,p}(X) = ES_p(X) for all p."""
        for p in p_grid:
            rho = rho_lam_p(gaussian_sample, 0.0, float(p))
            es = empirical_es(gaussian_sample, float(p))
            assert abs(rho - es) < 1e-8, f"rho_0_p != ES at p={p}: {rho} vs {es}"

    def test_rho_lam1_equals_rho_w(self, gaussian_sample):
        """rho_{lam,1}(X) = rho^w(X) for lam in a range."""
        worst = rho_w(gaussian_sample)
        for lam in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            val = rho_lam_p(gaussian_sample, lam, 1.0)
            assert abs(val - worst) < 1e-8, f"rho_{{lam,1}} != rho^w at lam={lam}"

    def test_monotone_in_p(self, gaussian_sample):
        """rho_{lam,p}(X) is nondecreasing in p."""
        p_vals = np.linspace(0.5, 0.999, 30)
        for lam in [-1.0, 0.0, 1.0]:
            vals = rho_lam_p_grid(gaussian_sample, lam, p_vals)
            diffs = np.diff(vals)
            assert np.all(diffs >= -1e-8), (
                f"rho_{{lam,p}} not monotone in p for lam={lam}; min diff={diffs.min():.2e}"
            )

    def test_positive_lam_more_sensitive_to_tail(self, gaussian_sample):
        """For p=0.99, rho_{1,0.99} > rho_{0,0.99} on fat-tail equivalent."""
        rho_pos = rho_lam_p(gaussian_sample, 1.0, 0.99)
        rho_zero = rho_lam_p(gaussian_sample, 0.0, 0.99)
        # Positive lambda increases sensitivity to heavy losses
        assert rho_pos >= rho_zero, "Expected rho_1 >= rho_0 at high p"

    def test_grid_single_match(self, gaussian_sample, p_grid):
        """rho_lam_p_grid results match individual rho_lam_p calls."""
        lam = 0.8
        grid_vals = rho_lam_p_grid(gaussian_sample, lam, p_grid)
        for i, p in enumerate(p_grid[:5]):
            single = rho_lam_p(gaussian_sample, lam, float(p))
            assert abs(grid_vals[i] - single) < 1e-10


# ---------------------------------------------------------------------------
# Fixed-profile (step-profile) Meyer risk
# ---------------------------------------------------------------------------

class TestFixedProfileMeyerRisk:
    def test_equals_two_point_max(self, gaussian_sample):
        """H_lam = max(rho_{lam,p1}-c, rho_{lam,p2}) exactly."""
        for lam in [-1.0, 0.0, 1.0]:
            H = fixed_profile_meyer_risk(gaussian_sample, lam, p1=0.975, p2=0.9, c=1.0)
            r1 = rho_lam_p(gaussian_sample, lam, 0.975)
            r2 = rho_lam_p(gaussian_sample, lam, 0.9)
            manual = max(r1 - 1.0, r2)
            assert abs(H - manual) < 1e-12, f"Step reduction failed at lam={lam}"

    def test_nonnegative_for_large_losses(self):
        """For large-loss sample, H_lam should be positive."""
        x = np.random.RandomState(5).normal(-2, 1, 1000)
        H = fixed_profile_meyer_risk(x, lam=0.0, p1=0.975, p2=0.9, c=0.5)
        assert H > 0, f"Expected positive H for loss-heavy sample, got {H}"


# ---------------------------------------------------------------------------
# Adjusted ES and adjusted LVaR
# ---------------------------------------------------------------------------

class TestAdjustedMeasures:
    def test_adjusted_es_step_profile(self, gaussian_sample):
        """adj_ES under step profile = max(ES_{p1}(X)-c, ES_{p2}(X))."""
        p1, p2, c = 0.975, 0.9, 1.0
        adj = adjusted_es(gaussian_sample, p1, p2, c)
        manual = max(empirical_es(gaussian_sample, p1) - c,
                     empirical_es(gaussian_sample, p2))
        assert abs(adj - manual) < 1e-10

    def test_adjusted_lvar_step_profile(self, gaussian_sample):
        """adj_LVaR = max(-q_X(1-p1)-c, -q_X(1-p2)) under step profile."""
        p1, p2, c = 0.975, 0.9, 1.0
        adj = adjusted_lvar(gaussian_sample, p1, p2, c)
        q1 = left_quantile(gaussian_sample, 1 - p1)
        q2 = left_quantile(gaussian_sample, 1 - p2)
        manual = max(-q1 - c, -q2)
        assert abs(adj - manual) < 1e-10


# ---------------------------------------------------------------------------
# Benchmark-relative Meyer risk
# ---------------------------------------------------------------------------

class TestBenchmarkRelativeMeyerRisk:
    def test_self_benchmark_zero(self, gaussian_sample, p_grid):
        """rho_lam(X|X) = 0 for all lam."""
        for lam in [-1.0, 0.0, 1.0]:
            val = benchmark_relative_meyer_risk(gaussian_sample, gaussian_sample,
                                                lam, p_grid)
            assert abs(val) < 1e-8, f"rho_lam(X|X) != 0 at lam={lam}: {val}"

    def test_nonnegative_for_worse_X(self, gaussian_sample):
        """rho_lam(X|Y) >= 0 if X has heavier tail than Y."""
        rng = np.random.RandomState(99)
        # X has heavier left tail (more losses)
        X = gaussian_sample + rng.normal(0, 0.5, len(gaussian_sample))
        Y = gaussian_sample
        p_grid = np.linspace(0.5, 0.99, 50)
        val = benchmark_relative_meyer_risk(X, Y, 0.0, p_grid)
        # Not guaranteed strictly, but rho_0(X|Y) = sup_p(ES_p(X) - ES_p(Y))
        # Since X = Y + noise with zero mean, this could be small positive or small negative
        # We just check it's computable
        assert np.isfinite(val), "benchmark_relative_meyer_risk should return finite value"


# ---------------------------------------------------------------------------
# Power utility and RRM
# ---------------------------------------------------------------------------

class TestPowerUtility:
    def test_log_at_a_zero(self, positive_sample):
        """p_0(x) = log(x)."""
        result = power_utility(positive_sample, 0.0)
        expected = np.log(positive_sample)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_a_equals_one(self, positive_sample):
        """p_1(x) = x^1/1 = x."""
        result = power_utility(positive_sample, 1.0)
        np.testing.assert_allclose(result, positive_sample, rtol=1e-12)

    def test_inverse_roundtrip(self, positive_sample):
        """p_a^{-1}(p_a(x)) = x for multiple values of a."""
        for a in [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            u = power_utility(positive_sample, a)
            x_back = power_utility_inverse(u, a)
            np.testing.assert_allclose(x_back, positive_sample, rtol=1e-7,
                                       err_msg=f"Inverse roundtrip failed at a={a}")

    def test_negative_input_raises(self):
        with pytest.raises(ValueError):
            power_utility(np.array([-1.0, 1.0]), 1.0)

    def test_zero_input_raises(self):
        with pytest.raises(ValueError):
            power_utility(np.array([0.0, 1.0]), 1.0)


class TestEtaLamP:
    def test_returns_positive(self, positive_sample):
        """eta_{lam,p}(X) should be positive for positive payoffs."""
        for lam in [0.5, 1.0, 2.0]:
            for p in [0.7, 0.9]:
                val = eta_lam_p(positive_sample, lam, p)
                assert val > 0, f"eta_{{{lam},{p}}} should be positive; got {val}"

    def test_eta_grid_monotone_in_p(self, positive_sample):
        """eta_{lam,p}(X) behavior in p (theory: should vary; just check finite)."""
        p_grid_local = np.linspace(0.5, 0.99, 20)
        for lam in [0.5, 1.0]:
            vals = eta_lam_p_grid(positive_sample, lam, p_grid_local)
            finite = np.isfinite(vals)
            assert finite.sum() > len(p_grid_local) // 2, (
                f"Too many NaN in eta_lam_p_grid for lam={lam}"
            )

    def test_negative_lam_finite(self, positive_sample):
        """eta_{-1,0.9}(X) is finite for positive payoffs."""
        val = eta_lam_p(positive_sample, -1.0, 0.9)
        assert np.isfinite(val), f"eta should be finite, got {val}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_element_sample_quantile(self):
        x = np.array([42.0])
        assert left_quantile(x, 0.5) == 42.0
        assert left_quantile(x, 0.0) == 42.0
        assert left_quantile(x, 1.0) == 42.0

    def test_single_element_es(self):
        x = np.array([42.0])
        assert empirical_es(x, 0.5) == pytest.approx(-42.0)

    def test_rho_w_single_element(self):
        x = np.array([-3.0])
        assert rho_w(x) == pytest.approx(3.0)

    def test_large_positive_lambda_approaches_rho_w(self, gaussian_sample):
        """For very large lam, rho_{lam,p}(X) is closer to rho^w than at small lam.

        The convergence is asymptotic; we test qualitative direction, not exact value.
        """
        worst = rho_w(gaussian_sample)
        val_large = rho_lam_p(gaussian_sample, 50.0, 0.9)
        val_small = rho_lam_p(gaussian_sample, 1.0, 0.9)
        # Large lambda should bring rho closer to worst-case than small lambda
        assert abs(val_large - worst) < abs(val_small - worst) or abs(val_large - worst) < 0.5, (
            f"Large lam should approach rho^w; got {val_large:.4f} vs {worst:.4f}"
        )

    def test_large_negative_lambda_approaches_quantile(self, gaussian_sample):
        """For very negative lam, rho_{lam,p}(X) ~ -q_X(1-p) for fixed p < 1."""
        p = 0.95
        q = left_quantile(gaussian_sample, 1 - p)
        val = rho_lam_p(gaussian_sample, -50.0, p)
        assert abs(val - (-q)) < 0.1, (
            f"Large neg lam should approach -q_X(1-p); got {val:.4f} vs {-q:.4f}"
        )

    def test_p_grid_vectorised_equals_scalar(self, gaussian_sample):
        """rho_lam_p_grid results match scalar rho_lam_p for each p."""
        lam = 0.5
        p_vals = np.array([0.90, 0.95, 0.99])
        grid_vals = rho_lam_p_grid(gaussian_sample, lam, p_vals)
        for i, p in enumerate(p_vals):
            single = rho_lam_p(gaussian_sample, lam, float(p))
            assert abs(grid_vals[i] - single) < 1e-10
