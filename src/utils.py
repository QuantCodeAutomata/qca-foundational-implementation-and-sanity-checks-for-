"""Utility functions for numerical stability in risk computations."""
import numpy as np


def safe_log(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Compute log with lower-bound clipping to avoid -inf."""
    return np.log(np.maximum(x, eps))


def safe_exp(x: np.ndarray, clip: float = 500.0) -> np.ndarray:
    """Compute exp with clipping to avoid overflow."""
    return np.exp(np.clip(x, -clip, clip))


def safe_divide(num: np.ndarray, den: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Element-wise division with safe handling of zero denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(den) > 0, num / den, fill)
    return result
