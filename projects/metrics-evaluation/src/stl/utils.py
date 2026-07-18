import numpy as np

def robust_std(residuals: np.ndarray) -> float:
    """MAD-based robust std, more resistant to outliers skewing the noise band itself."""
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return 1.4826 * mad if mad > 0 else (np.std(residuals) or 1e-9)