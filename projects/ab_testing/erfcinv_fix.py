"""Provide a portable `erfcinv` implementation for A/B testing notebooks."""

try:
    from scipy.special import erfcinv
except ImportError:
    # Fallback to erfinv if erfcinv not available
    import numpy as np
    from scipy.special import erfinv

    def erfcinv(x):
        """Approximate the complementary inverse error function."""
        return -np.sqrt(2) * erfinv(1 - x)
