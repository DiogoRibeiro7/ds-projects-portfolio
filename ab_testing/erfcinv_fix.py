# Add this to notebooks using erfcinv:
try:
    from scipy.special import erfcinv
except ImportError:
    # Fallback to erfinv if erfcinv not available
    import numpy as np
    from scipy.special import erfinv

    def erfcinv(x):
        return -np.sqrt(2) * erfinv(1 - x)
