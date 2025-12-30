
# Add this to notebooks using erfcinv:
try:
    from scipy.special import erfcinv
except ImportError:
    # Fallback to erfinv if erfcinv not available
    from scipy.special import erfinv
    import numpy as np
    def erfcinv(x):
        return -np.sqrt(2) * erfinv(1 - x)
