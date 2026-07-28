
def Fit(values_mch, fitfun, **kwargs):
    """Multi-channel fit (ex. d.E, d.nt) with the fitfun (ex. gaussian_fit)."""
    return [fitfun(v, **kwargs) for v in values_mch]

