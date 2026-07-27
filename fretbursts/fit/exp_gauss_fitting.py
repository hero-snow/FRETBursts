from scipy.special import erfc


def exp_gauss_pdf(x, mu, sig, lamb):
    return 0.5*lamb*exp(0.5*lamb*(2*mu+lamb*sig**2-2*x)) * \
            erfc((mu+lamb*sig**2-x)/(sqrt(2)*sig))
