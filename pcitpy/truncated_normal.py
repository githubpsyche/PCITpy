# %% markdown
# # truncated_normal
# generate N samples from a truncated normal distribution with mean=mu, sigma=sigma and with bounds A and B
#
# **USAGE**:
# [out] = truncated_normal(a, b, mu, sigma, n)
#
# **INPUTS**:
# - A: lower bound
# - B: upper bound
# - mu: mean
# - sigma: sigma, standard deviation
# - N: number of samples
#
# **OUTPUTS**:
# - out: mean + truncated Gaussian noise with mean=mu, sigma=sigma, between bounds A and B
#
# **EXAMPLE USAGE**:
# - truncated_normal(-1, 1, 0.1, 0.001, 1000)

# %%
from scipy import special
from numpy.random import rand
from math import sqrt


def truncated_normal(a, b, mu, sigma, n):
    if (b - a) < 0:
        raise ValueError('Lower bound is greater then upper bound!')
    elif sigma <= 0:
        raise ValueError('Sigma is <= 0!')

    phi_l = 0.5 * special.erfc(-(((a - mu) / sigma) / sqrt(2)))
    phi_r = 0.5 * special.erfc(-(((b - mu) / sigma) / sqrt(2)))

    # Refer to http://www.maths.uq.edu.au/~chancc/10Fstat3001/ass4sol.pdf for truncated normal dist sampling below --
    # If this source does not exist then refer to code in the link below, 
    # http://www.wiley.com/legacy/wileychi/koopbayesian/supp/normt_rnd.m 
    return mu + sigma * (sqrt(2) * special.erfinv(2 * (phi_l + (phi_r - phi_l) * rand(int(n), 1)) - 1))


# %% markdown 
# ## Testing 
# Tests are passed if histograms exhibit similar (normal) distributions with the specified 
# mean and sigma and truncations. 

# %%
def test_truncated_normal(a=-1.0, b=1.0, mu=0.1, sigma=0.001, n=10000.0):
    # numpy
    import numpy as np

    # package enabling access/control of matlab from python
    import matlab.engine

    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()

    # paths to matlab helper and model functions
    eng.addpath('../original')

    # generate output
    python_output = truncated_normal(a, b, mu, sigma, n)
    matlab_output = np.asarray(eng.truncated_normal(a, b, mu, sigma, n))

    # result is stochastic, so we only test for the same shape
    assert np.shape(python_output) == np.shape(matlab_output)

    # and visualize histograms of output
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    n_bins = 100

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(python_output, bins=n_bins)
    axs[1].hist(matlab_output, bins=n_bins)

    print('Tests are passed if histograms exhibit similar (truncated normal) distributions!')


# %%
# run tests only when this is main file!
if __name__ == '__main__':
    test_truncated_normal()

# %%
