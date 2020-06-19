# %% [markdown]
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
    if (b-a) < 0:
        raise ValueError('Lower bound is greater then upper bound!')
    elif sigma <= 0:
        raise ValueError('Sigma is <= 0!')
    
    PHIl = 0.5 * special.erfc(-(((a - mu) / sigma) / sqrt(2)))
    PHIr = 0.5 * special.erfc(-(((b - mu) / sigma) / sqrt(2)))
    print(PHIl, PHIr)
    
    # Refer to http://www.maths.uq.edu.au/~chancc/10Fstat3001/ass4sol.pdf for truncated normal dist sampling below -- If this source does not exist then refer to code in the link below,
    # http://www.wiley.com/legacy/wileychi/koopbayesian/supp/normt_rnd.m
    return mu + sigma * (sqrt(2) * special.erfinv(2 * (PHIl + (PHIr - PHIl) * rand(int(n), 1)) - 1))


# %% [markdown]
# ## Testing

# %%
def test_truncated_normal(A=-1.0, B=1.0, mu=0.1, sigma=0.001, N=10000.0):
    
    # numpy
    import numpy as np 
    
    # package enabling access/control of matlab from python
    import matlab.engine
    
    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()
    
    # paths to matlab helper and model functions
    eng.addpath('../original')
    
    # generate output
    python_output = truncated_normal(A, B, mu, sigma, N)
    matlab_output = np.asarray(eng.truncated_normal(A, B, mu, sigma, N))
    
    # result is stochastic, so we only test for the same shape
    assert np.shape(python_output) == np.shape(matlab_output)
    
    # and visualize histograms of output
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    n_bins = 100

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(python_output, bins=n_bins)
    axs[1].hist(matlab_output, bins=n_bins)
    
    print('All tests passed!')


# %%
# run tests only when this is main file!
if __name__ == '__main__':
    test_truncated_normal()

# %%
