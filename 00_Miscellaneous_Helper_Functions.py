# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# hide
# default_exp helpers
from nbdev.showdoc import *

# # Additional Helper Functions
# > Miscellaneous functions reused across the toolbox to carry out simple numerical operations

# **Note:** A few helper functions included in the MATLAB version of the PCIT toolbox are not present here. When it comes to numerical helper functions `logsumexp.m` and `round_to.m`, this is because a Python version of the function is already included in a required scientific programming library such as `scipy` or `numpy`. When it comes to visualization helper functions `savesamesize.m` and `jbfill.m`, it's because equivalent Python functions are either unnecessary or not applicable in the context of `matplotlib`'s library. In either case, equivalent functionality is achieved without implementation of an additional toolbox function.

# +
# export
# hide
from scipy import stats

def likratiotest(l1, l0, k1, k0):
    """Performs the likelihood ratio test for nested models.
      
    **Arguments**:  
    - L1: log-likelihood for alternative model
    - L0: log-likelihood for the null model
    - K1: number of parameters for alternative model
    - K0: number of parameters for null model (K1 > K0)
    
    **Returns**:  
    - D: deviate score: -2*log(L1-L0)
    - p: p-value from chi-squared test with degrees of freedom = K1-K0
    """
    D = -2 * (l0 - l1)  # deviance score
    df = k1 - k0  # degrees of freedom
    p = stats.chi2.sf(D, df)  # chi-square test
    return D, p


# -

show_doc(likratiotest, title_level=2)

# This function implements the $\beta_1$ likelihood-ratio test described in Section 4.10 of the [P-CIT Toolbox Manual](https://github.com/PrincetonUniversity/p-cit-toolbox). 
#
# If we want to build confidence that this function is an exact reproduction of the MATLAB implementation, we can ask whether our version returns `(-20, 1)` for the arguments `(l1=10.0, l0=20.0, k1=8.0, k0=5.0)` as the MATLAB version does:

likratiotest(l1=10.0, l0=20.0, k1=8.0, k0=5.0)

# `(-20.0, 1.0)`

# +
# export
# hide
from scipy import special
from numpy.random import rand
from math import sqrt


def truncated_normal(a, b, mu, sigma, n):
    """Generates N samples from a truncated normal distribution with mean=mu, 
    sigma=sigma and with bounds A and B.
    
    We use this function in our toolbox to add noise from a truncated Gaussian 
    distribution.
    
    **Arguments**:  
    - A: lower bound
    - B: upper bound
    - mu: mean
    - sigma: sigma, standard deviation
    - N: number of samples
    
    **Returns** array of length N containing mean + truncated Gaussian noise with 
    mean=mu, sigma=sigma, between bounds A and B
    """
    
    if (b - a) < 0:
        raise ValueError('Lower bound is greater then upper bound!')
    elif sigma <= 0:
        raise ValueError('Sigma is <= 0!')

    phi_l = 0.5 * special.erfc(-(((a - mu) / sigma) / sqrt(2)))
    phi_r = 0.5 * special.erfc(-(((b - mu) / sigma) / sqrt(2)))

    # Refer to http://www.maths.uq.edu.au/~chancc/10Fstat3001/ass4sol.pdf for truncated normal dist sampling below --
    # If this source does not exist then refer to code in the link below, 
    # http://www.wiley.com/legacy/wileychi/koopbayesian/supp/normt_rnd.m 
    return mu + sigma * (sqrt(2) * special.erfinv(2 * (phi_l + (phi_r - phi_l) * rand(int(n))) - 1))


# -

show_doc(truncated_normal, title_level=2)

# As an example, we can generate and visualize the result of applying the function to some arbitrary paremeters:

# +
import matplotlib.pyplot as plt

# generate a sample
sample = truncated_normal(a=.7, b=1.3, mu=1.0, sigma=.1, n=10000.0)

# visualize its distribution
n_bins = 100
plt.hist(sample, bins=n_bins)
plt.title('truncated_normal(a=.7, b=1.3, mu=1.0, sigma=.1, n=10000.0)')
plt.savefig('figures/truncated_normal_example.svg')
plt.show()

sample
# -

# ![](figures/truncated_normal_example.svg)
# ```
# array([0.88296168, 1.22801193, 0.8565991 , ..., 0.87280798, 1.09364587,
#        0.96110738])
# ```

# +
# export
# hide
import numpy as np


def scale_data(data, lower=-1.0, upper=1.0):
    """Scale the elements of all the column vectors in Data within the range 
    of [Lower Upper]; default range is [-1 1]
    
    We use this function in our toolbox to scale the predictor variable 
    between 0 and 1. 

    **Arguments**:  
    - Data: data, numeric vector
    - Lower: lower range, numeric
    - Upper: upper range, numeric

    **Returns**:
    - scaled: 1xN array containing scaled data
    """
    if lower > upper:
        raise ValueError('Wrong lower of upper values!')

    max_v = np.amax(data, axis=0)
    min_v = np.amin(data, axis=0)
    shape = np.shape(data)

    if len(shape) < 2:
        r, c = 1, shape[0]
    elif len(shape) > 2:
        r, c = shape[0], np.prod(shape[1:])
    else:
        r, c = shape

    scaled = ((data - np.ones((r, 1)) * min_v) * (
            np.ones((r, 1)) * (
            (upper - lower) * (
            np.ones((1, c)) / (max_v - min_v))))) + lower

    return scaled


# -

show_doc(scale_data, title_level=2)

# As an example we can execute the function with a few arbitrary parameters:

scale_data([8.3256, 1000.0, 23.0, 564.0], 0, 1)

# As specified, it returns the vector scaled between 0 and 1.
#
# ```array([[0.        , 1.        , 0.0147976 , 0.56033956]])```
#
# Notably, the output is 2-dimensional rather than a vector; this enforces consistency in the shape of the function's output regardless of the shape of the input. It also helps ensure compatibility with other code translated from the MATLAB implementation of the toolbox.
