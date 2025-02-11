# %% markdown
# # likratiotest
# Likelihood ratio test for nested models.
#
# **USAGE**:
# [D p] = likratiotest(L1,L0,K1,K0)
#
# **INPUTS**:
# - L1 - log-likelihood for alternative model
# - L0 - log-likelihood for null model
# - K1 - number of parameters for alternative model
# - K0 - number of parameters for null model (K1>K0)
#
# **OUTPUTS**:
# - D - deviance score: -2*log(L1-L0)
# - p - p-value from chi-squared test with degrees of freedom = K1-K0

# %%
from scipy import stats


def likratiotest(l1, l0, k1, k0):
    D = -2 * (l0 - l1)  # deviance score
    df = k1 - k0  # degrees of freedom
    p = stats.chi2.sf(D, df)  # chi-square test
    return D, p


# %% markdown
# ## Testing
# In this test, arbitrary parameters are selected and output under each version is compared.

# %%
def test_likratiotest(l1=10.0, l0=20.0, k1=8.0, k0=5.0):
    # numpy
    import numpy as np

    # package enabling access/control of matlab from python
    import matlab.engine

    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()

    # paths to matlab helper and model functions
    eng.addpath('../original')

    # generate output
    python_output = likratiotest(l1, l0, k1, k0)
    matlab_output = np.asarray(eng.likratiotest(l1, l0, k1, k0, nargout=2))
    assert np.all(python_output == matlab_output)

    print('All tests passed!')


# %%
# run tests only when is main file!
if __name__ == '__main__':
    test_likratiotest()
