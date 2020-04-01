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

# +
from scipy import stats

def likratiotest(L1, L0, K1, K0):
    D = -2*(L0-L1)            # deviance score
    df = K1-K0                # degrees of freedom
    p = stats.chi2.sf(D, df)  # chi-square test
    return D, p


# -

# ## Testing Code

def test_likratiotest(L1=10.0, L0=20.0, K1=8.0, K0=5.0):
    
    # numpy
    import numpy as np 
    
    # package enabling access/control of matlab from python
    import matlab.engine
    
    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()
    
    # paths to matlab helper and model functions
    eng.addpath('../original')
    
    # generate output
    python_output = likratiotest(L1, L0, K1, K0)
    matlab_output = np.asarray(eng.likratiotest(L1, L0, K1, K0, nargout=2))
    assert np.all(python_output == matlab_output)
    
    print('All tests passed!')


# run tests only when is main file!
if __name__ == '__main__':
    test_likratiotest()
