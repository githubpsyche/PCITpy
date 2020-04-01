# # logsumexp
# Return log(sum(exp(a), dim)) while avoiding numerical underflow (at least in the `MATLAB` implementation).
#
# - Default is dim = 1 (rows) returns a row vector.  
# - `logsumexp(a, 1)` will sum across columns and return a column vector.  
# - Unlike matlab's `sum`, it will not switch the summing direction if you provide a row vector.

from scipy.special import logsumexp


# ## Testing

def test_logsumexp(A=None, dim=None):
    # numpy
    import numpy as np 
    
    # package enabling access/control of matlab from python
    import matlab.engine
    
    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()
    
    # paths to matlab helper and model functions
    eng.addpath('../original')
    
    # example parameters
    if A is None:
        A = [[1.7, 1.2, 1.5],[1.3, 1.6, 1.99]]
    if dim is None:
        dim = 2
        
    # generate output
    python_output = logsumexp(A, dim-1)
    matlab_output = np.asarray(eng.logsumexp(matlab.double(A), dim)).reshape(-1)
    assert np.all(python_output == matlab_output)
    
    print('All tests passed!')


# run tests only when is main file!
if __name__ == '__main__':
    test_logsumexp()
