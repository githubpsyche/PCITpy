# %% [markdown]
# # round_to
# Round the number to the no. of digits
#
# **USAGE**:
# [ROUNDED_NUMBER] = ROUND_TO(NUMBER, DIGITS)
#
# **INPUTS**:
# - number: The number to be rounded, numeric e.g. 6.784577854
# - digits: The number of digits to be rounded to, numeric e.g. 4
#
# **OUTPUTS**:
# - rounded_number: The rounded number, numeric e.g. 6.7846

# %%
import numpy as np

def round_to(number, digits=1):
    digits = np.power(10, digits)
    return np.round(number * digits) / digits

# %% [markdown]
# ## Testing
# Assumes kernel is active at the base of the project directory.

# %%
def test_round_to(number=6.784577854, digits=4.0):

    # numpy
    import numpy as np

    # package enabling access/control of matlab from python
    import matlab.engine

    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()

    # paths to matlab helper and model functions
    eng.addpath('original')

    # generate output
    python_output = round_to(number, digits)
    matlab_output = np.asarray(eng.round_to(number, digits))
    assert np.all(python_output == matlab_output)

    print('All tests passed!')

# %%
# run tests only when is main file!
if __name__ == '__main__':
    test_round_to()
