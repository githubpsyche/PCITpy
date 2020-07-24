# %% markdown
# # scale_data
# Scale the elements of all the column vectors in Data within the range of [Lower Upper]; default range is [-1 1]
#
# **USAGE**:
# [SCALED] = SCALE_DATA(DATA, LOWER, UPPER)
#
# **INPUTS**:
# - Data: data, numeric vector e.g. [8.3256, 1000, 23, 564]
# - Lower: lower range, numeric e.g. 0
# - Upper: upper range, numeric e.g. 1
#
# **OUTPUTS**:
# - scaled: scaled data e.g. [0, 0.1760, 0.0026, 1.0000]
# %%
import numpy as np


def scale_data(data, lower=-1.0, upper=1.0):
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


# %% markdown
# ## Testing
# Compares result of applying `scale_data` to an example list `[8.3256, 1000.0, 23.0, 564.0]` between codebase versions.

# %%
def test_scale_data(data=None, lower=0.0, upper=1.0):

    # package enabling access/control of matlab from python
    import matlab.engine

    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()

    # paths to matlab helper and model functions
    eng.addpath('../original')

    # set data parameter if undefined
    if data is None:
        data = [8.3256, 1000.0, 23.0, 564.0]

    # generate output
    python_output = scale_data(data, lower, upper)
    matlab_output = np.asarray(eng.scale_data(matlab.double(data), lower, upper))
    assert np.all(python_output == matlab_output)

    print('All tests passed!')


# %%
# run tests only when this is main file!
if __name__ == '__main__':
    test_scale_data()
