# %% markdown
# # common_to_all_curves
# To perform sanity checks on the input data and the algorithm parameter struct.
# Massage the data (i.e. drop outliers, zscore data, etc)
#
# **USAGE**:
# - common_to_all_curves(CURVE_TYPE, GET_INFO, VARARGIN)
#
# **INPUTS**:
# - curve_type: Family of curves, string, e.g. 'free_rmpw'
# - get_info: Cues for specific information / computation, string, e.g. 'initial_sampling'
# - varargin: Has arguments depending on the computation
#
# **OUTPUTS**:
# - out: the output of all computations
#
# **EXAMPLE USAGE**:
# - common_to_all_curves('horz_indpnt', 'initial_sampling', 1000, 4)
# - common_to_all_curves('horz_indpnt', 'check_if_exceed_bounds', some_matrix)
# - common_to_all_curves('horz_indpnt', 'curve_volumes', 5)
# - common_to_all_curves('horz_indpnt', 'flip_vertical_params', some_matrix)
# - common_to_all_curves('horz_indpnt', 'sort_horizontal_params', some_matrix)
# - common_to_all_curves('horz_indpnt', 'draw_bcm_curves', [0.2, 0.5, 0.1, 0.1, 0.1, 0.1], 4)
# - common_to_all_curves('horz_indpnt', 'auto_generate', 'con', 4)
# - common_to_all_curves('horz_indpnt', 'weighted_curve', importance_sampler_mat, 0.9, 4)

# %%
import numpy as np
from family_of_curves import family_of_curves
import matplotlib.pyplot as plt


def common_to_all_curves(curve_type, get_info, *varargin):
    # Checks if input arguments are passed in
    if len(varargin) == 0:
        raise ValueError('No input arguments!')

    if get_info == 'initial_sampling':
        if len(varargin) < 2:
            raise ValueError('Missing input parameters in {} computation!'.format(get_info))

        nParticles = varargin[0]
        if nParticles <= 0:
            raise ValueError('Number of particles will need to > 0!')

        resolution = varargin[1]
        if resolution <= 0:
            raise ValueError('Resolution will need to > 0!')

        bounds = family_of_curves(curve_type, 'get_bounds')
        nParams = family_of_curves(curve_type, 'get_nParams')
        out = np.full((nParticles, nParams), np.nan)

        # Uniform sampling each curve parameter bounded by its respective bounds
        for i in range(nParams):
            out[:, i] = np.random.uniform(low=bounds[i, 0], high=bounds[i, 1], size=(nParticles))

        out = np.round_(out, resolution)

        if np.any(np.isnan(out)):
            raise ValueError('NaNs in initial sampling output matrix!')

    elif get_info == 'check_if_exceed_bounds':
        if len(varargin) < 1:
            raise ValueError('Missing input parameters in {} computation!'.format(get_info))

        out = varargin[0]
        nParams = family_of_curves(curve_type, 'get_nParams')
        if (out.size == 0) or (np.shape(out)[1] != nParams):
            raise ValueError('Not a valid input matrix!')

        bounds = family_of_curves(curve_type, 'get_bounds')
        nParams = family_of_curves(curve_type, 'get_nParams')

        # If a curve parameter is found to exceeding bounds then it is set to the bounds
        # E.g. if a vertical parameter is -1.02 then it is set to -1 since -1 is the lower bound for vertical parameters
        for i in range(nParams):
            out[:, i] = np.maximum(out[:, i], bounds[i, 0])
            out[:, i] = np.maximum(out[:, i], bounds[i, 1])

    elif get_info == 'curve_volumes':
        if len(varargin) < 1:
            raise ValueError('Missing input parameters in {} computation!'.format(get_info))

        resolution = varargin[0]
        if resolution <= 0:
            raise ValueError('Resolution will need to > 0!')

        bounds = family_of_curves(curve_type, 'get_bounds')
        nParams = family_of_curves(curve_type, 'get_nParams')
        total_vol = 1

        # Lebesgue measure http://en.wikipedia.org/wiki/Lebesgue_measure
        for i in range(nParams):
            total_vol = total_vol * len(np.arange(bounds[i, 0], bounds[i, 1], 1 / np.power(10, resolution)))
        out = total_vol

    elif get_info == 'flip_vertical_params':
        if len(varargin) < 1:
            raise ValueError('Missing input parameters in {} computation!'.format(get_info))

        input_params = varargin[0]
        nParams = family_of_curves(curve_type, 'get_nParams')
        if (not input_params) or (np.shape(input_params)[1] != nParams):
            raise ValueError('Not a valid input matrix!')

        out = input_params
        vertical_params = family_of_curves(curve_type, 'get_vertical_params_only')

        # Flipping vertical parameters of the curve. If a y1 = -0.4, flipping it will result in 0.4
        for i in range(len(vertical_params)):
            out[:, vertical_params[i]] = np.multiply(input_params[:, vertical_params[i]], -1)

    elif get_info == 'sort_horizontal_params':
        if len(varargin) < 1:
            raise ValueError('Missing input parameters in {} computation!'.format(get_info))

        input_params = varargin[0]
        nParams = family_of_curves(curve_type, 'get_nParams')
        if (input_params.size == 0) or (np.shape(input_params)[1] != nParams):
            raise ValueError('Not a valid input matrix!')

        out = input_params
        horizontal_params = family_of_curves(curve_type, 'get_horizontal_params_only')
        if len(horizontal_params) != 2:
            raise ValueError('Incorrect horizontal parameters count for {} family of curves'.format(curve_type))

        # This piece of code ensures that x1 <= x2 especially for the horz_indpnt family of curves
        idx = input_params[:, horizontal_params[0]] > input_params[:, horizontal_params[1]]
        out[idx, horizontal_params[0]] = input_params[idx, horizontal_params[1]]
        out[idx, horizontal_params[1]] = input_params[idx, horizontal_params[0]]

        if not np.all(out[:, horizontal_params[0]] <= out[:, horizontal_params[1]]):
            raise ValueError('Horizontal parameter 1 is NOT <= Horizontal parameter 2 in {} family of curves'.format(
                curve_type))

    elif get_info == 'draw_bcm_curve':
        if len(varargin) < 2:
            raise ValueError('Missing input parameters in {} computation!'.format(get_info))
        input_params = varargin[0]
        resolution = varargin[1]
        if resolution <= 0:
            raise ValueError('Resolution will need to > 0!')

        # This draws a BCM curve for you. If you passed in the 'input_params' as 'con' then it randomly draws a
        # theory consistent curve; 'inc' - theory inconsistent curve
        if (input_params == 'con') or (input_params == 'inc'):
            input_params = common_to_all_curves(curve_type, 'auto_generate', input_params, resolution)

        nParams = family_of_curves(curve_type, 'get_nParams')
        if (not input_params) or (np.shape(input_params)[1] != nParams):
            raise ValueError('Not a valid input matrix!')

        # If instead you passed in [y1, x1, x2, y2, y3 and y4] into 'input_params' then it draws a curve directly
        # rather then randomly generating one for you
        out = family_of_curves(curve_type, 'get_curve_xy_vals', input_params)

        fig, ax = plt.subplots()
        ax.plot(out['xval'], out['yval'])

        ax.set(xlabel='Activation', ylabel='Change in Memory Strength',
               title=out['title_string'])
        ax.set_ylim(-1.2, 1.2)
        ax.grid()

    elif get_info == 'auto_generate':
        if len(varargin) < 2:
            raise ValueError('Missing input parameters in {} computation!'.format(get_info))

        input_params = varargin[0]
        resolution = varargin[1]
        if resolution <= 0:
            raise ValueError('Resolution will need to > 0!')

        nSamples = 100
        nParam = family_of_curves(curve_type, 'get_nParams')
        params = np.full((nSamples, nParam), np.nan)
        out = np.full((nParam), np.nan)

        # Generate 100 curves and randomly pick a theory consistent or inconsistent curve depending on the request
        params = common_to_all_curves(curve_type, 'initial_sampling', nSamples, resolution)
        if curve_type == 'horz_indpnt':  # Enforce the right ordering for the horizontal curve parameters i.e. x1 < x2
            params = common_to_all_curves(curve_type, 'sort_horizontal_params', params)

        if np.any(np.isnan(params)):
            raise ValueError('NaNs in curve parameter matrix!')
        params_indices = family_of_curves(curve_type, 'count_particles', params)

        if input_params == 'con':
            th_con_params_indices = np.where(params_indices != 0)  # Finding the theory consistent trial indices
            if len(th_con_params_indices) <= 0:
                raise ValueError('Did not generate any theory consistent indices!')

            # Randomly permuting the th_con trial indices
            th_con_params_indices = th_con_params_indices[np.random.permutation(np.shape(th_con_params_indices)[0])]
            out = params[th_con_params_indices[0], :]  # picking one consistent particle

        elif input_params == 'inc':
            th_inc_params_indices = np.where(
                np.logical_not(params_indices))  # Finding theory inconsistent trial indices
            if len(th_inc_params_indices) <= 0:
                raise ValueError('Did not generate any theory inconsistent indices!')

            # Randomly permuting the th_inc trial indices
            th_inc_params_indices = th_inc_params_indices[np.random.permutation(np.shape(th_inc_params_indices)[0])]
            out = params[th_inc_params_indices[0], :]  # picking one inconsistent particle
        else:
            raise ValueError('Invalid string! valid ones include ''con'' or ''inc'' only')

        if np.any(np.isnan(out)):
            raise ValueError('NaNs in curve parameters!')

    elif get_info == 'weighted_curve':
        raise ValueError('Feature not added yet!')

    else:
        raise ValueError('Invalid operation!')

    return out


# %% markdown
# ## Testing
# - common_to_all_curves('horz_indpnt', 'initial_sampling', 1000, 4)
# - common_to_all_curves('horz_indpnt', 'check_if_exceed_bounds', some_matrix)
# - common_to_all_curves('horz_indpnt', 'curve_volumes', 5)
# - common_to_all_curves('horz_indpnt', 'flip_vertical_params', some_matrix)
# - common_to_all_curves('horz_indpnt', 'sort_horizontal_params', some_matrix)
# - common_to_all_curves('horz_indpnt', 'auto_generate', 'con', 4)
# - common_to_all_curves('horz_indpnt', 'weighted_curve', importance_sampler_mat, 0.9, 4)

# %%
def test_common_to_all_curves():
    ############################################################################
    # setup

    # numpy
    import numpy as np

    # package enabling access/control of matlab from python
    import matlab.engine

    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()

    # paths to matlab helper and model functions
    eng.addpath('../original')

    ############################################################################
    # test initial sampling via matplotlib
    python_output = common_to_all_curves('horz_indpnt', 'initial_sampling', 1000, 4)
    matlab_output = np.asarray(eng.common_to_all_curves('horz_indpnt', 'initial_sampling', 1000.0, 4.0))

    # result is stochastic, so we only test for the same shape
    assert np.shape(python_output) == np.shape(matlab_output)

    # and visualize histograms of output
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    n_bins = 100

    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(python_output)  # , bins=n_bins)
    axs[1].hist(matlab_output)  # , bins=n_bins)

    ############################################################################
    # check_if_exceed_bounds

    # use initial_sampling to generate suitable param file
    param = common_to_all_curves('horz_indpnt', 'initial_sampling', 1000, 4)

    # generate output
    python_output = common_to_all_curves('horz_indpnt', 'check_if_exceed_bounds',
                                         param)  # Check whether curve parameters lie within the upper and lower bounds
    matlab_output = np.asarray(
        eng.common_to_all_curves('horz_indpnt', 'check_if_exceed_bounds', matlab.double(param.tolist())))

    assert np.all(python_output == matlab_output)

    ############################################################################
    # next curve_volumes
    python_output = common_to_all_curves('horz_indpnt', 'curve_volumes', 5)
    matlab_output = eng.common_to_all_curves('horz_indpnt', 'curve_volumes', 5.0)

    #############################################################################
    # auto_generate
    python_output = common_to_all_curves('horz_indpnt', 'auto_generate', 'con', 4.0)
    matlab_output = eng.common_to_all_curves('horz_indpnt', 'auto_generate', 'con', 4.0)
    print(python_output, matlab_output)

    print('All tests passed!')


# %%
# run tests only when this is main file!
if __name__ == '__main__':
    test_common_to_all_curves()

# %%

# %%
