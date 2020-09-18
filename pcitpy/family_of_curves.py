# %% markdown
# # family_of_curves
# For each of the family of curves this script has curve specific computations like number of curve parameters, boundaries, log likelihood, etc
#
# **USAGE**:
# [OUTPUT] = FAMILY_OF_CURVES(CURVE_TYPE, GET_INFO, VARARGIN)
#
# **INPUTS**:
# - curve_type: Family of curves, string, e.g. 'horz_indpnt'
# - get_info: Cues for specific information / computation, string, e.g. 'get_nParams'
# - varargin: Is either empty or has arguments depending on the computation
#
# **OUTPUTS**:
# - output: Holds the output of all computations

# %%
import numpy as np
from numpy import matlib

from pcitpy.family_of_distributions import family_of_distributions


def family_of_curves(curve_type, get_info, *varargin):
    if curve_type == 'horz_indpnt':
        return horz_indpnt_curve(get_info, varargin)
    else:
        raise ValueError('Invalid curve!')


def horz_indpnt_curve(get_info, input_params):
    """
    Order of curve parameters y1, x1, x2, y2, y3 and y4
    Note always x1 will need to precede x2 (both when passing in curve parameters as well as when plotting)
    """

    nParam = 6
    curve_type = 'horz_indpnt'

    if get_info == 'get_nParams':  # --> (1), number of curve parameters
        return nParam

    elif get_info == 'get_bounds':  # --> (2), Absolute boundaries of the curve parameters
        return np.array([[-1, 1], [0, 1], [0, 1], [-1, 1], [-1, 1], [-1, 1]])

    # --> (3), Indices of vertical parameters i.e. the ones corresponding to v* in the order of curve parameters
    elif get_info == 'get_vertical_params_only':
        return np.array([0, 3, 4, 5])

    # --> (4), Indices of horizontal parameters i.e. the ones corresponding to h* in the order of curve parameters
    elif get_info == 'get_horizontal_params_only':
        return np.array([1, 2])

    # --> (5), Get the curve y-vals (with or without net effects) for each of the P particle curves
    # and then compute the log probability mass function (pmf)
    elif get_info == 'compute_likelihood':
        if len(input_params) <= 5:
            raise ValueError('Missing input parameters!')

        net_effect_clusters = input_params[0]
        particles = int(input_params[1])
        y1 = input_params[2][:, 0]
        x1 = input_params[2][:, 1]
        x2 = input_params[2][:, 2]
        y2 = input_params[2][:, 3]
        y3 = input_params[2][:, 4]
        y4 = input_params[2][:, 5]
        b0 = input_params[3][0]
        b1 = input_params[3][1]
        data = input_params[4]
        distribution = input_params[5]
        dist_specific_params = input_params[6]

        data_matrix_columns = input_params[7]
        predictor_var_column = data_matrix_columns['predictor_var']
        dependent_var_column = data_matrix_columns['dependent_var']
        net_effect_clusters_column = data_matrix_columns['net_effect_clusters']

        del input_params
        if np.logical_not(np.all(np.all(x1 <= x2))):
            raise ValueError(
                'Horizontal parameter 1 is NOT <= Horizontal parameter 2 in {}s family of curves'.format(curve_type))

        x = np.full((len(net_effect_clusters), particles), np.nan)
        y = np.array([])

        # Map the predictor variables to the associated y values for all curves / particles simultaneously
        for i in range(len(net_effect_clusters)):
            cluster_idx = np.nonzero(data[:, net_effect_clusters_column] == net_effect_clusters[i])[0]
            X = np.zeros((len(cluster_idx), particles))
            for j in range(len(cluster_idx)):
                if np.isnan(data[cluster_idx[j], predictor_var_column]):
                    x[i, :] = 0
                else:
                    # If an activation is falling in the third segment of the curve then get the associated y val
                    ix3 = data[cluster_idx[j], predictor_var_column] > x2
                    X[j, ix3] = (np.multiply(np.divide(y4[ix3] - y3[ix3], 1 - x2[ix3]),
                                             data[cluster_idx[j], predictor_var_column] - 1)) + y4[ix3]

                    # If an activation is falling in the first segment of the curve then get the associated y val
                    ix2 = np.logical_and(np.logical_not(ix3), data[cluster_idx[j], predictor_var_column] > x1)  # seg #1
                    X[j, ix2] = (np.multiply(np.divide(y3[ix2] - y2[ix2], x2[ix2] - x1[ix2]),
                                             data[cluster_idx[j], predictor_var_column] - x1[ix2])) + y2[ix2]

                    # If an activation is falling in the first segment of the curve then get the associated y val
                    ix1 = np.logical_not(ix3) & np.logical_not(ix2) & (
                            data[cluster_idx[j], predictor_var_column] > 0)  # segment #1
                    X[j, ix1] = np.multiply(np.divide(y2[ix1] - y1[ix1], x1[ix1]),
                                            data[cluster_idx[j], predictor_var_column]) + y1[ix1]

                    # If an activation is at the intercept of the curve then get the associated y val
                    ix0 = np.logical_not(ix3) & np.logical_not(ix2) & np.logical_not(ix1) & (
                            data[cluster_idx[j], predictor_var_column] == 0)  # Intercept (Boundary condition)
                    X[j, ix0] = y1[ix0]

                    # If an item has net effects then taking the sum below will compute the net effects.
                    # If an item has no net effect then this loop will be executed only once and the sum has no effect
                    x[i, :] = np.sum(X, axis=0)

            # Our model enforces that the DV will need to be unique for items within a net effect cluster
            # i.e. all 1's or all 0's
            if len(np.unique(data[cluster_idx, dependent_var_column])) != 1:
                raise ValueError('Dependent var is NOT unique for net effect cluster {}'.format(i))

            # We accumulate the dependent variables for each net effect cluster
            y = np.concatenate((y, np.unique(data[cluster_idx, dependent_var_column])))

        del X
        del ix0
        del ix1
        del ix2
        del ix3
        del x1
        del x2
        del y1
        del y2
        del y3
        del y4
        del data
        del data_matrix_columns
        if np.any(np.isnan(x)):
            raise ValueError('NaNs in trials x particles matrix!')
        if np.any(np.isinf(x)):
            raise ValueError('Inf in trials x particles matrix!')

        # Compute z = beta_0 + beta_1 x predictor variable
        z = (b1 * x) + b0

        out = {'w': family_of_distributions(distribution, 'compute_densities', z, y, dist_specific_params),
               'net_effects': x,
               'dependent_var': y}

    # --> (6), Use some criterion to carve out the curve space into theory consistent and theory inconsistent
    elif get_info == 'count_particles':
        if len(input_params) <= 0:
            raise ValueError('Missing input parameters!')
        if np.logical_not(np.all(np.all(input_params[0][:, 1] <= input_params[0][:, 2]))):
            raise ValueError(
                'Horizontal parameter 1 is NOT <= Horizontal parameter 2 in {}s family of curves'.format(curve_type))

        # The different branches of theory-consistent curves can be defined in terms of the location of the dip in
        # the curve (the point that anchors the weakening part of the curve) and the rise in the curve (the point
        # that anchors the strengthening part of the curve). More formally, -- The dip in a theory-consistent curve
        # is a point that is located horizontally between the left edge of the curve and the rise. Within this
        # horizontal range, the dip is the lowest point on the curve; it also has to fall below zero on the y-axis.
        # -- The rise in a theory-consistent curve is a point that is located horizontally to the right of the dip.
        # Within this horizontal range, the rise is the highest point on the curve; it also has to fall above zero on
        # the y-axis. Branch I: y2 defines the dip and y3 defines the rise -1 <= y2 < 0, y2 is the dip so it must
        # fall below zero 0 < y3 <= 1, y3 is the rise so it must fall above zero -1 <= y4 <= y3, y4 can hold any
        # value that is below the rise (y3) y2 < y1 <= 1, y1 can hold any value that is above the dip (y2) Branch II:
        # y2 defines the dip and y4 defines the rise -1 <= y2 < 0, y2 is the dip so it must fall below zero 0 < y4 <=
        # 1, y4 is the rise so it must fall above zero y2 <= y3 <= y4, y3 can hold any value between the dip and the
        # rise y2 < y1 <= 1, y1 can hold any value that is above the dip (y2) Branch III: y3 defines the dip and y4
        # defines the rise -1 <= y3 < 0, y3 is the dip so it must fall below zero 0 < y4 <= 1, y4 is the rise so it
        # must fall above zero y3 < y1 <= 1, y1 can hold any value that is above the dip (y3) y3 <= y2 <= 1,
        # y2 can hold any value that is above the dip (y3)

        # Fetch the indices of the theory consistent particles First two lines ensure that the horizontal parameters
        # cannot be 0 OR 1, since that would eliminate a line segment altogether
        return (input_params[0][:, 1] != 0) & (input_params[0][:, 1] != 1) \
               & (input_params[0][:, 2] != 0) & (input_params[0][:, 2] != 1) \
               & (input_params[0][:, 3] >= -1) & (input_params[0][:, 3] < 0) \
               & (input_params[0][:, 4] > 0) & (input_params[0][:, 4] <= 1) \
               & (input_params[0][:, 5] >= -1) & (input_params[0][:, 5] <= input_params[0][:, 4]) \
               & (input_params[0][:, 0] > input_params[0][:, 3]) & (input_params[0][:, 0] <= 1) \
               & (input_params[0][:, 3] >= -1) & (input_params[0][:, 3] < 0) \
               & (input_params[0][:, 5] > 0) & (input_params[0][:, 5] <= 1) \
               & (input_params[0][:, 4] >= input_params[0][:, 3]) & (input_params[0][:, 4] <= input_params[0][:, 5]) \
               & (input_params[0][:, 0] > input_params[0][:, 3]) & (input_params[0][:, 0] <= 1) \
               & (input_params[0][:, 4] >= -1) & (input_params[0][:, 4] < 0) \
               & (input_params[0][:, 5] > 0) & (input_params[0][:, 5] <= 1) \
               & (input_params[0][:, 0] > input_params[0][:, 4]) & (input_params[0][:, 0] <= 1) \
               & (input_params[0][:, 3] >= input_params[0][:, 4]) & (input_params[0][:, 3] <= 1)

    elif get_info == 'get_curve_xy_vals':
        # --> (7), This is the same as compute_likelihood in the sense map the predictor variable to the curve y val
        # but there are couple of differences ...
        # 1. We only know the curve parameters and we have to map all the
        # x_values (0-to-1) to the curve y values where as in compute_likelihood we had specific x values (predictor
        # variables) and curve parameters 2. There is NO net effect cluster concept here 3. We DO NOT compute the pmf
        # as well Hence parts of the code will look similar but we felt these two chunks of code will need to be
        # separate
        if len(input_params) <= 0:
            raise ValueError('Missing input parameters!')

        if len(input_params) > 1:
            resolution = input_params[1]
        else:
            resolution = 4

        particles = np.shape(input_params[0][:, 1])[0]

        if np.any(input_params[0][:, [1, 4, 5, 6]] < -1) or np.any(input_params[0][:, [1, 4, 5, 6]] > 1):
            raise ValueError('Vertical curve parameters exceed bounds [-1, 1]!')
        if np.any(input_params[0][:, [2, 3]] < 0) or np.any(input_params[0][:, [2, 3]] > 1):
            raise ValueError('Horizontal curve parameters exceed bounds [0, 1]!')

        x_value = np.arange(0, 1 + resolution, np.power(.1, resolution))
        x_value = np.matlib.repmat(x_value, particles, 1)
        y_value = np.full(np.shape(x_value), np.nan)
        out = {}

        y1 = np.matlib.repmat(input_params[0][:, 0], 1, np.shape(x_value)[1])
        x1 = np.matlib.repmat(input_params[0][:, 1], 1, np.shape(x_value)[1])
        x2 = np.matlib.repmat(input_params[0][:, 2], 1, np.shape(x_value)[1])
        y2 = np.matlib.repmat(input_params[0][:, 3], 1, np.shape(x_value)[1])
        y3 = np.matlib.repmat(input_params[0][:, 4], 1, np.shape(x_value)[1])
        y4 = np.matlib.repmat(input_params[0][:, 5], 1, np.shape(x_value)[1])
        if not np.all(np.all(x1 <= x2)):
            raise ValueError(
                'Horizontal parameter 1 is NOT <= Horizontal parameter 2 in {} family of curves'.format(curve_type))

        ix3 = x_value > x2  # segment 3
        y_value[ix3] = np.multiply(np.divide(y4[ix3] - y3[ix3], 1 - x2[ix3]), x_value[ix3] - 1) + y4[ix3]

        ix2 = np.logical_not(ix3) & (x_value > x1)  # segment 2
        y_value[ix2] = np.multiply(np.divide(y3[ix2] - y2[ix2], x2[ix2] - x1[ix2]), x_value[ix2] - x1[ix2]) + y2[ix2]

        ix1 = np.logical_not(ix3) & np.logical_not(ix2) & (x_value > 0)  # segment 1
        y_value[ix1] = np.multiply(np.divide(y2[ix1] - y1[ix1], x1[ix1]), x_value[ix1]) + y[ix1]

        ix0 = np.logical_not(ix3) & np.logical_not(ix2) & np.logical_not(ix1) & (x_value == 0)  # intercept
        y_value[ix0] = y1[ix0]

        if np.any(np.isnan(y_value)):
            raise ValueError('NaNs in trials x particles matrix!')
        if np.any(np.isinf(y_value)):
            raise ValueError('Inf in trials x particles matrix!')
        out['xval'] = x_value
        out['yval'] = y_value
        if particles == 1:
            out['curve_params'] = input_params[0]
            out['title_string'] = 'y1={}, x1={}, x2={} y2={}, y3={}, y4={}'.format(
                y1[0], x1[0], x2[0], y2[0], y3[0], y4[0])
    else:
        raise ValueError('Invalid operation!')

    return out


# %% markdown
# ## Testing
# Most function options are simple enough to not require tests.
# `compute_likelihood`, `count_particles`, and `get_curve_xy_vals` are key exceptions.
#
# `compute_likelihood` depends on the most parameters.
# We pregenerate an example set of parameters within `data/test/test_compute_likelihood.mat`.
# We then load the example and use it as parameters for both versions of the `compute_likelihood` option.

# %%

def test_compute_likelihood():
    # numpy
    import numpy as np

    # package enabling access/control of matlab from python
    import matlab.engine

    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()

    # load input (and matlab output)
    data = eng.load('..\\data\\test\\test_compute_likelihoods.mat')
    data['param'] = np.asarray(data['param'])
    ana_opt = data['ana_opt']
    ana_opt['ptl_chunk_idx'] = np.array(ana_opt['ptl_chunk_idx'])
    ana_opt['data_matrix_columns'] = {key: int(value) for value, key in enumerate(ana_opt['data_matrix_columns'])}
    preprocessed_data = np.asarray(data['preprocessed_data'])
    hold_betas = np.array([0, 1])
    ptl_idx = 0
    matlab_output = data['output_struct']

    # generate output
    python_output = family_of_curves(ana_opt['curve_type'], 'compute_likelihood',
                                     np.asarray(ana_opt['net_effect_clusters']),
                                     ana_opt['ptl_chunk_idx'][ptl_idx, 2],
                                     data['param'][int(ana_opt['ptl_chunk_idx'][ptl_idx, 0]):int(
                                         ana_opt['ptl_chunk_idx'][ptl_idx, 1]), :], hold_betas, preprocessed_data,
                                     ana_opt['distribution'], ana_opt['dist_specific_params'],
                                     ana_opt['data_matrix_columns'])

    return python_output


# %%
# run tests only when is main file!
if __name__ == '__main__':
    python_output = test_compute_likelihood()
