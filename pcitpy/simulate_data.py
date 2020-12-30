import os
import sys
import numpy as np
from family_of_curves import *

sys.path.insert(0, '/Users/Arlene1/Documents')


def simulate_data(analysis_id, noise_knob, curve_type, yval_distribution, net_effects, varargin):

    # Check if the correct number of arguments are passed in (may not need this, python has automatic error)
    if len(varargin) < 6:
        raise Exception('Missing input parameters')

    # Generating a random number (numpy automatically does this based on system clock)
    np.random.seed()

    # Getting current directory
    curr_dir = os.getcwd()

    # Setting the target directory
    results_dir = curr_dir + '/pcit_test/results'
    target_dir = results_dir + analysis_id

    # Create the results, target folder if it doesn't exist
    if os.path.isdir(results_dir) == False:
        os.makedirs(results_dir)
    if os.path.isdir(target_dir) == False:
        os.makedirs(target_dir)

    # Setting the number of subjects
    nSubjects = 35

    # Setting the number of samples per subject
    nItems = 8

    #Setting the item_spread = .15, to ensure each sample repetitions (i.e. items) have at least .15 spread
    item_spread = 0.15

    # Setting the resolution
    resolution = 4

    # Draw a curve and get the x, y values and curve parameters
    if curve_type == 'con' or curve_type == 'inc':
        curve_params = common_to_all_curves(curve_type, 'auto_generate', varargin[0], resolution)
    elif isinstance(varargin, list):
        curve_params = np.array((varargin))
        # out = 100
    else:
        raise ValueError('Invalid varargin! The valid arguments are "con", "inc" or [y1, x1, x2, y2, y3, y4]');

    out = family_of_curves(curve_type, 'get_curve_xy_vals', curve_params)


    # House keeping
    subj_id_list = [];
    item_list = [];
    predictor_var_list = [];
    dependent_var_list = [];
    net_effects_list = [];
    net_eff_counter = 1;

    # For each subject
    for sub in range(nSubjects):
        subj_obs = [];
        yvals = []
        if net_effects > 0:
            tmp_net_effects_list = [];

            # Uniformly sample 'nItems' items per subject between item_spread and (1- item_spread)
            # Now the [item_spread, (1-item_spread)] boundary case comes from the fact you are going to sample
            # net effects for each item as 'each item +/- item_spread' bin boundary.
            subj_act_means = np.random.uniform(item_spread, (1-item_spread), nItems);

            # For each item, sample 'net_effects' amount of net effects
            for m in range(len(subj_act_means)):
                subj_obs = [subj_obs, np.random.uniform(subj_act_means[m]-item_spread, subj_act_means[m]+item_spread, net_effects)]



    return out


# run tests only when is main file!
if __name__ == '__main__':
    import scipy.io

    #output_name = '../data/results/'

    #importance_sampler_mat = output_name + 'analysis-sim-2c_importance_sampler'
    #importance_sampler_mat = scipy.io.loadmat(importance_sampler_mat)

    simulate_data('my_analysis_id', 0.001, 'horz_indpnt', 'bernoulli', 10, [0.6, 0.2, 0.5, -0.4, 0.5, 0.9])





