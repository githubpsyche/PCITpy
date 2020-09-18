import os
import sys
import numpy as np

sys.path.insert(0, '/Users/Arlene1/Documents')


def simulate_data(analysis_id, noise_knob, curve_type, yval_distribution, net_effects, varargin):

    # # Check if the correct number of arguments are passed in (may not need this, python has automatic error)
    # if len(varargin) < 6:
    #     raise Exception('Missing input parameters')

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
    if curve_type = 'con' or curve_type = 'inc':
        print('hi')



