# %% markdown
# # run_importance_sampler
# sets up the data matrix (number of samples x 6 columns) and the 'analysis_settings' struct with algorithm parameters
#
# **USAGE**:
# - Specify parameters within this script and then execute `run_importance_sampler()`
#
# **INPUTS**:
# - None
#
# **OUTPUTS**:
# - None

# %%
from scipy.io import loadmat
import os

from importance_sampler import importance_sampler


def run_importance_sampler():
    # Populating the analysis_settings struct with algorithm settings
    analysis_settings = {}
    analysis_settings['analysis_id'] = 'my_analysis_id'  # analysis_id: specifies the target directory
    analysis_settings['em_iterations'] = 20  # Number of expectation maximization iterations
    analysis_settings['particles'] = 100000  # Number of particles to be used in the importance sampling algorithm
    analysis_settings['curve_type'] = 'horz_indpnt'  # Name of family of curves to be used. Refer to family_of_curves

    # Name of the distribution (and the default canonical link function which maps the predictor variable to the DV)
    analysis_settings['distribution'] = 'bernoulli'
    analysis_settings['dist_specific_params'] = {}  # For normal distribution the additional parameter is sigma
    analysis_settings['dist_specific_params']['sigma'] = 1

    analysis_settings['beta_0'] = 0  # Initializing beta_0 for linear predictor
    analysis_settings['beta_1'] = 1  # Initializing beta_1 for linear predictor
    analysis_settings['tau'] = 0.05  # Specifies the radius to sample curves in the curve space

    # Specifies if the analyses will need to run on a specific category. Vector length should be greater than 0.
    # For instance [2] will cause the analyses to be run only on the second category
    # [] will run the analyses on all categories
    analysis_settings['category'] = []

    # specifies how many std dev away from group mean will the predictor variable outliers need to be dropped
    analysis_settings['drop_outliers'] = 3

    # if TRUE, the independent variables will be z-scored within each subject
    analysis_settings['zscore_within_subjects'] = False

    # Registering which column in the data matrix is carrying which piece of information
    analysis_settings['data_matrix_columns'] = {}
    analysis_settings['data_matrix_columns']['subject_id'] = 0
    analysis_settings['data_matrix_columns']['trials'] = 1
    analysis_settings['data_matrix_columns']['category'] = 2
    analysis_settings['data_matrix_columns']['predictor_var'] = 3
    analysis_settings['data_matrix_columns']['dependent_var'] = 4
    analysis_settings['data_matrix_columns']['net_effect_clusters'] = 5

    analysis_settings['resolution'] = 4  # Denotes the resolution in which the data will be processed

    # Denotes the number of chunks you plan to partition the trials x particles matrix.
    # An example chunk size will be 2 for a 3000 x 50,000 matrix
    analysis_settings['particle_chunks'] = 2
    analysis_settings['bootstrap'] = False  # indicates that this run is a bootstrap run
    analysis_settings['bootstrap_run'] = -1  # if non-negative, specify bootstrap sample number unique for each sample

    analysis_settings['scramble'] = False  # indicates whether this run is a scramble run
    analysis_settings['scramble_run'] = -1  # if non-negative, specify bootstrap sample number unique for each sample
    analysis_settings['scramble_style'] = -1  # choosing the appropriate scramble option from three options below
    if analysis_settings['scramble_style'] > 0:
        if analysis_settings['scramble_style'] == 1:
            analysis_settings['scramble_style'] = 'within_subjects_within_categories'
        elif analysis_settings['scramble_style'] == 2:
            analysis_settings['scramble_style'] = 'within_subjects_across_categories'
        elif analysis_settings['scramble_style'] == 3:
            analysis_settings['scramble_style'] = 'across_subjects_across_categories'
        else:
            raise ValueError('Invalid scramble style given!')

    # %%%%%%%%%%%%%%%%%%%%
    # Reading in the data
    # %%%%%%%%%%%%%%%%%%%%
    # The lines below load the simulated data into the raw_data matrix.
    # Replace these lines of the code with code to load your actual data

    results_dir = os.path.join(os.getcwd(), 'results')
    data_path = os.path.join(results_dir, analysis_settings['analysis_id'],
                             analysis_settings['analysis_id'] + '_simulated_data.mat')
    simulated_data = loadmat(data_path)
    raw_data = simulated_data['raw_data']
    importance_sampler(raw_data, analysis_settings)


# %% markdown
# ## Testing
# `run_importance_sampler` is designed to be modified to specify P-CIT parameters before
# execution. We produce customized version of the function (`eval_run_importance_sampler`) that alters where data is
# sourced from into `data/test/test.m`
#
# Furthermore, to test implementation equivalence, we also prevent the function from calling `importance_sampler` in its
# last line, and instead return its main products `raw_data` and `analysis_settings` for comparison between MATLAB and
# Python implementations.

# %%
def eval_run_importance_sampler():
    # Populating the analysis_settings struct with algorithm settings
    analysis_settings = {}
    analysis_settings['analysis_id'] = 'test'  # analysis_id: specifies the target directory
    analysis_settings['em_iterations'] = 2  # Number of expectation maximization iterations
    analysis_settings['particles'] = 100000  # Number of particles to be used in the importance sampling algorithm
    analysis_settings['curve_type'] = 'horz_indpnt'  # Name of family of curves to be used. Refer to family_of_curves

    # Name of the distribution (and the default canonical link function which maps the predictor variable to the DV)
    analysis_settings['distribution'] = 'bernoulli'
    analysis_settings['dist_specific_params'] = {}  # For normal distribution the additional parameter is sigma
    analysis_settings['dist_specific_params']['sigma'] = 1

    analysis_settings['beta_0'] = 0  # Initializing beta_0 for linear predictor
    analysis_settings['beta_1'] = 1  # Initializing beta_1 for linear predictor
    analysis_settings['tau'] = 0.05  # Specifies the radius to sample curves in the curve space

    # Specifies if the analyses will need to run on a specific category. Vector length should be greater than 0.
    # For instance [2] will cause the analyses to be run only on the second category
    # [] will run the analyses on all categories
    analysis_settings['category'] = []

    # specifies how many std dev away from group mean will the predictor variable outliers need to be dropped
    analysis_settings['drop_outliers'] = 3

    # if TRUE, the independent variables will be z-scored within each subject
    analysis_settings['zscore_within_subjects'] = False

    # Registering which column in the data matrix is carrying which piece of information
    analysis_settings['data_matrix_columns'] = {}
    analysis_settings['data_matrix_columns']['subject_id'] = 0
    analysis_settings['data_matrix_columns']['trials'] = 1
    analysis_settings['data_matrix_columns']['category'] = 2
    analysis_settings['data_matrix_columns']['predictor_var'] = 3
    analysis_settings['data_matrix_columns']['dependent_var'] = 4
    analysis_settings['data_matrix_columns']['net_effect_clusters'] = 5

    analysis_settings['resolution'] = 4  # Denotes the resolution in which the data will be processed

    # Denotes the number of chunks you plan to partition the trials x particles matrix.
    # An example chunk size will be 2 for a 3000 x 50,000 matrix
    analysis_settings['particle_chunks'] = 2
    analysis_settings['bootstrap'] = False  # indicates that this run is a bootstrap run
    analysis_settings['bootstrap_run'] = -1  # if non-negative, specify bootstrap sample number unique for each sample

    analysis_settings['scramble'] = False  # indicates whether this run is a scramble run
    analysis_settings['scramble_run'] = -1  # if non-negative, specify bootstrap sample number unique for each sample
    analysis_settings['scramble_style'] = -1  # choosing the appropriate scramble option from three options below
    if analysis_settings['scramble_style'] > 0:
        if analysis_settings['scramble_style'] == 1:
            analysis_settings['scramble_style'] = 'within_subjects_within_categories'
        elif analysis_settings['scramble_style'] == 2:
            analysis_settings['scramble_style'] = 'within_subjects_across_categories'
        elif analysis_settings['scramble_style'] == 3:
            analysis_settings['scramble_style'] = 'across_subjects_across_categories'
        else:
            raise ValueError('Invalid scramble style given!')

    # %%%%%%%%%%%%%%%%%%%%
    # Reading in the data
    # %%%%%%%%%%%%%%%%%%%%
    # The lines below load the simulated data into the raw_data matrix.
    # Replace these lines of the code with code to load your actual data

    results_dir = os.path.join('data')
    data_path = os.path.join(results_dir, analysis_settings['analysis_id'],
                             analysis_settings['analysis_id'] + '.mat')
    data = loadmat(data_path)['data']
    return data, analysis_settings


# %%
def test_run_importance_sampler():
    # numpy
    import numpy as np

    # package enabling access/control of matlab from python
    import matlab.engine

    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()

    # paths to matlab helper and model functions
    eng.addpath('../original')

    # generate output
    python_data, python_analysis_settings = eval_run_importance_sampler()
    matlab_data, matlab_analysis_settings = eng.eval_run_importance_sampler(nargout=2)

    assert np.all(np.asarray(matlab_data) == np.asarray(python_data))
    print('key python_value matlab_value')
    for key in python_analysis_settings:
        print(key, python_analysis_settings[key], matlab_analysis_settings[key])


# %%
# run tests only when is main file!
if __name__ == '__main__':
    test_run_importance_sampler()
