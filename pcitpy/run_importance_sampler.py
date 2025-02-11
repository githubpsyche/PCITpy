# AUTOGENERATED! DO NOT EDIT! File to edit: 00_Parameter_Configuration.ipynb (unless otherwise specified).

__all__ = ['run_importance_sampler']

# Cell
# hide
from .importance_sampler import importance_sampler
from scipy.io import loadmat
import os


def run_importance_sampler(analysis_settings=None, run_sampler=True):
    """
    Sets up the data matrix (number of samples x 6 columns) and the `analysis_settings` dictionary with algorithm
    parameters then (by default) starts the importance sampler routine.

    This is the driver routine that you will use to load your data matrix and also set parameters for the curve-fitting
    procedure. Running the function will initialize the parameters and initiate the core `importance_sampler` function
    of the toolbox.

    **Arguments**:
    - analysis_settings: optional parameter configuration as dictionary. Default parameters will be used otherwise.
    - run_sampler: if True, runs importance_sampler using specified settings. Otherwise returns loaded data matrix and
        analysis_settings dictionary.
    """

    if analysis_settings is None:

        # Populating the analysis_settings struct with algorithm settings
        analysis_settings = {}
        analysis_settings['working_dir'] = '.' # specifies the root subdirectory to find data and store results
        analysis_settings['analysis_id'] = 'my_analysis_id'  # analysis_id: specifies the target directory
        analysis_settings['em_iterations'] = 20  # Number of expectation maximization iterations
        analysis_settings['particles'] = 100000  # Number of particles to be used in the importance sampling algorithm
        analysis_settings['curve_type'] = 'horz_indpnt'  # Name of family of curves to be used. See family_of_curves

        # Name of the distribution (and the default canonical link function which maps the predictor variable to the DV)
        analysis_settings['distribution'] = 'bernoulli'
        analysis_settings['dist_specific_params'] = {}  # For normal distribution the additional parameter is sigma
        analysis_settings['dist_specific_params']['sigma'] = 1

        analysis_settings['beta_0'] = 0  # Initializing beta_0 for linear predictor
        analysis_settings['beta_1'] = 1  # Initializing beta_1 for linear predictor
        analysis_settings['tau'] = 0.05  # Specifies the radius to sample curves in the curve space

        # Specifies if analyses need to run on a specific category. Vector length should be greater than 0. For instance
        # [2] will cause the analyses to be run only on the second category [] will run the analyses on all categories
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
        analysis_settings['bootstrap_run'] = -1  # if non-negative, specify bootstrap sample # unique for each sample

        analysis_settings['scramble'] = False  # indicates whether this run is a scramble run
        analysis_settings['scramble_run'] = -1  # if non-negative, specify bootstrap sample # unique for each sample
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

    results_dir = os.path.join(os.getcwd(), analysis_settings['working_dir'])
    data_path = os.path.join(results_dir, analysis_settings['analysis_id'],
                             analysis_settings['analysis_id'] + '.mat')
    data = loadmat(data_path)['data']
    if run_sampler:
        importance_sampler(data, analysis_settings)
    else:
        return data, analysis_settings