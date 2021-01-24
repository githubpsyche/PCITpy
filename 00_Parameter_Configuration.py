# -*- coding: utf-8 -*-
# hide
# default_exp run_importance_sampler
from nbdev.showdoc import *

# # Parameter Configuration

# Here we review the requirements for and toolbox functions supporting the configuration of parameters for curve-fitting with P-CIT before execution of the core `importance_sampler` procedure. Most language-agnostic guidance is recreated directly from the [P-CIT Toolbox Manual](https://github.com/PrincetonUniversity/p-cit-toolbox) provided by Annamalai Natarajan, Samuel Gershman, Luis Piloto, Greg Detre, and Kenneth Norman with Princeton University. 

# ## Data Matrix

# The dimensionality of the data matrix that holds the information required for the analysis is $T \times 6$.
# $T$ here corresponds to the total number of entries (over all subjects, all items, all repetitions) that
# you would like analyzed. All entries in the data matrix will need to be numeric (exception-the
# predictor variable values can be set to NaN; see discussion of baseline items below). The 6 columns
# correspond to the _subject id_, _sample number_, _category_, _predictor variable_, _dependent variable_ and _net effect cluster_ respectively. Before we explain each of these columns in detail, let us look at an example data matrix. Table 1 in the P-CIT Toolbox Manual shows data entries from a variant of the think/no-think dataset. The entries in the table correspond to no-think and baseline items from two categories, face and scene. The predictor variable here is the difference between relevant-category and irrelevant-category classifier readouts. The voxels that were fed into the classifier were selected using a bilateral two-region (fusiform gyrus and parahippocampal gyrus) mask. In this variant of the dataset the
# classifier, was trained on four categories (face, scene, car, shoe) and tested on two categories (face,
# scene) only.

# - **Column 1** -_subject id_: Within a data matrix, each subject should be assigned a unique subject ID. From the example data matrix you can see that subject IDs are the same for all entries within a subject.
#
# - **Column 2** -_trial number_: The trial number must be unique within each subject but can be repeated across subjects. In the example data matrix, the trial numbers go from 1 through 106 for subject 1 before being repeated for subject 2.
#
# - **Column 3** -_category_: Category numbers can be used to represent different conditions within the experiment. The curve-fitting procedure can fit a curve to just one category if need be. If this is irrelevant to your dataset, just set it to -1. From the example data matrix the entries are divided up into to either category 1 (face trials) or category 2 (scene trials).
#
# - **Column 4** -_predictor variable_: This column should be populated with predictor variable data from your experiments (e.g., classifier outputs, reaction times - whatever information it is that you are using to predict the dependent variable). In the example data matrix, we are using the difference between relevant-category and irrelevant-category classifier readouts as our predictor variable. Also, note some entries (trial numbers 105, 106) in the example data matrix are NaNs. These entries correspond to the baseline items, where no predictor variable information is available (see the "Computing importance weights for individual curves" section of the main paper, and the "Anchoring the vertical position of the curve using baseline items" section of the supplementary materials). When computing the net effects these baseline items, by default, have their net effects set to zero.
#
# - **Column 5** -_dependent variable_: You can populate this column with dependent variable data from your experiments. In the example data matrix, the dependent variable is Bernoulli distributed (i.e., 0 or 1). The toolbox is equipped to handle normally distributed continuous dependent variables as well; refer to Section 6.3 of the Manual. 
#
# - **Column 6** -_net effect cluster_ : This tells the toolbox which trials to group together when computing net effects. In the think/no-think experiment, all 12 repetitions of a given no-think item are grouped within the same net effect cluster. Note that, for all trials within a given net effect cluster, the predictor variable values can be different across these trials, but they all must share the same dependent variable value (e.g., in the think/no-think experiment, each repetition of a given no-think item was associated with a different level of classifier evidence, but all of these repetitions were associated with same dependent variable value - the item was either remembered correctly or incorrectly on the final test). Each net effect cluster should have its own unique identifier value (e.g., in the think/no-think dataset, all of the repetitions of the first no-think item from the first participant were assigned a net effect cluster value of 1; all of the repetitions of the second no-think item from the first participant were assigned a net effect cluster value of 2; and so on). Note also that the trials belonging to a given net effect cluster do not need to be contiguous to one another in the matrix (see Table 1 in the P-CIT Toolbox Manual).
#
# Table 1 in the P-CIT Toolbox Manual depicts a variant of the think/no-think analysis in which 8 no-think items were repeated 12 times and 10 baseline items were assigned one row each; this brings the total number of rows per subject to 106 (8 x 12 + 10 x 1). In the think/no-think paper, authors analyzed data from a total of 26 subjects, hence the total number of rows in the data matrix for this variant of the think/no-think analysis is 2756. They discuss other variants of the think/no-think analysis in Section 4.12 of the Manual.

# +
# export
# hide
from pcitpy.pcitpy import importance_sampler
from scipy.io import loadmat
import os


def run_importance_sampler(analysis_settings=None, run_sampler=True):
    """Sets up the data matrix (number of samples x 6 columns) and the 
    `analysis_settings` dictionary with algorithm parameters before starting 
    the importance sampler.
    
    This is the driver routine that you will use to load your data matrix and 
    also set parameters for the curve-fitting procedure. Running the function 
    will initialize the parameters and initiate the core `importance_sampler` 
    function of the toolbox.

    **Arguments**:  
    - analysis_settings: optional parameter configuration as dictionary. 
        Default parameters will be used otherwise.
    - run_sampler: if True, runs importance_sampler using specified settings. 
        Otherwise returns loaded data matrix and analysis_settings dictionary.
    """
    
    if analysis_settings is None:
        
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
                             analysis_settings['analysis_id'] + '.mat')
    data = loadmat(data_path)['data']
    raw_data = simulated_data['raw_data']
    if run_sampler:
        importance_sampler(raw_data, analysis_settings)
    else:
        return data, analysis_settings


# -

show_doc(run_importance_sampler, title_level=2)

# In its original implementation in MATLAB, as opposed to accepting any 
# input parameter configuration, parameters were specified within 
# `run_importance_sampler` by altering variable assignments in its code. For 
# a similar workflow, you could configure your own copy of the function 
# (click "source" above to locate the current implementation) to specify 
# parameters and run the function. Alternatively, you can provide an 
# `analysis_settings` dictionary object as an argument to 
# `run_importance_sampler`, specifying parameters as a set of key-value 
# pairs.
#
# Refer to Table 2 in the P-CIT Toolbox Manual for parameters, description, 
# example usage and default settings. All these parameters are initialized 
# in `run_importance_sampler`. The default settings correspond to the 
# settings that were used to do curve-fitting on scene, no-think trials in 
# the Detre et al. paper. Settings related to bootstrap and scramble 
# analyses are explained in more detail in Sections 4.8 and 4.9 
# respectively of the P-CIT Toolbox Manual.
