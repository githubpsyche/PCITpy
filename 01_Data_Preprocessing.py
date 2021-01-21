# hide
# default_exp pcitpy
from nbdev.showdoc import *

# # Data Pre-Processing

# Missing and/or incorrect parameters initialized in `run_importance_sampler` are reset in `preprocessing_setup`. `preprocessing_setup` also preprocesses data based on the specified parameter configuration. From the parameter settings in Table 2 of the P-CIT Toolbox Manual, we see that the predictor variable can be z-scored and outliers can be dropped. We also can generate bootstrap data, scramble the dependent variable, scale the predictor variable between 0 and 1 (this is a mandatory step), and we can perform the analysis on one or more categories while leaving out data from irrelevant categories. Trials where the predictor variable is set to NaN are filtered out (rows removed) for purposes of z-scoring, dropping outliers and scaling. These filtered rows are appended to the data matrix following those pre-processing steps. For, 
#
# **Simple data** analysis (includes both think/no-think and simulated data) the order of pre-processing
# is:
# 1. Filter out irrelevant category data entries (rows) from the data matrix
# 2. Drop outliers in the predictor variable, if drop outliers > 0
# 3. Z-score predictor-variable data within subjects, if zscore within subjects = TRUE
# 4. Scale predictor variable between 0 and 1
#
# **Bootstrap data** analysis the order of pre-processing is:
# 1. Generate bootstrap data from the original data matrix (see the "Nonparametric statistical tests" section of the main paper, and Section 4.8 of the Manual).
# 2. Filter out irrelevant category data entries (rows) from the data matrix
# 3. Drop outliers in the predictor variable, if drop outliers > 0
# 4. Z-score predictor-variable data within subjects, if zscore within subjects = TRUE
# 5. Scale predictor variable between 0 and 1
#
# **Scramble data** analysis the order of pre-processing is:
# 1. Filter out irrelevant category data entries (rows) from the data matrix
# 2. Drop outliers in the predictor variable, if drop outliers > 0
# 3. Z-score predictor-variable data within subjects, if zscore within subjects = TRUE
# 4. Scale predictor variable between 0 and 1
# 5. Scramble the dependent variable depending on the scrambling technique (see the "Nonparametric statistical tests" section of the main paper, and Section 4.8 of the Manual).

# +
# export
# hide

# helper functions from pcitpy
from PCITpy.family_of_curves import family_of_curves
from PCITpy.helpers import scale_data

# other dependencies
import numpy as np
from scipy import stats
import datetime
import random
import os


def preprocessing_setup(data, analysis_settings):
    """Performs sanity checks on the input data and the algorithm parameter 
    struct. Massages the data (i.e. drop outliers, zscore data, etc).

    **Arguments**:  
    - data: Input data matrix (total number of trials x 6 columns)  
    - analysis_settings: Struct with algorithm parameters  

    **Returns**:  
    - data: Input data matrix (if applicable, outlier free, zscored, category specific data only, etc)  
    - analysis_settings: Struct with algorithm parameters; some additional parameters are added to this struct as well  
    
    """
    
    print('********** START OF MESSAGES **********')

    # Checks if the data matrix has 6 columns
    number_of_columns = np.shape(data)[1]
    if number_of_columns != 6:
        raise ValueError('Incorrect number of columns ({}) in the input matrix!'.format(number_of_columns))

    # Registering which column in the data matrix is carrying which piece of information
    if (not ('data_matrix_columns' in analysis_settings)) or (not analysis_settings['data_matrix_columns']):
        # Setting it to the default
        analysis_settings['data_matrix_columns'] = {}
        analysis_settings['data_matrix_columns']['subject_id'] = 0
        analysis_settings['data_matrix_columns']['trials'] = 1
        analysis_settings['data_matrix_columns']['category'] = 2
        analysis_settings['data_matrix_columns']['predictor_var'] = 3
        analysis_settings['data_matrix_columns']['dependent_var'] = 4
        analysis_settings['data_matrix_columns']['net_effect_clusters'] = 5

    subject_id_column = analysis_settings['data_matrix_columns']['subject_id']
    trials_column = analysis_settings['data_matrix_columns']['trials']
    category_column = analysis_settings['data_matrix_columns']['category']
    predictor_var_column = analysis_settings['data_matrix_columns']['predictor_var']
    dependent_var_column = analysis_settings['data_matrix_columns']['dependent_var']
    net_effect_clusters_column = analysis_settings['data_matrix_columns']['net_effect_clusters']

    # Checks if the em iterations is specified; if not specified then it is set to a default of 20
    if (not ('em_iterations' in analysis_settings)) or (analysis_settings['em_iterations'] <= 0):
        analysis_settings['em_iterations'] = 20
        print('Missing number of iterations! It is set to a default of {}'.format(analysis_settings['em_iterations']))

    # Checks if the no. of particles is specified; if not specified then it is set to a default of 1000
    if (not ('particles' in analysis_settings)) or (analysis_settings['particles'] <= 0):
        analysis_settings['particles'] = 100000
        print('Missing number of particles! It is set to a default of {}'.format(analysis_settings['particles']))

    # Checks if the family of curves is specified; if not then set to 'horz_indpnt' (Refer to family of curves)
    if (not ('curve_type' in analysis_settings)) or (not analysis_settings['curve_type']):
        analysis_settings['curve_type'] = 'horz_indpnt';
        print('Missing family of curves! It is set to a default of {}'.format(analysis_settings['curve_type']))

    # Checks if the family of curves exist by fetching the number of curve parameters. This is just a sanity check
    if not isinstance(family_of_curves(analysis_settings['curve_type'], 'get_nParams'), int):
        raise ValueError('{} - Does not exist! Check family_of_curves.m script'.format(analysis_settings['curve_type']))

    # Checks if the distribution is specified;
    # If not specified and if the dependent variable is binary it's set to 'bernoulli'; otherwise set to to 'normal'
    if (not ('distribution' in analysis_settings)) or (not analysis_settings['distribution']):
        if len(np.unique(data[:, dependent_var_column])) == 2:
            analysis_settings['distribution'] = 'bernoulli'
        else:
            analysis_settings['distribution'] = 'normal'
        print('Missing distribution! based on the dependent variable it is set to {}'.format(
            analysis_settings['distribution']))

    # Checks if the distribution specific parameters exist
    if (not ('dist_specific_params' in analysis_settings)) or (not analysis_settings['dist_specific_params']):
        if analysis_settings['distribution'] == 'bernoulli':

            # For a Bernoulli dist there are no parameters so it is empty. We still need the struct to exist
            analysis_settings['dist_specific_params'] = {}

        elif analysis_settings['distribution'] == 'normal':

            # For normal distribution the additional parameter is sigma. We pass in sigma here.
            analysis_settings['dist_specific_params'] = {}
            analysis_settings['dist_specific_params']['sigma'] = 1  # Default is 1
            print('Missing sigma for normal distribution! It is set to {}'.format(
                analysis_settings['dist_specific_params']['sigma']))

    # Checks if normal distribution specific parameter is valid i.e. sigma > 0
    if (analysis_settings['distribution'] == 'normal') and (analysis_settings['dist_specific_params']['sigma'] <= 0):
        raise ValueError('Normal distribution sigma will need to > 0! sigma = {}'.format(
            analysis_settings['dist_specific_params']['sigma']))

    # Checks if beta_0 is specified; if not specified then it is set to a default of 0
    if not ('beta_0' in analysis_settings):
        analysis_settings['beta_0'] = 0
        print('Missing initial setting for beta_0! It is set to a default of {}'.format(analysis_settings['beta_0']))

    # Checks if beta_1 is specified; if not specified then it is set to a default of 1
    if not ('beta_1' in analysis_settings):
        analysis_settings['beta_1'] = 1
        print('Missing initial setting for beta_1! It is set to a default of {}'.format(analysis_settings['beta_1']))

    # Checks if tau is specified; if not specified then it is set to a default of 0.05
    if not ('tau' in analysis_settings):
        analysis_settings['tau'] = 0.05
        print('Missing initial setting for tau! It is set to a default of {}'.format(analysis_settings['tau']))

    # Checks if this is a bootstrap run; if not specified then it is set to a default of false
    if not ('bootstrap' in analysis_settings):
        analysis_settings['bootstrap'] = False
        print('Missing initial setting for beta_1! It is set to a default of {}'.format(analysis_settings['bootstrap']))

    # Checks if bootstrap flag is boolean
    if not (type(analysis_settings['bootstrap']) == bool):
        raise ValueError('analysis_settings.bootstrap field will need to be boolean!')

    # Checks if this is a scramble run; if not specified then it is set to a default of false
    if not ('scramble' in analysis_settings):
        analysis_settings['scramble'] = False

    # Checks if scramble flag is boolean
    if not (type(analysis_settings['scramble']) == bool):
        raise ValueError('analysis_settings.scramble field will need to be boolean!')

    # Errors if both bootstrap and scramble flags exist
    if analysis_settings['scramble'] and analysis_settings['bootstrap']:
        raise ValueError(
            'Cannot run both scramble AND bootstrap analyses at the same time! Set any one flag to be false')

    # Builds a bootstrap data matrix from the original data matrix
    if analysis_settings['bootstrap'] and not (analysis_settings['scramble']):

        # We need a bootstrap sample number
        if (not ('bootstrap_run' in analysis_settings)) or (not analysis_settings['bootstrap_run']):
            raise ValueError(
                'Missing bootstrap sample number! set analysis_settings.bootstrap_run to a valid sample number')

        bootstrap_data = []
        new_cluster_count = 1
        new_subject_count = 1

        # Get the number of subjects from the data matrix
        number_of_subjects = len(np.unique(data[:, subject_id_column]))

        # Randomly sample with replacement the number of subjects thus generating our bootstrap sample
        subj_num_with_replacement = random.choices(np.arange(number_of_subjects), k=number_of_subjects)

        # For each subject in our bootstrap sample gather all relevant information
        for i in range(len(subj_num_with_replacement)):
            subj_idx = np.where(data[:, subject_id_column] == subj_num_with_replacement[i])

            # Recreate a new net effect cluster since this will need to be unique in the data matrix
            # (by repeatedly sampling subjects we could be repeating the net effect clusters)
            cluster_vector = data[subj_idx, net_effect_clusters_column]
            cluster_numbers = np.unique[cluster_vector]
            for j in range(len(cluster_numbers)):
                target_idx = np.where(data[subj_idx, net_effect_clusters_column] == cluster_numbers[j])
                cluster_vector[target_idx] = new_cluster_count
                new_cluster_count += 1

            # Recreate a new subject id
            # (by repeatedly sampling subjects we could be repeating the subject id's)
            # Gather all information into a bootstrap_data matrix
            bootstrap_data.append(np.concatenate(np.repmat(new_subject_count, len(subj_idx), 1),
                                                 data[subj_idx, trials_column:dependent_var_column], cluster_vector))
            new_subject_count += 1

        # Perform some sanity checks to ensure that the bootstrap_data matrix is similar to the actual data matrix
        if not np.all(np.shape(bootstrap_data) == np.shape(data)):
            raise ValueError('Size of bootstrap dataset NOT the same as original data!')
        if not (len(np.unique(data[:, net_effect_clusters_column])) == len(
                np.unique(bootstrap_data[:, net_effect_clusters_column]))):
            raise ValueError('The number of clusters are not the same in the original and bootstrap sample!')
        if not np.array_equal(data[:, subject_id_column], bootstrap_data[:, subject_id_column]):
            raise ValueError('The ordering of subjects are not the same in the original and bootstrap sample!')

        # Store away the bootstrap sample subject information for future reference
        analysis_settings['bootstrap_run_subj_id'] = subj_num_with_replacement
        data = bootstrap_data

    # Checks if analysis will be performed for a specific category; if not then set to [] i.e. NOT category specific
    if not ('category' in analysis_settings):
        analysis_settings.category = []
        print(
            'Missing category specific analyses information! We are going to ignore the category dimension i.e. all '
            'trials from all categories will be analysed')

        # If this analysis is to be performed for a specific category then filters out data from other irrelevant categories
    if len(analysis_settings['category']) > 0:
        target_cat_idx = []
        data_cat = np.unique(data[:, category_column])
        for c in range(len(analysis_settings['category'])):
            cat_exist = np.where(data_cat == analysis_settings['category'][c])[0]
            if cat_exist.size == 0:
                raise ValueError('Category does not exist! You have set analysis_settings.category[{}]={}'.format(
                    c, analysis_settings['category'][c]))
            target_cat_idx = np.concatenate(target_cat_idx,
                                            np.where(data[:, category_column] == analysis_settings['category'][c])[0])
        data = data[target_cat_idx, :]

    # Checks if outliers (i.e. data trials) will need to dropped; if not specified then set to 'DO NOT DROP OUTLIERS'
    if not ('drop_outliers' in analysis_settings):
        analysis_settings['drop_outliers'] = 3
        print(
            'Missing drop_outliers specific information! We are dropping outliers that are {} standard deviations away from the group mean'.format(
                analysis_settings['drop_outliers']))

    # If this analysis requires the outliers dropped, then drops the data trials within std devs from the GROUP MEAN
    if analysis_settings['drop_outliers'] > 0:
        # NaN's do not qualify as outliers so we filter them out and add them at the end of this step
        nan_free_idx = np.logical_not(np.isnan(data[:, predictor_var_column]))

        # NaN free data
        nan_free_data = data[nan_free_idx, :]
        std_dev_predictor_var = np.std(nan_free_data[:, predictor_var_column], ddof=1) * analysis_settings[
            'drop_outliers']
        mean_predictor_var = np.mean(nan_free_data[:, predictor_var_column])
        predictor_var_idx = (nan_free_data[:, predictor_var_column] > (mean_predictor_var - std_dev_predictor_var)) & (
                nan_free_data[:, predictor_var_column] < (mean_predictor_var + std_dev_predictor_var))
        print('{} trials are dropped since they are regarded as outliers'.format(
            np.shape(nan_free_data)[subject_id_column] - np.sum(predictor_var_idx)))
        nan_free_data_outlier_dropped = nan_free_data[predictor_var_idx, :]

        # NaN's trials
        nan_data = data[np.logical_not(nan_free_idx), :]

        # Combine the NaN data with the outlier free data
        data = np.concatenate(nan_free_data_outlier_dropped, nan_data) if np.shape(nan_data)[
                                                                              0] > 0 else nan_free_data_outlier_dropped

    # Following the 'filter by category' and 'drop outliers', if applicable, we check if the data matrix is empty
    number_of_trials = np.shape(data)[subject_id_column]
    if number_of_trials <= 0:
        raise ValueError('No input data!')

    # Checks if we need to zscore predictor var within subjects, if not specified then it is set to default of FALSE
    if not ('zscore_within_subjects' in analysis_settings):
        analysis_settings['zscore_within_subjects'] = 0
        print('Missing zscore_within_subjects information! We are NOT zscoring within subjects')

    # Verifies if zscore within subjects is boolean
    if not (type(analysis_settings['zscore_within_subjects']) == bool):
        raise ValueError('zscore_within_subjects field will need to be boolean!')

    # Zscore the predictor variable within each subject
    if analysis_settings['zscore_within_subjects']:
        # NaN's do not qualify to be zscored
        nan_free_idx = np.logical_not(np.isnan(data[:, predictor_var_column]))
        # NaN free data
        nan_free_data = data[nan_free_idx, :]
        # Get the list of subject id's (we use this cell array in zscoring the data within each subject, if applicable)
        subject_id_list = np.unique(nan_free_data[:, subject_id_column])
        # We get the number of subjects
        number_of_subjects = len(subject_id_list)
        if number_of_subjects <= 0:
            raise ValueError('Not valid number of subjects!')
        for s in range(number_of_subjects):
            subject_idx = np.where(nan_free_data[:, subject_id_column] == subject_id_list[s])[0]
            nan_free_data[subject_idx, predictor_var_column] = stats.zscore(
                nan_free_data[subject_idx, predictor_var_column], ddof=1)
        print('Predictor variables within each subject are zscored!')
        # NaN's trials
        nan_data = data[np.logical_not(nan_free_idx), :]
        # Combine the NaN data with the outlier free data
        data = np.concatenate(nan_free_data, nan_data) if np.shape(nan_data)[0] > 0 else nan_free_data

    # Checks if resolution is specified, if not specified then set to default of 4. This translates to 1e-4 = 0.0001
    if (not ('resolution' in analysis_settings)) or (analysis_settings['resolution'] <= 0):
        analysis_settings['resolution'] = 4
        print('Missing resolution! It is set to a default of %d'.format(analysis_settings['resolution']))

    # if we have normally distributed data, we want to z-score the dependent variable
    if analysis_settings['distribution'] == 'normal':
        data[:, dependent_var_column] = stats.zscore(data[:, dependent_var_column], ddof=1)

    # We scale the predictor var to be between 0 and 1 and round it to 4 digits
    nan_free_idx = np.logical_not(np.isnan(data[:, predictor_var_column]))
    nan_free_data = data[nan_free_idx, :]
    nan_free_data[:, predictor_var_column] = np.round(scale_data(nan_free_data[:, predictor_var_column], 0, 1),
                                                      analysis_settings['resolution'])
    nan_data = data[np.logical_not(nan_free_idx), :]
    data = np.concatenate(nan_free_data, nan_data) if np.shape(nan_data)[0] > 0 else nan_free_data

    # Scrambling the data matrix
    if analysis_settings['scramble']:
        if (not ('scramble_run' in analysis_settings)) or (not analysis_settings['scramble_run']):
            raise ValueError(
                'Missing scramble sample number! set analysis_settings.scramble_run to a valid sample number')
        if (not ('scramble_style' in analysis_settings)) or (not analysis_settings['scramble_style']):
            analysis_settings[
                'scramble_style'] = 'within_subjects_within_categories'  # most conservative of all scramble techniques
            print('Missing scramble style! It is set a default of {}'.format(analysis_settings['scramble_style']))

        # We get the list of subject id's
        subject_id_list = np.unique(data[:, subject_id_column])
        # We get the number of subjects in this analysis
        number_of_subjects = len(subject_id_list)
        if number_of_subjects <= 0:
            raise ValueError('Not valid number of subjects!')

        if analysis_settings['scramble_style'] == 'within_subjects_within_categories':
            # Here scramble all DVs WHILE respecting the net effect boundaries, subject groupings and category groupings
            categories = np.unique(data[:, category_column])
            for s in range(number_of_subjects):
                for c in range(len(categories)):
                    subject_category_idx = np.where((data[:, subject_id_column] == subject_id_list[s]) & (
                            data[:, category_column] == categories[c]))[0]
                    if len(subject_category_idx) > 1:
                        data[subject_category_idx, dependent_var_column] = scramble_dependent_variable(
                            data[subject_category_idx, dependent_var_column],
                            data[subject_category_idx, net_effect_clusters_column])

        elif analysis_settings['scramble_style'] == 'within_subjects_across_categories':
            # Here we scramble all dependent variables WHILE respecting the net effect boundaries and subject groupings
            for s in range(number_of_subjects):
                subject_idx = np.where(data[:, subject_id_column] == subject_id_list[s])[0]
                if len(subject_idx) > 1:
                    data[subject_idx, dependent_var_column] = scramble_dependent_variable(
                        data[subject_idx, dependent_var_column], data[subject_idx, net_effect_clusters_column])

        elif analysis_settings['scramble_style'] == 'across_subjects_across_categories':
            # Here we scramble all dependent variables WHILE respecting the net effect boundaries
            all_idx = np.arange(np.shape(data)[0])
            if len(all_idx) > 1:
                data[all_idx, dependent_var_column] = scramble_dependent_variable(
                    data[all_idx, dependent_var_column], data[all_idx, net_effect_clusters_column])

        else:
            raise ValueError('Invalid analysis_settings.scramble_style={}'.format(analysis_settings['scramble_style']))

    # Our data matrix looks like data = [subject id, item, category, predictor var, dependent var, net effect cluster]
    # We verify if the subject id and dependent var columns are unique for the net effect clusters
    # Below is a example of a valid data matrix (note dependent variable is unique within net effect cluster 111)
    # data(1, :) = [24, 1, 1, 0.3333, 0, 111]
    # data(2, :) = [24, 2, 2, 0.2222, 0, 111]
    # data(3, :) = [24, 3, 1, 0.4444, 0, 111]
    # Below is a example of an invalid data matrix (note dependent variable is not unique within net effect cluster 111)
    # data(1, :) = [24, 1, 1, 0.3333, 0, 111]
    # data(2, :) = [24, 2, 2, 0.2222, 1, 111]
    # data(3, :) = [24, 3, 1, 0.4444, 0, 111]

    # Fetching the net effect clusters
    net_effect_clusters = np.unique(data[:, net_effect_clusters_column])
    analysis_settings['net_effect_clusters'] = net_effect_clusters

    # If net effect clusters exist verify if the Subject Id and dependent variable are unique for those clusters
    if len(net_effect_clusters) != np.shape(data)[0]:
        for i in range(len(net_effect_clusters)):
            cluster_idx = np.where(data[:, net_effect_clusters_column] == net_effect_clusters[i])[0]
            if len(np.shape(np.unique(data[cluster_idx, [subject_id_column, dependent_var_column]], axis=0))) != 1:
                raise ValueError('Subject Id and/or dependent variable not unique for net effect cluster {}! Check '
                                 'the data matrix'.format(net_effect_clusters[i]))
    else:
        # If net effect clusters DO NOT exist then we treat each row as a net effect cluster by itself
        print('Each row will be treated separately. We will NOT be computing the net effect of any rows')

    # We create an analysis id unique to this analysis
    if (not ('analysis_id' in analysis_settings)) or (not analysis_settings['analysis_id']):
        time = datetime.datetime.now()
        analysis_settings['analysis_id'] = '{}-{}-{}-{}-{}'.format(time.month, time.day, time.hour, time.minute,
                                                                   time.second)

    # We create a results directory if no specific target directory is mentioned
    if (not ('target_dir' in analysis_settings)) or (not analysis_settings['target_dir']):
        results_dir = os.path.join(os.getcwd(), 'results')
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        analysis_settings['target_dir'] = results_dir

    # target_directory = 'results/analysis_id'
    analysis_settings['target_dir'] = os.path.join(analysis_settings['target_dir'], analysis_settings['analysis_id'])
    if not os.path.isdir(analysis_settings['target_dir']):
        os.mkdir(analysis_settings['target_dir'])

    # Due to memory constraints we perform two chunking tricks

    # Chunking trick I
    # In the curve fitting algorithm we need to compute the p(current iteration curves | previous
    # iteration curves). This matrix is huge when the number of particles (curves) is large, say 100,000. Even with a
    # 8 Gb RAM, dedicated to Matlab, we still get a out of memory errors. To avoid this problem we chunk the matrix
    # into smaller, more manageable matrices. Setting the chunk size to be particles x 0.05 -> 100,000 x 0.05 = 5000,
    # translates to p(current iteration curves(5000 curves at a time) | previous iteration curves).
    analysis_settings['wgt_chunks'] = analysis_settings['particles'] * 0.05
    # If the chunk size is less then 5000 we set it be the number of particles itself
    if analysis_settings['wgt_chunks'] < 5000:
        analysis_settings['wgt_chunks'] = analysis_settings['particles']

    # Chunking trick II
    if not ('particle_chunks' in analysis_settings):
        analysis_settings['particle_chunks'] = 2
        print('Missing particle chunks! It is set to a default of {}'.format(analysis_settings['particle_chunks']))

    # Depending on the number of particle chunks we get start, end points and the number of particles within each chunk.
    # For instance 1000 particles divided into 4 chunks will look like,
    # 0	250	250
    # 250	500	250
    # 500	750	250
    # 750	1000	250

    dummy = np.arange(0, analysis_settings['particles'],
                      analysis_settings['particles'] / analysis_settings['particle_chunks'])
    analysis_settings['ptl_chunk_idx'] = np.stack(
        (dummy, dummy + analysis_settings['particles'] / analysis_settings['particle_chunks'],
         np.full(np.shape(dummy), analysis_settings['particles'] / analysis_settings['particle_chunks'])),
        axis=1)

    # Storing analysis relevant information into the analysis_settings struct
    # We get the list of subject id's
    subject_id_list = np.unique(data[:, subject_id_column])

    # We get the number of subjects in this analysis
    analysis_settings['nSubjs'] = len(subject_id_list)
    if analysis_settings['nSubjs'] <= 0:
        raise ValueError('Not valid number of subjects!')

    print('********** END OF MESSAGES **********')
    return data, analysis_settings


# -

show_doc(preprocessing_setup, title_level=2)


# The function can be run and tested in isolation from the toolbox pipeline with a line like `preprocessing_setup(run_importance_sampler(run_sampler=False))`.

# +
# export
# hide

def scramble_dependent_variable(target_dependent_variables, 
                                target_net_effect_clusters, testing=False):
    """Takes dependent variable vector and scramble it such that the net 
    effect cluster groupings are NOT violated.

    **Arguments**:  
    - target_dependent_variables: The vector you would like scrambled  
    - target_net_effect_clusters: The groupings that you would like to NOT violate. Follow the example below  

    **Returns** a scrambled vector
    """
    
    if np.logical_not(np.shape(target_dependent_variables) == np.shape(target_net_effect_clusters)):
        raise ValueError('Size of input vectors must be the same!')

    # Detailed example
    # example data matrix: target_dependent_variables = [1, 0, 1, 0, 0, 0, 1]
    # and target_net_effect_clusters = [3, 5, 3, 7, 7, 5, 8]

    # Fetch the sorted list of net effect clusters and their respective locations
    # e.g. for [3, 5, 3, 7, 7, 5, 8] will return [3, 3, 5, 5, 7, 7, 8] and [1, 3, 2, 6, 4, 5, 7]
    sorted_net_effect_clusters = np.sort(target_net_effect_clusters)
    sorted_net_effect_clusters_indices = np.argsort(target_net_effect_clusters)
    just_ones = np.ones(np.shape(sorted_net_effect_clusters))  # Populate a vector full of ones

    # compute the length of each net effect cluster
    # e.g. for [3, 5, 3, 7, 7, 5, 8] will return [2, 2, 2, 1] i.e. 3 is repeated twice and so on
    length_of_each_net_effect_cluster = np.transpose(np.bincount(sorted_net_effect_clusters))
    length_of_each_net_effect_cluster = length_of_each_net_effect_cluster[
        length_of_each_net_effect_cluster.astype(bool)]

    # Get the unique list of clusters (i.e. excluding repetitions if any) e.g. [3, 5, 7, 8]
    unique_net_effect_clusters, unique_indices = np.unique(target_net_effect_clusters, return_index=True)
    # Get the associated dependent variables (one per cluster; recall it is unique within a cluster) e.g. [1, 0, 0, 1]
    associated_dependent_variables = np.array(target_dependent_variables)[unique_indices.astype(int)]
    # scramble the dependent variables e.g. [0, 0, 1, 1]
    scrambled_indices = np.random.permutation(len(associated_dependent_variables))
    scrambled_dependent_variables = associated_dependent_variables[scrambled_indices]
    if testing:
        scrambled_dependent_variables = np.array([0, 0, 1, 1])

    # Now we will need to repeat each scrambled dependent variable for the length of that net effect cluster. The
    # next three lines will result in [0, 0, 0, 0, 1, 1, 1] corresponding to [3, 3, 5, 5, 7, 7, 8] since the
    # scrambled dependent variable looks like [0, 0, 1, 1] for [3, 5, 7, 8]
    cumsum_clusters = np.cumsum(length_of_each_net_effect_cluster)
    indicator_vector = np.zeros((cumsum_clusters[-1]))
    indicator_vector[np.append(np.array([0]), [cumsum_clusters[:-1]])] = 1

    # Store the scrambled dependent variable in the respective cluster locations
    # The original vector looked like [3, 5, 3, 7, 7, 5, 8] so the scrambled vector will look like [0, 0, 0, 1, 1, 0, 1]
    scrambled_vector = np.full(np.shape(sorted_net_effect_clusters_indices), np.nan)
    scrambled_vector[np.array(sorted_net_effect_clusters_indices)] = scrambled_dependent_variables[
        np.cumsum(indicator_vector.astype(int)) - 1]

    if np.any(np.isnan(scrambled_vector)):
        raise ValueError('Nan''s in scrambled dependent variable vector!')

    return scrambled_vector


# -

show_doc(scramble_dependent_variable, title_level=2)

# While `preprocessing_setup` is hard to demonstrate in isolation, its helper function `scramble_dependent_variable` is straightforward to illustrate:

scramble_dependent_variable([1, 0, 1, 0, 0, 0, 1], [3, 5, 3, 7, 7, 5, 8])

# `array([1., 1., 1., 0., 0., 1., 0.])`
