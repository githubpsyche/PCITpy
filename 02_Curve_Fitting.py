# %%
# hide
# default_exp importance_sampler
from nbdev.showdoc import *

# %% [markdown]
# # Curve Fitting

# %% [markdown]
# For description of the curve-fitting procedure, see the "Estimating the plasticity curve" section of
# main paper and the "curve-fitting algorithm details" section of the supplementary materials.

# %% [markdown]
# ## Ongoing Development Workflow
# This core function of PCITpy is still a work in progress! 
# 
# During development, `importance_sampler` is refactored into a sequence of notebook cells, with a unique cell for each
# `em_iteration`. The IPython cell magic `%store` is used to cache results and outputs of each unit of the subdivided
# function, easing the task of translating and testing a function that takes a long time to run. Rather than having to
# repeatedly run these computations, I can just cache and execute them only once. 
# 
# Once the final cell returns output that matches what MATLAB gives, I'll worry about optimization, and eventually
# re-refactor the finished code into a packageable function. With the docstring:
 
# %%
# export 

def importance_sampler(raw_data, analysis_settings):
   """
   Recovers a curve that best explains the relationship between the predictor and dependent variables
   
   **Arguments**:
   - raw_data: The data matrix (total number of trials x 6 columns). Refer to RUN_IMPORTANCE_SAMPLER()
   - analysis_settings: A struct that holds algorithm relevant settings. Refer to RUN_IMPORTANCE_SAMPLER()

    Saves a .mat file in `current_path/analysis_id/analysis_id_importance_sampler.mat`
   """
   pass

# %% [markdown]
# ## Cache Reinitialization
# Here we re-load all the variables relevant to ongoing development so far.

# %% 
var_list = ["ana_opt", "bounds", "exp_max_f_values", "hold_betas", "hold_betas_per_iter", "importance_sampler",
            "nParam", "normalized_w", "preprocessed_data", "tau", "seed", "dependent_var", "em", "exp_max_f_values",
            "f_value", "net_effects", "param", "prev_iter_curve_param", "ptl_idx", "result", "w"]
for var in var_list:
    %store -r $var
    pass
    
# %% [markdown]
# ## Function Inputs
# We'll grab and define the variables that usually work as the inputs into `importance_sampler`

# %%
from pcitpy.run_importance_sampler import run_importance_sampler

if not all([var in globals() for var in var_list]):
    analysis_settings = {'working_dir': 'data', 
        'analysis_id': 'test',
        'em_iterations': 2,
        'particles': 100000,
        'curve_type': 'horz_indpnt',
        'distribution': 'bernoulli',
        'dist_specific_params': {'sigma': 1},
        'beta_0': 0,
        'beta_1': 1,
        'tau': 0.05,
        'category': [],
        'drop_outliers': 3,
        'zscore_within_subjects': False,
        'data_matrix_columns': {'subject_id': 0,
        'trials': 1,
        'category': 2,
        'predictor_var': 3,
        'dependent_var': 4,
        'net_effect_clusters': 5},
        'resolution': 4,
        'particle_chunks': 2,
        'bootstrap': False,
        'bootstrap_run': -1,
        'scramble': False,
        'scramble_run': -1,
        'scramble_style': -1}

    raw_data, analysis_settings = run_importance_sampler(analysis_settings, run_sampler=False)

# %% [markdown]
# ## Initialization
# All the code up to the first `em_iteration`. 
#
# Some potential problem areas with my caching approach are evident. First, I don't use a consistent random seed ( or
# date) between cached and 'live' variables. I can choose a static random seed to cache and restore at each test, but
# will have to remember to reconfigure the code later. It's okay to reinitialize `time` at each execution.

# %%
# helper functions from pcitpy
import math
from numba import vectorize, float64, int32, int64, njit
from pcitpy.preprocessing_setup import preprocessing_setup
from pcitpy.family_of_curves import family_of_curves
from pcitpy.common_to_all_curves import common_to_all_curves
from pcitpy.family_of_distributions import family_of_distributions
from pcitpy.helpers import likratiotest, truncated_normal

# other dependencies
import numpy as np
import datetime
import random
from scipy import optimize
import scipy.io as sio
from scipy import special


time = datetime.datetime.now()
print('Start time {}/{} {}:{}'.format(time.month, time.day, time.hour, time.minute))

if "seed" in globals():
    random.seed(seed)
if not all([var in globals() for var in var_list]):

    # Resetting the random number seed
    random.seed()
    seed = random.getstate()

    # Preprocessing the data matrix and updating the analysis_settings struct with additional/missing information
    preprocessed_data, ana_opt = preprocessing_setup(raw_data, analysis_settings)
    del raw_data
    del analysis_settings

    # Housekeeping
    importance_sampler = {}  # Creating the output struct
    hold_betas_per_iter = np.full((ana_opt['em_iterations'] + 1, 2), np.nan)  # Matrix to hold betas over em iterations
    exp_max_f_values = np.full((ana_opt['em_iterations'], 1), np.nan)  # Matrix to hold the f_values over em iterations
    normalized_w = np.full((ana_opt['em_iterations'] + 1, ana_opt['particles']),
                            np.nan)  # to hold the normalized weights

    # fetch parameters
    tau = ana_opt['tau']  # Store the tau for convenience
    bounds = family_of_curves(ana_opt['curve_type'], 'get_bounds')  # Get the curve parameter absolute bounds
    nParam = family_of_curves(ana_opt['curve_type'], 'get_nParams')  # Get the number of curve parameters
    hold_betas = [ana_opt['beta_0'], ana_opt['beta_1']]  # Store the betas into a vector

# %% [markdown]
# ## First `em_iteration`
# Two of these happen in our tests since beyond the second iteration function behavior is the same (as far as we know). 

# %%

if not all([var in globals() for var in var_list]):

    em = 0 # for em in range(ana_opt['em_iterations']):  # for every em iteration
    hold_betas_per_iter[em, :] = hold_betas  # Store the logreg betas over em iterations
    print('Betas: {}, {}'.format(hold_betas[0], hold_betas[1]))
    print('EM Iteration: {}'.format(em))

    # Initialize the previous iteration curve parameters, weight vector, net_effects and dependent_var matrices
    # Matrix to hold the previous iteration curve parameters
    prev_iter_curve_param = np.full((ana_opt['particles'], family_of_curves(ana_opt['curve_type'], 'get_nParams')),
                                    np.nan)
    w = np.full((ana_opt['particles']), np.nan)  # Vector to hold normalized weights

    # Matrix to hold the predictor variables (taking net effects if relevant) over all particles
    net_effects = np.full((len(ana_opt['net_effect_clusters']), ana_opt['particles']), np.nan)
    dependent_var = np.array([])  # can't be initialized in advance as we don't know its length (dropping outliers)

    # Sampling curve parameters
    if em == 0:  # only for the first em iteration
        param = common_to_all_curves(ana_opt['curve_type'], 'initial_sampling',
                                        ana_opt['particles'], ana_opt['resolution'])  # Good old uniform sampling
    else:  # for em iterations 2, 3, etc
        # Sample curve parameters from previous iteration's curve parameters based on normalized weights
        prev_iter_curve_param = param  # we need previous iteration's curve parameters to compute likelihood

        # Here we sample curves (with repetitions) based on the weights
        param = prev_iter_curve_param[random.choices(np.arange(ana_opt['particles']),
                                                        k=ana_opt['particles'], weights=normalized_w[em - 1, :]), :]
        # Add Gaussian noise since some curves are going to be identical due to the repetitions
        # NOISE: Sample from truncated normal distribution using individual curve parameter bounds,
        # mean = sampled curve parameters and sigma = tau
        for npm in range(nParam):
            param[:, npm] = truncated_normal(bounds[npm, 0], bounds[npm, 1],
                                                param[:, npm], tau, ana_opt['particles'])

    # Check whether curve parameters lie within the upper and lower bounds
    param = common_to_all_curves(ana_opt['curve_type'], 'check_if_exceed_bounds', param)
    if ana_opt['curve_type'] == 'horz_indpnt':
        # Check if the horizontal curve parameters are following the right trend i.e. x1 < x2
        param = common_to_all_curves(ana_opt['curve_type'], 'sort_horizontal_params', param)

    # Compute the likelihood over all subjects (i.e. log probability mass function if logistic regression)
    #  This is where we use the chunking trick II
    for ptl_idx in range(np.shape(ana_opt['ptl_chunk_idx'])[0]):
        output_struct = family_of_curves(
            ana_opt['curve_type'], 'compute_likelihood', ana_opt['net_effect_clusters'],
            ana_opt['ptl_chunk_idx'][ptl_idx, 2],
            param[int(ana_opt['ptl_chunk_idx'][ptl_idx, 0]):int(ana_opt['ptl_chunk_idx'][ptl_idx, 1]), :],
            hold_betas, preprocessed_data,
            ana_opt['distribution'], ana_opt['dist_specific_params'], ana_opt['data_matrix_columns'])

        # Gather weights
        w[int(ana_opt['ptl_chunk_idx'][ptl_idx, 0]):int(ana_opt['ptl_chunk_idx'][ptl_idx, 1])] = output_struct['w']

        # Gather predictor variable
        net_effects[:, int(ana_opt['ptl_chunk_idx'][ptl_idx, 0]):int(ana_opt['ptl_chunk_idx'][ptl_idx, 1])] = \
            output_struct['net_effects']
        if ptl_idx == 0:
            # Gather dependent variable only once, since it is the same across all ptl_idx
            dependent_var = output_struct['dependent_var']

    del output_struct
    if np.any(np.isnan(w)):
        raise ValueError('NaNs in normalized weight vector w!')

    # Compute the p(theta) and q(theta) weights
    if em > 0:
        p_theta_minus_q_theta = compute_weights(
            ana_opt['curve_type'], ana_opt['particles'], normalized_w[em - 1, :],
            prev_iter_curve_param, param, ana_opt['wgt_chunks'], ana_opt['resolution'])
        w += p_theta_minus_q_theta

    w = np.exp(w - special.logsumexp(w))  # Normalize the weights using logsumexp to avoid numerical underflow
    normalized_w[em, :] = w  # Store the normalized weights

    # Optimize betas using fminunc
    optimizing_function = family_of_distributions(ana_opt['distribution'], 'fminunc_both_betas', w, net_effects,
                                                    dependent_var, ana_opt['dist_specific_params'])

    result = optimize.minimize(optimizing_function, np.array(hold_betas), jac=True,
                                options={'disp': True, 'return_all': True})
    hold_betas = result.x
    f_value = result.fun

    exp_max_f_values[em] = f_value  # gather the f_values over em iterations

    # # Successive Iterations

    # # Variable Caching
    # Here we build a list of and cache all global variables that were active after an uncached run of the above cells.
    # By setting `var_list` to this list near the start of the notebook, we should reload the relevant variables instead
    # of running all this code again. 
    for var in var_list:
        %store $var
        pass

# %% [markdown]
# ### Further Caching
# Testing is currently focused on this second iteration. We can define an additional buffer cell that manages caching
# for the parts of this iteration we're satisfied with. To do that in an orderly way, we can update our var list, add
# the new code to the conditional-wrapped cell above, move the %store snippet originally located at the end of our first
# iteration cell to our buffer cell, and temporarily prevent retrieval from storage in our initial caching code cell.
# Then when we run the notebook up to the code cell under development, the cache will be updated to include the result
# of the newly validated operations. Re-enabling cache retrieval will finalize this process.

# %% [markdown]
# ## Second `em_iteration`
# Two of these happen in our tests since beyond the second iteration, function behavior is the same (as far as we know).

# %%

em = 1 # for em in range(ana_opt['em_iterations']):  # for every em iteration
hold_betas_per_iter[em, :] = hold_betas  # Store the logreg betas over em iterations
print('Betas: {}, {}'.format(hold_betas[0], hold_betas[1]))
print('EM Iteration: {}'.format(em))

# Initialize the previous iteration curve parameters, weight vector, net_effects and dependent_var matrices
# Matrix to hold the previous iteration curve parameters
prev_iter_curve_param = np.full((ana_opt['particles'], family_of_curves(ana_opt['curve_type'], 'get_nParams')),
                                np.nan)
w = np.full((ana_opt['particles']), np.nan)  # Vector to hold normalized weights

# Matrix to hold the predictor variables (taking net effects if relevant) over all particles
net_effects = np.full((len(ana_opt['net_effect_clusters']), ana_opt['particles']), np.nan)
dependent_var = np.array([])  # can't be initialized in advance as we don't know its length (dropping outliers)

# Sampling curve parameters
if em == 0:  # only for the first em iteration
    param = common_to_all_curves(ana_opt['curve_type'], 'initial_sampling',
                                    ana_opt['particles'], ana_opt['resolution'])  # Good old uniform sampling
else:  # for em iterations 2, 3, etc
    # Sample curve parameters from previous iteration's curve parameters based on normalized weights
    prev_iter_curve_param = param  # we need previous iteration's curve parameters to compute likelihood

    # Here we sample curves (with repetitions) based on the weights
    param = prev_iter_curve_param[random.choices(np.arange(ana_opt['particles']),
                                                    k=ana_opt['particles'], weights=normalized_w[em - 1, :]), :]
    # Add Gaussian noise since some curves are going to be identical due to the repetitions
    # NOISE: Sample from truncated normal distribution using individual curve parameter bounds,
    # mean = sampled curve parameters and sigma = tau
    for npm in range(nParam):
        param[:, npm] = truncated_normal(bounds[npm, 0], bounds[npm, 1],
                                            param[:, npm], tau, ana_opt['particles'])

# Check whether curve parameters lie within the upper and lower bounds
param = common_to_all_curves(ana_opt['curve_type'], 'check_if_exceed_bounds', param)
if ana_opt['curve_type'] == 'horz_indpnt':
    # Check if the horizontal curve parameters are following the right trend i.e. x1 < x2
    param = common_to_all_curves(ana_opt['curve_type'], 'sort_horizontal_params', param)

    # Compute the likelihood over all subjects (i.e. log probability mass function if logistic regression)
#  This is where we use the chunking trick II
for ptl_idx in range(np.shape(ana_opt['ptl_chunk_idx'])[0]):
    output_struct = family_of_curves(
        ana_opt['curve_type'], 'compute_likelihood', ana_opt['net_effect_clusters'],
        ana_opt['ptl_chunk_idx'][ptl_idx, 2],
        param[int(ana_opt['ptl_chunk_idx'][ptl_idx, 0]):int(ana_opt['ptl_chunk_idx'][ptl_idx, 1]), :],
        hold_betas, preprocessed_data,
        ana_opt['distribution'], ana_opt['dist_specific_params'], ana_opt['data_matrix_columns'])

    # Gather weights
    w[int(ana_opt['ptl_chunk_idx'][ptl_idx, 0]):int(ana_opt['ptl_chunk_idx'][ptl_idx, 1])] = output_struct['w']

    # Gather predictor variable
    net_effects[:, int(ana_opt['ptl_chunk_idx'][ptl_idx, 0]):int(ana_opt['ptl_chunk_idx'][ptl_idx, 1])] = \
        output_struct['net_effects']
    if ptl_idx == 0:
        # Gather dependent variable only once, since it is the same across all ptl_idx
        dependent_var = output_struct['dependent_var']

del output_struct
if np.any(np.isnan(w)):
    raise ValueError('NaNs in normalized weight vector w!')

# Compute the p(theta) and q(theta) weights
if em > 0:
    p_theta_minus_q_theta = compute_weights(
        ana_opt['curve_type'], ana_opt['particles'], normalized_w[em - 1, :],
        prev_iter_curve_param, param, ana_opt['wgt_chunks'], ana_opt['resolution'])
    w += p_theta_minus_q_theta

w = np.exp(w - special.logsumexp(w))  # Normalize the weights using logsumexp to avoid numerical underflow
normalized_w[em, :] = w  # Store the normalized weights

# Optimize betas using fminunc
optimizing_function = family_of_distributions(ana_opt['distribution'], 'fminunc_both_betas', w, net_effects,
                                                dependent_var, ana_opt['dist_specific_params'])

result = optimize.minimize(optimizing_function, np.array(hold_betas), jac=True,
                            options={'disp': True, 'return_all': True})
hold_betas = result.x
f_value = result.fun

exp_max_f_values[em] = f_value  # gather the f_values over em iterations

# %% [markdown]
# ## After Em Iterations
# Sure would be a dream come true if I got to this point...

# %%
hold_betas_per_iter[em + 1, :] = hold_betas  # Store away the last em iteration betas
print('>>>>>>>>> Final Betas: {}, {} <<<<<<<<<'.format(hold_betas[0], hold_betas[1]))

# Flipping the vertical curve parameters if beta_1 is negative
importance_sampler['flip'] = False
neg_beta_idx = hold_betas[1] < 0
if neg_beta_idx:
    print('!!!!!!!!!!!!!!!!!!!! Beta 1 is flipped !!!!!!!!!!!!!!!!!!!!')
    hold_betas[1] = hold_betas[1] * -1
    param = common_to_all_curves(ana_opt['curve_type'], 'flip_vertical_params', param)
    importance_sampler['flip'] = True

w = np.full((ana_opt['particles']), np.nan)  # Clearing the weight vector

# Used for a likelihoods ratio test to see if our beta1 value is degenerate
w_null_hypothesis = np.full((ana_opt['particles']), np.nan)

# The null hypothesis for the likelihoods ratio test states that our model y_hat = beta_0 + beta_1 * predictor
# variable is no different than the simpler model y_hat = beta_0 + beta_1 * predictor variable WHERE BETA_1 =
# ZERO i.e. our model is y_hat = beta_0
null_hypothesis_beta = [hold_betas[0], 0]

for ptl_idx in range(np.shape(ana_opt.ptl_chunk_idx)[0]):
    output_struct = family_of_curves(
        ana_opt['curve_type'], 'compute_likelihood', ana_opt['net_effect_clusters'],
        ana_opt['ptl_chunk_idx'][ptl_idx, 3],
        param[ana_opt['ptl_chunk_idx'][ptl_idx, 1]:ana_opt['ptl_chunk_idx'][ptl_idx, 2], :], hold_betas,
        preprocessed_data,
        ana_opt['distribution'], ana_opt['dist_specific_params'], ana_opt['data_matrix_columns'])
    w[ana_opt['ptl_chunk_idx'][ptl_idx, 1]:ana_opt['ptl_chunk_idx'][ptl_idx, 2]] = output_struct['w']

# this code computes the log likelihood of the data under the null hypothesis i.e. using null_hypothesis_beta
# instead of hold_betas -- it's "lazy" because, unlike the alternative hypothesis, we don't have to compute the
# data likelihood for each particle because it's exactly the same for each particle (b/c compute_likelihood uses
# z = beta_1 * x + beta_0, but (recall that our particles control the value of x in this equation) beta_1 is zero
# for the null hypothesis) that's why we pass in the zero vector representing a single particle with irrelevant
# weights so we don't have to do it for each particle unnecessarily
output_struct_null_hypothesis_lazy = family_of_curves(
    ana_opt['curve_type'], 'compute_likelihood', ana_opt['net_effect_clusters'], 1, [0, 0, 0, 0, 0, 0],
    null_hypothesis_beta, preprocessed_data, ana_opt['distribution'], ana_opt['dist_specific_params'],
    ana_opt['data_matrix_columns'])
data_likelihood_null_hypothesis = output_struct_null_hypothesis_lazy['w']
data_likelihood_alternative_hypothesis = w

w = w + p_theta_minus_q_theta
if np.any(np.isnan(w)):
    raise ValueError('NaNs in normalized weight vector w!')

w = np.exp(w - special.logsumexp(w))  # Normalize the weights using logsumexp to avoid numerical underflow
normalized_w[em + 1, :] = w  # Store the normalized weights

# Added for debugging chi-sq, might remove eventually
importance_sampler['data_likelihood_alternative_hypothesis'] = data_likelihood_alternative_hypothesis
importance_sampler['data_likelihood_null_hypothesis'] = data_likelihood_null_hypothesis

# we calculate the data_likelihood over ALL particles by multiplying the data_likelihood for each particle by
# that particle's importance weight
dummy_var, importance_sampler['likratiotest'] = likratiotest(
    w * np.transpose(data_likelihood_alternative_hypothesis), data_likelihood_null_hypothesis, 2, 1)

if np.any(np.isnan(normalized_w)):
    raise ValueError('NaNs in normalized weights vector!')
if np.any(np.isnan(exp_max_f_values)):
    raise ValueError('NaNs in Expectation maximilzation fval matrix!')
if np.any(np.isnan(hold_betas_per_iter)):
    raise ValueError('NaNs in hold betas matrix!')

importance_sampler['normalized_weights'] = normalized_w
importance_sampler['exp_max_fval'] = exp_max_f_values
importance_sampler['hold_betas_per_iter'] = hold_betas_per_iter
importance_sampler['curve_params'] = param
importance_sampler['analysis_settings'] = ana_opt

if ana_opt['bootstrap']:
    sio.savemat('{}/{}_b{}_importance_sampler.mat'.format(ana_opt['target_dir'], ana_opt['analysis_id'],
                                                            ana_opt['bootstrap_run']),
                {'importance_sampler': importance_sampler})
elif ana_opt['scramble']:
    sio.savemat('{}/{}_s{}_importance_sampler.mat'.format(ana_opt['target_dir'], ana_opt['analysis_id'],
                                                            ana_opt['scramble_run']),
                {'importance_sampler': importance_sampler})
else:
    sio.savemat('{}/{}_importance_sampler.mat'.format(ana_opt['target_dir'], ana_opt['analysis_id']),
                {'importance_sampler': importance_sampler})
print('Results are stored in be stored in {}'.format(ana_opt['target_dir']))

time = datetime.datetime.now()
print('Finish time {}/{} {}:{}'.format(time.month, time.day, time.hour, time.minute))


# %%
show_doc(importance_sampler, title_level=2)
