# %% markdown
# # importance_sampler
# Recovers a curve that best explains the relationship between the predictor and dependent variables
#
# **USAGE**:
# [] = IMPORTANCE_SAMPLER(RAW_DATA, ANALYSIS_SETTINGS)
#
# **INPUTS**:
# - raw_data: The data matrix (total number of trials x 6 columns). Refer to RUN_IMPORTANCE_SAMPLER()
# - analysis_settings: A struct that holds algorithm relevant settings. Refer to RUN_IMPORTANCE_SAMPLER()
#
# **OUTPUTS**:
# - Saves a .mat file in current_path/analysis_id/analysis_id_importance_sampler.mat

# %%
# helper functions from pcitpy
import math

from preprocessing_setup import preprocessing_setup
from family_of_curves import family_of_curves
from common_to_all_curves import common_to_all_curves

# other dependencies
import numpy as np
import datetime
import random
from scipy import optimize
import scipy.io as sio
from scipy import special

from pcitpy.family_of_distributions import family_of_distributions
from pcitpy.likratiotest import likratiotest
from pcitpy.truncated_normal import truncated_normal


def importance_sampler(raw_data, analysis_settings):
    time = datetime.datetime.now()
    print('Start time {}/{} {}:{}'.format(time.month, time.day, time.hour, time.minute))

    # Resetting the random number seed
    random.seed()

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
    global tau
    global bounds
    global w
    global net_effects
    global dependent_var

    # fetch parameters
    tau = ana_opt['tau']  # Store the tau for convenience
    bounds = family_of_curves(ana_opt['curve_type'], 'get_bounds')  # Get the curve parameter absolute bounds
    nParam = family_of_curves(ana_opt['curve_type'], 'get_nParams')  # Get the number of curve parameters
    hold_betas = [ana_opt['beta_0'], ana_opt['beta_1']]  # Store the betas into a vector

    for em in range(ana_opt['em_iterations']):  # for every em iteration
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
        dependent_var = []  # cannot be initialized in advance since we don't know its length (dropping outliers)

        # Sampling curve parameters
        if em == 0:  # only for the first em iteration
            param = common_to_all_curves(ana_opt['curve_type'], 'initial_sampling',
                                         ana_opt['particles'], ana_opt['resolution'])  # Good old uniform sampling
        else:  # for em iterations 2, 3, etc
            # Sample curve parameters from previous iteration's curve parameters based on normalized weights
            prev_iter_curve_param = param  # we need previous iteration's curve parameters to compute likelihood

            # Here we sample curves (with repetitions) based on the weights
            param = prev_iter_curve_param[random.choices(np.arange(ana_opt['particles']),
                                                         k=ana_opt['particles'], weights=normalized_w[em, :]), :]
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

        result = optimize.minimize(optimizing_function, np.array(hold_betas),
                                   options={'disp': True, 'return_all': True})
        hold_betas = result.x
        f_value = result.fun

        exp_max_f_values[em] = f_value  # gather the f_values over em iterations

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


# %% markdown
# ## `IMPORTANCE_SAMPLER` Testing
# We use `eval_run_importance_sampler` to separately generate the relevant data from our development dataset.

# %%
def test_importance_sampler():
    # numpy
    import numpy as np

    # package enabling access/control of matlab from python
    import matlab.engine

    # matlab instance with relevant paths
    eng = matlab.engine.start_matlab()

    # paths to matlab helper and model functions
    eng.addpath('../original')

    # generate input data and settings
    from run_importance_sampler import eval_run_importance_sampler
    python_data, python_analysis_settings = eval_run_importance_sampler()
    matlab_data, matlab_analysis_settings = eng.eval_run_importance_sampler(nargout=2)

    # generate output
    importance_sampler(python_data, python_analysis_settings)
    eng.importance_sampler(matlab_data, matlab_analysis_settings, nargout=0)


# %% markdown
# # compute_weights
# To compute (P_theta - Q_theta)
#
# **USAGE**:
# [p_theta_minus_q_theta] = compute_weights(
#       curve_name, nParticles, normalized_w, prev_iter_curve_param, param, wgt_chunks, resolution)
#
# **INPUTS**:
# - curve_name: Name of the family of curves (explicitly passed in)
# - nParticles: Number of particles to be used (explicitly passed in)
# - normalized_w: Previous iteration's normalized weights
# - prev_iter_curve_param: Curve parameters held for the previous iteration
# - param: Curve parameters held for the current iteration
# - wgt_chunks: Size of chunk. To deal with limited RAM we break up matrix into smaller matrices
# - resolution: Resolution to which the activations are rounded of
#
# **OUTPUTS**:
# - p_theta_minus_q_theta: Vector of length P (particles)

# %%
def compute_weights(curve_name, nParticles, normalized_w, prev_iter_curve_param, param, wgt_chunks, resolution):
    global which_param
    total_vol = common_to_all_curves(curve_name, 'curve_volumes',
                                     resolution)  # Get the curve volumes (Lesbesgue measure)
    nParam = family_of_curves(curve_name, 'get_nParams')  # Get the number of curve parameters

    # Computing q(theta), i.e. what is the probability of a curve given all curves from the previous iteration
    # P(theta|old_theta)
    q_theta = np.zeros((nParticles, 1))
    reduced_nParticles = nParticles / wgt_chunks
    reduced_nParticles_idx = np.concatenate(np.arange(0, reduced_nParticles, nParticles),
                                            np.arange(0, reduced_nParticles, nParticles) + reduced_nParticles - 1)

    for idx in range(np.shape(reduced_nParticles_idx)[1]):
        prob_grp_lvl_curve = np.zeros((nParticles, reduced_nParticles))
        target_indices = np.arange(reduced_nParticles_idx[1, idx], reduced_nParticles_idx[2, idx])
        for npm in range(nParam):
            which_param = npm
            nth_grp_lvl_param = np.repmat(param[:, npm], 1, reduced_nParticles)
            nth_prev_iter_curve_param = np.transpose(prev_iter_curve_param[target_indices, npm])
            prob_grp_lvl_curve = np.add(prob_grp_lvl_curve,
                                        np.vectorize(compute_trunc_likes)(nth_grp_lvl_param, nth_prev_iter_curve_param))

        if np.any(np.isnan(prob_grp_lvl_curve)):
            raise ValueError('NaNs in probability of group level curves matrix!')

        q_theta = np.add(q_theta, (np.exp(prob_grp_lvl_curve) * np.transpose(normalized_w[target_indices])))

    if np.any(np.isnan(q_theta)):
        raise ValueError('NaNs in q_theta vector!')

    # Computing p(theta) prior i.e. what is the probability of a curve in the curve space
    p_theta = np.ones((nParticles, 1))
    p_theta = np.multiply(p_theta, (1 / total_vol))
    if len(np.unique(p_theta)) != 1:
        raise ValueError('p_theta is NOT unique!')
    if np.any(np.isnan(p_theta)):
        raise ValueError('NaNs in p_theta vector!')

    p_theta_minus_q_theta = np.transpose(np.log(p_theta)) - np.transpose(np.log(q_theta));
    return p_theta_minus_q_theta


# %% markdown
# # compute_trunc_likes
# To compute (P_theta - Q_theta)
#
# **USAGE**:
# [log_likelihood] = compute_trunc_likes(x, mu)
#
# **INPUTS**:
# - x: Kth curve parameter in the current iteration
# - mu: All curve parameters from the previous iteration
# - tau: Similar to sigma in Gaussian distribution (via global)
# - bounds and which_param: Are used to fetch the respective curve parameter's absolute abounds (via global).
#       This is required since we are computing likelihoods for truncated normal
#
# NOTE: tau and bounds are NOT modified throughout this code once initially set
#
# **OUTPUTS**:
# - log_likelihood of that curve given all the P curves from the previous iteration

# %%
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(-(
        np.divide(bounds[which_param, 2] - mu, np.multiply(tau, math.sqrt(2)))))))) + (
                             np.multiply(-0.5, np.log(2) + np.log(np.pi)) - (
                         np.multiply(0.5, np.power(np.divide(x - mu, tau), 2))))
    return log_likelihood


# %%
# run tests only when is main file!
if __name__ == '__main__':
    test_importance_sampler()
