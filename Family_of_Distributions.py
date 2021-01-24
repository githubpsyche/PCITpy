# %%
# hide
# default_exp family_of_distributions
from nbdev.showdoc import *

# %%
# export
# hide
import numpy as np


def family_of_distributions(distribution_name, get_info, *varargin):
    """For each of the family of distributions this script performs specific 
    computations like number of pdf/pmf, etc.
    
    This function contains the probability density computation, the 
    distribution-specific fminunc to optimize the objective function, and the 
    associated partial derivatives for each of the dependent-variable 
    distributions that are currently covered by the toolbox (i.e., Bernoulli, 
    normal)
    
    **Arguments**:  
    - distribution_name: distribution name, string, e.g. 'bernoulli', 'normal'
    - get_info: Cues for specific information / computation, string, e.g. 
        'get_nParams'
    - varargin: Is either empty or has arguments depending on the computation
    
    **Returns** output of all computations.
    """
    if distribution_name == 'bernoulli':
        return bernoulli_distribution(get_info, varargin)
    elif distribution_name == 'normal':
        return normal_distribution(get_info, varargin)
    else:
        raise ValueError('Invalid distribution!')


# %%
show_doc(family_of_distributions, title_level=1)


# %%
# export
# hide
def bernoulli_distribution(get_info, input_params):
    """If get_info is `compute_densities`, computes the log probability 
    densities of the curves specified by input_params using the bernoulli 
    distribution. Otherwise passes parameters to `fminunc_both_betas`. 
    """
    
    # --> Compute the log densities. We compute the log(probability function)
    if get_info == 'compute_densities':  
        if len(input_params) <= 1:
            raise ValueError('Missing input parameters!')
        z = input_params[0]
        y = input_params[1]
        del input_params

        # Compute fz = 1 / (1 + exp(-z) - Logistic function
        fz = 1 / (1 + np.exp(-z))
        del z
        fz = np.fmax(fz, np.finfo(float).eps)
        fz = np.fmin(fz, 1 - np.finfo(float).eps)

        # Compute bern_log_pmf = p ^ k + (1 - p) ^ (1 - k). 
        # http://en.wikipedia.org/wiki/Bernoulli_distribution
        # Here p = fz and k = y. 
        # Taking the log results in y x log(fz) + (1 - y) x log(1 - fz).
        return np.sum((np.log(fz).T * y).T + (
            np.log(1 - fz).T * np.subtract(1, y)).T, axis=0)

    elif get_info == 'fminunc_both_betas':
        if len(input_params) <= 1:
            raise ValueError('Missing input parameters!')
        return lambda betas: fminunc_bernoulli_both(
            betas, input_params[0], input_params[1], input_params[2])
    else:
        raise ValueError('Invalid operation!')


# %%
show_doc(bernoulli_distribution, title_level=2)


# %%
# export
# hide
def fminunc_bernoulli_both(betas, w, net_effects, dependent_var):
    """Optimizes logistic regression betas using bernoulli cost function F
    
    **Arguments**:  
    - betas: The current betas that were used to compute likelihoods
    - w: Weight vector that holds the normalized weights for P particles
    - net_effects: Predictor variable Matrix (number of trials x particles)
    - dependent_var: Dependent variable Matrix (number of trials x 1)
    
    **Returns**:
    - f: Scalar, Objective function
    - g: Vector of length 2 i.e. gradients with respect to beta_0 and beta_1
    """

    beta_0 = betas[0]
    beta_1 = betas[1]

    z = (beta_1 * net_effects) + beta_0
    fz = 1 / (1 + np.exp(-z))
    if np.any(np.isinf(fz)):
        raise ValueError('Inf in fz matrix!')
    fz = np.fmax(fz, np.finfo(float).eps)
    fz = np.fmin(fz, 1 - np.finfo(float).eps)

    # Cost function
    # We will need to maximize the betas but fminunc minimizes hence a -ve.
    # Here we compute the log pmf over all trials and then component multiply 
    # by the weights and then sum them up over all particles
    f = -np.sum(w * np.sum((np.log(fz).T * dependent_var).T + (
        np.log(1 - fz).T * np.subtract(1, dependent_var)).T, axis=0), axis=0)

    # Here we take the partial derivative of log pmf over beta_0 and beta_1 
    # respectively, component multiply by the weights and sum them up over all 
    # particles
    g = np.zeros(2)
    g[0] = -np.sum(w * np.sum(
        (dependent_var - (np.exp(z) / (1 + np.exp(z))).T).T, axis=0), axis=0)
    g[1] = -np.sum(w * np.sum((net_effects.T * dependent_var).T - (
        (net_effects * np.exp(z)) / (1 + np.exp(z))), axis=0))
    if np.any(np.isinf(g)):
        raise ValueError('Inf in partial derivative!')
    if np.any(np.isnan(g)):
        raise ValueError('NaN in partial derivative!')

    return f, g


# %%
show_doc(fminunc_bernoulli_both, title_level=3)


# %%
# export
# hide 

def normal_distribution(get_info, input_params):
    """If get_info is `compute_densities`, computes the log probability 
    densities of the curves specified by input_params using the normal 
    distribution. Otherwise passes parameters to `fminunc_normal_both`. 
    """
    if get_info == 'compute_densities':
        if len(input_params) <= 2:
            raise ValueError('Missing input parameters!')

        mu = input_params[0]
        y = input_params[1]
        dist_specific_params = input_params[2]
        sigma = dist_specific_params['sigma']

        # Compute log_pdf http://en.wikipedia.org/wiki/Normal_distribution
        return np.sum(np.subtract((1 / np.power(sigma, 2)) * np.subtract(
            np.multiply(y, mu), np.add(np.multiply(.5, np.power(mu, 2)), 
                                       np.multiply(.5, np.power(y, 2))))), 
                      (.5 * np.log(2 * np.pi * np.power(sigma, 2))))

    # --> (2), This fetches the right function handle for the fminunc
    elif get_info == 'fminunc_both_betas':  
        if len(input_params) <= 3:
            raise ValueError('Missing input parameters!')

        return lambda betas: fminunc_normal_both(
            betas, input_params[0], input_params[1], 
            input_params[2], input_params[3])

    else:
        raise ValueError('Invalid operation!')


# %%
show_doc(normal_distribution, title_level=2)


# %%
# export
# hide
def fminunc_normal_both(betas, w, net_effects, dependent_var, dist_specific_params):
    """Optimizes logistic regression betas using normal cost function F
     
    **Arguments**:
    - betas: The current betas that were used to compute likelihoods
    - w: Weight vector that holds the normalized weights for P particles
    - net_effects: Predictor variable Matrix (number of trials x particles)
    - dependent_var: Dependent variable Matrix (number of trials x 1)
    - sigma: Used to specify variance in the Normal distribution
    
    **Returns**:
    - f: Scalar, Objective function
    - g: Vector of length 2 i.e. gradients with respect to beta_0 and beta_1
    """

    beta_0 = betas[0]
    beta_1 = betas[1]
    sigma = dist_specific_params['sigma']

    mu = (beta_1 * net_effects) + beta_0

    # Cost function
    # We will need to maximize the betas but fminunc minimizes hence a -ve.
    # Here we compute the log pdf over all trials and then component multiply by the weights
    # and then sum them up over all particles
    f = -np.sum(
        w * np.sum(np.subtract(np.multiply((1 / np.power(sigma, 2)), np.subtract(np.multiply(dependent_var, mu), np.add(
            np.multiply(.5, np.power(mu, 2)), np.multiply(.5, np.power(dependent_var, 2))))),
                               (.5 * np.log(2 * np.pi * np.power(sigma, 2))))))

    # Here we take the partial derivative of log pdf over beta_0 and beta_1 respectively,
    # component multiply by the weights and sum them up over all particles
    g = [-np.sum(w * np.sum(
        (1 / np.power(sigma, 2)) * np.subtract(dependent_var, np.add(beta_0, np.multiply(beta_1, net_effects))))),
         -np.sum(w * np.sum(np.multiply(np.divide(net_effects, np.power(sigma, 2)),
                                        np.subtract(dependent_var, np.add(beta_0, np.multiply(beta_1, net_effects))))))]

    if np.any(np.isinf(g)):
        raise ValueError('Inf in partial derivative!')
    if np.any(np.isnan(g)):
        raise ValueError('NaN in partial derivative!')

    return f, g


# %%
show_doc(fminunc_normal_both, title_level=3)
