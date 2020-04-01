# # family_of_distributions
# For each of the family of distributions this script performs specific computations like number of pdf/pmf, etc
#
# **USAGE**:
# [OUTPUT] = FAMILY_OF_DISTRIBUTIONS(DISTRIBUTION_NAME, GET_INFO, VARARGIN)
#
# **INPUTS**:
# - distribution_name: distribution name, string, e.g. 'bernoulli', 'normal'
# - get_info: Cues for specific information / computation, string, e.g. 'get_nParams'
# - varargin: Is either empty or has arguments depending on the computation
#
# **OUTPUTS**:
# - output: Holds the output of all computations

# +
import numpy as np

def family_of_distributions(distribution_name, get_info, *varargin):
    if distribution_name == 'bernoulli':
        return bernoulli_distribution(get_info, *varargin)
    elif distribution_name == 'normal':
        return normal_distribution(get_info, *varargin)
    else:
        raise ValueError('Invalid distribution!')


# +
def bernoulli_distribution(get_info, input_params):
    if get_info == 'compute_densities': # --> (1), Compute the log densities. NOTE: We compute the log(probability function)
        if len(input_params) <= 1:
            raise ValueError('Missing input parameters!')
        z = input_params[0]
        y = input_params[1]
        clear input_params
        
        # Compute fz = 1 / (1 + exp(-z) - Logistic function
        fz = 1 / (1 + np.exp(-z))
        clear z
        fz = max(fz, np.finfo(float).eps)
        fz = min(fz, 1-np.finfo(float).eps)
        
        # Compute bern_log_pmf = p ^ k + (1 - p) ^ (1 - k). http://en.wikipedia.org/wiki/Bernoulli_distribution
        # Here p = fz and k = y. Taking the log results in y x log(fz) + (1 - y) x log(1 - fz). This is written below in bsxfun syntax
        return np.sum(np.multiply(np.log(fz), y) + np.multiply(np.log(1-fz), np.subtract(1, y)))

    elif get_info == 'fminunc_both_betas':
        if len(input_params) <= 1:
            raise ValueError('Missing input parameters!')
        return lambda betas: fminunc_bernoulli_both(betas, input_params[0], input_params[1], input_params[2])
    else:
        raise ValueError('Invalid operation!')

def fminunc_bernoulli_both(betas, w, net_effects, dependent_var):

    # [F, G] = FMINUNC_BERNOULLI_BOTH(BETAS)
    # 
    # Purpose
    # 
    # To optimize logistic regression betas using cost function F
    #  
    # Input
    #
    # --betas: The current betas that were used to compute likelihoods
    # --w: Weight vector that holds the normalized weights for P particles
    # --net_effects: Predictor variable Matrix (number of trials x particles)
    # --dependent_var: Dependent variable Matrix (number of trials x 1)
    # 
    # Output
    #
    # --f: Scalar, Objective function
    # --g: Vector of length 2 i.e. gradients with respect to beta_0 and beta_1
    #

    beta_0 = betas[0]
    beta_1 = betas[1]

    z = (beta_1 * net_effects) + beta_0
    fz = 1 / 1+np.exp(-z)
    if np.any(np.isinf(fz)):
        raise ValueError('Inf in fz matrix!')
    fz = max(fz, np.finfo(float).eps)
    fz = min(fz, 1-np.finfo(float).eps)

    # Cost function
    # We will need to maximize the betas but fminunc minimizes hence a -ve.
    # Here we compute the log pmf over all trials and then component multiply by the weights and then sum them up over all particles
    f = -np.sum(w * np.sum(np.multiply(np.log(fz),dependent_var) + np.multiply(
        np.log(1-fz), np.subtract(1, dependent_var))))

    # Here we take the partial derivative of log pmf over beta_0 and beta_1 respectively, component multiply by the weights and sum them up over all paricles
    g = []
    g.append(-np.sum(w * np.sum(np.subtract(dependent_var, np.exp(z) / 1-np.exp(z)))))
    g.append(-np.sum(w * np.sum(np.multiply(net_effects, dependent_var) - (
        (net_effects * np.exp(z)) / (1+np.exp(z)))))
    if np.any(np.isinf(g)):
        raise ValueError('Inf in partial derivative!')
    if np.any(np.isnan(g)):
        raise ValueError('NaN in partial derivative!')

    return f,g


# +
def normal_distribution(get_info, input_params):
    if get_info == 'compute_densities':
        if len(input_params) <= 2:
            raise ValueError('Missing input parameters!')
        
        mu = input_params[0]
        y = input_params[1]
        dist_specific_params = input_params[2]
        clear input_params
        sigma = dist_specific_params['sigma']
        
        # Compute log_pdf http://en.wikipedia.org/wiki/Normal_distribution
        return np.sum(np.subtract((1 / np.power(sigma, 2)) * np.subtract(np.multiply(y, mu), np.add(np.multiply(.5,np.power(mu, 2)),
                    np.multiply(.5, np.power(y, 2))))), (.5 * np.log(2 * np.pi * np.power(sigma, 2))))
    
    elif get_info == 'fminunc_both_betas': # --> (2), This fetches the right function handle for the fminunc
        if len(input_params) <= 3:
            raise ValueError('Missing input parameters!')
        
        return lambda betas: fminunc_normal_both(betas, input_params[0], input_params[1], input_params[2], input_params[3])
    
    else:
        raise ValueError('Invalid operation!')
        
def fminunc_normal_both(betas, w, net_effects, dependent_var, dist_specific_params):
    
    # [F, G] = FMINUNC_NORMAL_BOTH(BETAS)
    # 
    # Purpose
    # 
    # To optimize logistic regression betas using cost function F
    #  
    # Input
    #
    # --betas: The current betas that were used to compute likelihoods
    # --w: Weight vector that holds the normalized weights for P particles
    # --net_effects: Predictor variable Matrix (number of trials x particles)
    # --dependent_var: Dependent variable Matrix (number of trials x 1)
    # --sigma: Used to specify variance in the Normal distribution
    # 
    # Output
    #
    # --f: Scalar, Objective function
    # --g: Vector of length 2 i.e. gradients with respect to beta_0 and beta_1
    
    beta_0 = betas[0]
    beta_1 = betas[1]
    sigma = dist_specific_params['sigma']
    
    mu = (beta_1 * net_effects) + beta_0
    
    # Cost function
    # We will need to maximize the betas but fminunc minimizes hence a -ve.
    # Here we compute the log pdf over all trials and then component multiply by the weights and then sum them up over all particles
    f = -np.sum(w * np.sum(np.subtract(np.multiply((1 / np.power(sigma, 2)), np.subtract(np.multiply(dependent_var, mu), np.add(
        np.multiply(.5,np.power(mu, 2)), np.multiply(.5,np.power(dependent_var,2))))), (.5 * np.log(2 * np.pi * np.power(sigma, 2))))))
    
    # Here we take the partial derivative of log pdf over beta_0 and beta_1 respectively, component multiply by the weights and sum them up over all paricles
    g = []
    g.append(-np.sum(w * np.sum((1 / np.power(sigma,2)) * np.subtract(dependent_var, np.add(beta_0, np.multiply(beta_1, net_effects))))))
    g.append(-np.sum(w * np.sum(np.multiply(np.divide(net_effects, np.power(sigma, 2)), np.subtract(dependent_var, 
                                np.add(beta_0, np.multiply(beta_1, net_effects)))))))
    
    if np.any(np.isinf(g)):
        raise ValueError('Inf in partial derivative!')
    if np.any(np.isnan(g)):
        raise ValueError('NaN in partial derivative!')
    
    return f,g
