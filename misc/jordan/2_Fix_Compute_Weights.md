---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Let's Make the Call to compute_trunc_likes not so slow


## Setup

```python
import numpy as np
from scipy import special
import math
```

```python
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(np.divide(x - mu, tau), 2))
    return log_likelihood
```

```python
tau = .05
bounds = np.array([[-1,  1],
       [ 0,  1],
       [ 0,  1],
       [-1,  1],
       [-1,  1],
       [-1,  1]])
which_param = 0

nth_grp_lvl_param = np.load('../nth_grp_lvl_param.npy')
nth_prev_iter_curve_param = np.load('../nth_prev_iter_curve_param.npy')
```

## Original Call

```python
test = np.array([compute_trunc_likes(nth_grp_lvl_param[:, i], nth_prev_iter_curve_param[i])
                                    for i in range(len(nth_prev_iter_curve_param))]).T
test
```

```python
np.shape(test)
```

```python
test[0]
```

```python
%%timeit
test = np.array([compute_trunc_likes(nth_grp_lvl_param[:, i], nth_prev_iter_curve_param[i])
                                    for i in range(len(nth_prev_iter_curve_param))]).T
```

## Vectorize?
Instead of acting over a single mu or column, I need operations that use the entire array such that `mu` is a vector and `x` is a 2-dimensional array.

```python
mu = nth_prev_iter_curve_param
x = nth_grp_lvl_param

log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
    -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
        -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
        -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(np.divide(x - mu, tau), 2))

log_likelihood
```

```python
np.shape(log_likelihood)
```

```python
log_likelihood
```

```python
log_likelihood[0]
```

```python
%%timeit
mu = nth_prev_iter_curve_param
x = nth_grp_lvl_param

log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
    -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
        -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
        -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(np.divide(x - mu, tau), 2))
```

It doesn't seem the code needs further alterations to support vectorization.


## Result
Let's test a call to compute_trunc_likes using the arrays.

```python
result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
result
```

```python
np.shape(result)
```

```python
%%timeit
result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

## This Needs to Be Several Times Faster. Where is the Bottleneck?

```python
%%timeit

# get rid of first term
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(np.divide(x - mu, tau), 2))
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit

# then the second hidden in the log term
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log((np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(np.divide(x - mu, tau), 2))
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit

# the entire second log term
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) ) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(np.divide(x - mu, tau), 2))
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2))))))))- np.multiply(.5, np.power(np.divide(x - mu, tau), 2))
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit

# the final term, the only one using x
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi))
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

It's the last term. I need to focus all efforts on optimizing the last term. Any  particular part?

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.power(np.divide(x - mu, tau), 2)
    return log_likelihood


result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x - mu, tau))
    return log_likelihood


result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(x - mu, 2))
    return log_likelihood


result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(np.divide(x, tau), 2))
    return log_likelihood


result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

## The Bottleneck is np.power
It's about 82ms with the full equation. The full term is `np.multiply(.5, np.power(np.divide(x - mu, tau), 2))`. Without it, speed is 19.8us. The contribution of each part?

Removing np.multiply obtains a runtime of 78.5ms.
Removing np.power obtains 22.1ms.
Removing divide gets 79.7ms.
Removing x-mu gets 77.6.

So the most impactful statement is np.power. Can I make it faster?

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.power(np.divide(x - mu, tau), 2))
    return log_likelihood


result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x - mu, tau) ** 2)
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

That's nice but I still need it to be 4 times faster to be MATLAB tier. 


## What's the Top Bottleneck Now?
The full term is now `np.multiply(.5, np.divide(x - mu, tau) ** 2)`. Takes 26.2ms to run. I need it down to 6.5ms. 

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x - mu, tau) ** 2)
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### Lesioning

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi))
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

remove multiply

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.divide(x - mu, tau) ** 2)
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

remove divide

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, (x - mu) ** 2)
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

remove sum

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x, tau) ** 2)
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

remove power

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x - mu, tau))
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

Runtime is similar across all operations now. While removing them all obtains huge gains, removing any individual operation only gives about 5ms. 

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (x - mu)
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

Even if it were just an addition operation, it'd take 12ms instead of my 6.5.


## The Number Precision Experiment
Let's take a step back and see if PCIT does ok with lower number precision.

```python
nth_grp_lvl_param = nth_grp_lvl_param.astype('float32')
nth_prev_iter_curve_param = nth_prev_iter_curve_param.astype('float32')
```

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x - mu, tau) ** 2)
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

Half precision grants half-speed runtime. Does PCIT work fine that way?

```
Start time 12/17 9:53
********** START OF MESSAGES **********
0 trials are dropped since they are regarded as outliers
********** END OF MESSAGES **********
Betas: 0, 1
EM Iteration: 0
Optimization terminated successfully.
         Current function value: 1243.911318
         Iterations: 6
         Function evaluations: 8
         Gradient evaluations: 8
Betas: 0.10592798970016724, 0.6316629314995463
EM Iteration: 1
```

Not too far off from Betas: 0.0882896038554186, 0.6902758140586037. But definitely meaningfully off. And the extra benefit isn't double. I found an improve from 4x slower to just 3x slower. So I'm ambivalent about the idea. But at least I've figured out how to institute it.


## Can We Vectorize The Loop Over Params?
To vectorize, I have to use every row of `param` at once.
`nth_grp_lvl_param` is just a selected column of param, repeated for 20 columns. Could I avoid doing that?
`idx` selects a subset of rows of `prev_iter_curve_param` to fill `nth_prev_iter_curve_param` while `npm` similarly picks the column filling the variable.

`nth_grp_lvl_param` has shape 100,000x20, while `prev_iter_curve_param` has shape 20. Each entry in `prev_ter_curve_param` is broadcast to each column in `nth_grp_lvl_param` during all this math - well, after a bit of computation on those entries.

```python
nParam = 6
tau = .05
bounds = np.array([[-1,  1],
       [ 0,  1],
       [ 0,  1],
       [-1,  1],
       [-1,  1],
       [-1,  1]])
which_param = 0
idx = 0
reduced_nParticles = 20

nth_grp_lvl_param = np.load('../nth_grp_lvl_param.npy')
nth_prev_iter_curve_param = np.load('../nth_prev_iter_curve_param.npy')
nth_prev_iter_curve_param
```

```python
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x - mu, tau) ** 2)
    return log_likelihood

result = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
alt_nth_grp_lvl_param = nth_grp_lvl_param
alt_nth_prev_iter_curve_param = np.tile(nth_prev_iter_curve_param, (6,1))
```

```python
compute_trunc_likes(alt_nth_grp_lvl_param, alt_nth_prev_iter_curve_param.T)
```

```python
np.shape(alt_nth_prev_iter_curve_param)
```

```python
nth_prev_iter_curve_param
```

```python
np.shape(compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param))
```

So I can get the same result using just a single column of `nth_grp_lvl_param` and a transpose. Could that improve runtime? How do I find out? Compare runtime against using the full number. It actu

```python
for npm in range(nParam):
    which_param = npm
    nth_grp_lvl_param = np.tile(
        param[:, npm].reshape(-1, 1), (1, reduced_nParticles))
    nth_prev_iter_curve_param = prev_iter_curve_param[target_indices, npm]
    trunc_likes = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
    prob_grp_lvl_curve = np.add(prob_grp_lvl_curve, trunc_likes)

    if np.any(np.isnan(prob_grp_lvl_curve)):
        raise ValueError('NaNs in probability of group level curves matrix!')
```

## Can I Optimize The Other Operations?

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x - mu, tau) ** 2)
    return log_likelihood

compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
#%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (.5 * (((x - mu)/ tau) ** 2))
    return log_likelihood

compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

## Focus On Just The Time-Consuming Term

```python
%%timeit
def quickie(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')
        
    return np.multiply(.5, np.divide(x - mu, tau) ** 2)

quickie(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit
def quickie(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')
        
    return (((x - mu)/ constant) ** 2)

quickie(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
constant = tau/np.sqrt(.5)
```

So I can stick the `.5` multiplication term onto `tau`, so that all the arithmetic broadcasting only happens once.


## Collapse The Arithmetic
So I can stick the `.5` multiplication term onto `tau`, so that all the arithmetic broadcasting only happens once.

```python
tau = .05
bounds = np.array([[-1,  1],
       [ 0,  1],
       [ 0,  1],
       [-1,  1],
       [-1,  1],
       [-1,  1]])
which_param = 0

nth_grp_lvl_param = np.load('../nth_grp_lvl_param.npy')
nth_prev_iter_curve_param = np.load('../nth_prev_iter_curve_param.npy')
```

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - np.multiply(.5, np.divide(x - mu, tau) ** 2)
    return log_likelihood

compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
nth_grp_lvl_param = nth_grp_lvl_param.astype('float32')
nth_prev_iter_curve_param = nth_prev_iter_curve_param.astype('float32')
```

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, special.erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))) + (np.multiply(-0.5, special.erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, math.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.divide(x - mu, tau/np.sqrt(.5)) ** 2)
    return log_likelihood

compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```


