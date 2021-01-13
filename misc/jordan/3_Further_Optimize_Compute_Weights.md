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

## Collapse The Arithmetic
So I can stick the `.5` multiplication term onto `tau`, so that all the arithmetic broadcasting only happens once.

```python
import numpy as np
from scipy import special
import math

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

## More Lesioning?


### It's Still Mostly The Last Term

```python
%%timeit 
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = (np.divide(x - mu, tau/np.sqrt(.5)) ** 2)
    return log_likelihood

compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### So Focus?

```python
%%timeit
def quickie(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')
        
    return ((x - mu) /  (tau/np.sqrt(.5))) ** 2

quickie(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit
def quickie(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')
        
    return ((x - mu) /  (tau/np.sqrt(.5)))

quickie(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit
def quickie(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')
        
    return ((x - mu))

quickie(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit
def quickie(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')
        
    return ((x))

quickie(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

The subtraction takes about 2ms. The division takes about 3ms. The power takes about 3ms. The rest of the computation takes about 2ms too.


## Settling
We got it down to around 2.5x slower than the original MATLAB code. Think I'll settle for 2.5x for now and finish the rest of the translation. When it's done, we can focus on optimization. 

```python

```
