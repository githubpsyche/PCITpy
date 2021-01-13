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

# Optimization With Numba
My approach so far has focused on working out the details of numpy, but there seems to be a fundamental limit to what I can achieve that way. On the other hand, there seems to be a small side issue that I noted in my last update: 

> I found a bug in the output of my `compute_weights` function: `p_theta_minus_q_theta` has the wrong shape, 20 x 100k instead of 1 x 100k. Obviously my hope is that this reflects an opportunity to further optimize compute weights, but it's probably a shape issue with a sum I did at some point in the calculation.

My next increment for PCITpy should be to resolve this issue and explore the use of numba (or, failing that, Cython) to address these sorts of bottlenecks in my code. I'll also ditch use of `float32` for optimization, as it certainly does change the behavior of my program.


## Setup
Loading relevant packages and data.

```python
import numpy as np
from scipy import special
from numba import jit
import math

tau = .05
bounds = np.array([[-1,  1],
       [ 0,  1],
       [ 0,  1],
       [-1,  1],
       [-1,  1],
       [-1,  1]])
which_param = 0

nth_grp_lvl_param = np.load('../../nth_grp_lvl_param.npy')
nth_prev_iter_curve_param = np.load('../../nth_prev_iter_curve_param.npy')
```

## Starting Point
The current version of the function prior to further optimization.

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

## Just Adding the Decorator

```python
%%timeit 
@jit(nopython=True)
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

`numba` complains about `special.erfc`. Kind of odd! But resources like [this one](https://github.com/numba/numba/issues/3086) suggest that `numba` just doesn't support that function. Okay. It does apparently support math.erfc. Should I try it?


### Comparing math and special erfc

```python
%%timeit
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    special.erfc(-np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))
    
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit
import math
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    math.erfc(-np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))
    
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

`math.erfc` is not vectorized. I need a vectorized version of `erfc`. Apparently I'm able to make one. Okay...

```python
from numba import vectorize, float64, njit

@vectorize([float64(float64)])
def erfc(x):
    return math.erfc(x)
```

```python
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    erfc(-np.divide(bounds[which_param, 1] - mu, np.multiply(tau, math.sqrt(2))))
    
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

```python
%%timeit
    
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

My vectorized `erfc` function is somewhat faster than the original `special.erfc` function.


## Decorator With Vectorized math.erfc

```python
import time

@njit(float64[:,:](float64[:,:], float64[:]), nogil=True,parallel=True, fastmath=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

Without parallelization, it takes a lot longer - almost ten times! With parallelization, it's 32 ms, compared to my original 26ms. And `fastmath` does nothing. Neither does `nogil`. Neither does eager compilation (telling Numba the function signature I'm expecting). How odd! Stuff that causes a switch to object mode actually speeds up my code, sadly. 


When I switch back to numpy functions for my squaring, square roots, etc, performance improves substantially. It seems I should almost default to numpy when using jit. Runtime is now 7ms!


# Review
Let's reconsider all those settings I explored for this.


## Vanilla njit

```python
import time 

@njit
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

## Just One Option


### njit with eager compilation alone

```python
@njit(float64[:,:](float64[:,:], float64[:]))
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with parallel alone

```python
@njit(parallel=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with nogil alone

```python
@njit(nogil=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with fastmath alone

```python
@njit(fastmath=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

## Two Options


### njit with eager compilation and nogil

```python
@njit(float64[:,:](float64[:,:], float64[:]), nogil=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with eager compilation and parallel

```python
@njit(float64[:,:](float64[:,:], float64[:]), parallel=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with eager compilation and fastmath

```python
@njit(float64[:,:](float64[:,:], float64[:]), fastmath=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with parallel and fastmath

```python
@njit(parallel=True, fastmath=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with parallel and nogil

```python
@njit(parallel=True, nogil=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with nogil and fastmath

```python
@njit(nogil=True, fastmath=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

## All But One


### njit with eager compilation and nogil and parallel

```python
@njit(float64[:,:](float64[:,:], float64[:]), nogil=True, parallel=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with eager compilation and nogil and fastmath

```python
@njit(float64[:,:](float64[:,:], float64[:]), nogil=True, fastmath=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with eager compilation and parallel and fastmath

```python
import time

@njit(float64[:,:](float64[:,:], float64[:]), parallel=True, fastmath=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

### njit with parallel and nogil and fastmath

```python
@njit(float64[:,:](float64[:,:], float64[:]), parallel=True, nogil=True, fastmath=True)
def compute_trunc_likes(x, mu):
    global tau, bounds, which_param

    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param)
```

## njit with eager compilation and parallel and nogil and fastmath

```python
import time
from numba import int64

@njit(float64[:,:](float64[:,:], float64[:], float64, int32[:,:], int64), parallel=True, fastmath=True)
def compute_trunc_likes(x, mu, tau, bounds, which_param):
    
    if tau <= 0:
        raise ValueError('Tau is <= 0!')

    # This ugly thing below is a manifestation of log(1 ./ (tau .* (normcdf((bounds(which_param, 2) - mu) ./ tau) -
    # normcdf((bounds(which_param, 1) - mu) ./ tau))) .* normpdf((x - mu) ./ tau)) Refer to
    # http://en.wikipedia.org/wiki/Truncated_normal_distribution for the truncated normal distribution
    log_likelihood = -(np.log(tau) + np.log(np.multiply(0.5, erfc(
        -np.divide(bounds[which_param, 1] - mu, np.multiply(tau, np.sqrt(2))))) + (np.multiply(-0.5, erfc(
            -np.divide(bounds[which_param, 0] - mu, np.multiply(tau, np.sqrt(2)))))))) + np.multiply(
            -.5, np.log(2) + np.log(np.pi)) - (np.square(np.divide(x - mu, tau/np.sqrt(.5))))
    return log_likelihood

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param, tau, bounds, which_param)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param, tau, bounds, which_param)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
```

```python
%%timeit 
compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param, tau, bounds, which_param)
```

So nogil probably does nothing but the rest of the options contribute some amount of performance. How much does this scale within the actual codebase?


## Integrating Into PCIT
I've already made an attempt, adding my vectorized `erfc` function with my new `compute_trunc_likes` and dependencies. Let's see if it works and compare performance with MATLAB.

First, let's time the MATLAB code.

```python
tau.dtype
```

## Still More to Do
I find that MATLAB gets 3795 iterations of compute_weights calls done in 5 minutes. Python is still only getting 938 iterations done. I might try going back to lower number precision. Could also try lesioning to find bottlenecks. That bug I detected as happening after all these iterations might also be an important clue.

```
Start time 1/4 3:8
********** START OF MESSAGES **********
0 trials are dropped since they are regarded as outliers
********** END OF MESSAGES **********
Betas: 0, 1
EM Iteration: 0
Optimization terminated successfully.
         Current function value: 1243.992529
         Iterations: 5
         Function evaluations: 7
         Gradient evaluations: 7
Betas: 0.10286141975053381, 0.6417252461065261
EM Iteration: 1
2021-01-04 03:12:00.616726
2021-01-04 03:17:11.770227
938
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
<ipython-input-6-1f3b069d409b> in compute_weights(curve_name, nParticles, normalized_w, prev_iter_curve_param, param, wgt_chunks, resolution)
     24                 nth_prev_iter_curve_param = prev_iter_curve_param[target_indices, npm]
---> 25                 trunc_likes = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param, tau, bounds, which_param)
     26                 prob_grp_lvl_curve = np.add(prob_grp_lvl_curve, trunc_likes)

KeyboardInterrupt: 

During handling of the above exception, another exception occurred:

AssertionError                            Traceback (most recent call last)
<ipython-input-9-2bc4507d9022> in <module>
      1 # run tests only when is main file!
      2 if __name__ == '__main__':
----> 3     test_importance_sampler()

<ipython-input-4-0f527f394b82> in test_importance_sampler()
     18 
     19     # generate output
---> 20     importance_sampler(python_data, python_analysis_settings)
     21     # eng.importance_sampler(matlab_data, matlab_analysis_settings, nargout=0)

<ipython-input-2-fd0875ea2295> in importance_sampler(***failed resolving arguments***)
    114         # Compute the p(theta) and q(theta) weights
    115         if em > 0:
--> 116             p_theta_minus_q_theta = compute_weights(
    117                 ana_opt['curve_type'], ana_opt['particles'], normalized_w[em - 1, :],
    118                 prev_iter_curve_param, param, ana_opt['wgt_chunks'], ana_opt['resolution'])

<ipython-input-6-1f3b069d409b> in compute_weights(curve_name, nParticles, normalized_w, prev_iter_curve_param, param, wgt_chunks, resolution)
     33         print(datetime.datetime.now())
     34         print(idx)
---> 35         assert(False)
     36 
     37     if np.any(np.isnan(q_theta)):

AssertionError: 
```

```python

```
