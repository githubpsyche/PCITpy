# January 25, 2021
My caching workflow need work. It's clunky and difficult to read. To make things simpler, I can just put the caching and the work under ongoing development into different notebooks. That way, I don't have to subdivide `importance_sampler` and caching commands so much on the one hand, and all those operations can be held separately from ongoing development on the other. 

The notebook with cached results will be what becomes our final implementation of our Curve Fitting notebook. We'll maintain a separate Curve_Fitting_Development notebook that defines our workspace for ongoing development. In the meantime, we'll rename the trainwreck notebook Curve_Fitting_Off and delete it at first opportunity.

Ugh, this failed. I'm just going to debug for now until I reach a point where caching is necessary. Then I will use a solution that avoids Jupyter cell magics, which don't play well with VSCode's main debugger.

## Running Issue List
- ~~I could probably time pre-compute_weights to see if there's a huge matlab advantage in that area as well~~
- Compute_Weights!!!

## Before Compute_Weights
Everything up to the `compute_weights` call takes only 4 minutes. That's annoying sure, but the core bottleneck remains this compute_weights function. 

I can start the MATLAB version _after_ the python version and hit the compute_weights bottleneck a noticeable amount of time before the python version does the same. Maybe it's unrealistic to hope the Python version can be faster, idk. Still, I at least need something in the general neighborhood - even taking twice as much time to handle the bottleneck would be tolerable.

## Compute_Weights
As always, the function takes far too long, even with my compute_trunc_likes function substantially optimized with numba. Is there anything I can do to optimize further? Is compute_trunc_likes still the bottleneck in the way of MATLAB-like speeds, or could including more operations under the njit umbrella be the key to escaping these speed doldrums?

And with the function taking so long to run, is there a sustainable caching strategy that avoids the huge runtimes?

How many idxs does the Python version manage in 5 minutes, compared to MATLAB?

|   | Python | Matlab
| - | ------ | ------
| 5 min | 963 idx | 3946 idx
| 1 min | 192.6 idx | 789.2 idx
| 1 idx | 0.0052 min | 0.0013 min
| 5000 idx | 25.96 min | 6.34 min
| 2 iter | 51.92 min | 12.68 min
| 20 iter | 519 min | 126 min
| 20 iter | 8.65 hrs | 2.1 hrs

That's depressing! Why am I spending so long on a toolbox that takes 8 hours to run while a toolbox that takes just 2 hours exists? I need it below 4 hours for me to feel okay with myself. 

Is the structure of the output appropriate? As far as I can tell.

### Further Development
How do I make progress on this? So far I've focused optimization efforts on compute_trunc_likes exclusively. All while nonetheless timing compute_weights for comparison with the MATLAB code. How about I scale up my numba optimization to include the inner loops?

```
      prob_grp_lvl_curve = np.zeros((nParticles, reduced_nParticles))
      target_indices = np.arange(
         reduced_nParticles_idx[0, idx], reduced_nParticles_idx[1, idx])
      for npm in range(nParam):
         which_param = npm
         nth_grp_lvl_param = np.tile(
               param[:, npm].reshape(-1, 1), (1, reduced_nParticles))
         nth_prev_iter_curve_param = prev_iter_curve_param[target_indices, npm]
         trunc_likes = compute_trunc_likes(nth_grp_lvl_param, nth_prev_iter_curve_param, tau, bounds, which_param)
         prob_grp_lvl_curve = np.add(prob_grp_lvl_curve, trunc_likes)
```

This would make the arguments to the bigger function:
- idx
- nParticles
- reduced_nParticles
- reduced_nParticles_idx
- nParam
- param
- prev_iter_curve_param

And still weird globals like:
- tau
- bounds

Let's see if I can pull this off without loads of refactoring. Uhh, seems hard given that I have to do a lot of numba shit.

```python
pickle.dump({'idx': idx, 'nParticles': nParticles, 'reduced_nParticles': reduced_nParticles, 'reduced_nParticles_idx': reduced_nParticles_idx, 'nParam': nParam, 'param': param, 'prev_iter_curve_param': prev_iter_curve_param, 'tau': tau, 'bounds': bounds}, open('for_testing_compute_weights', 'wb'))

```

Running that seems to work in the debugging context. While

```python
pickle.load(open('for_testing_compute_weights', 'rb'))
```

Successfully loads the file.

I'll put compute_weights in its own notebook, maybe even its own module