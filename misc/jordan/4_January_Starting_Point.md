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

## Arlene's Updates
- `simulate_data` started
- `family_of_curves.get_curve_xy_vals` finished
- Keeping her own notebook(s) to track progress
- Shared outcomes of simulations in `results` folder


## Plan
Now my goal is to set up some code to help debug past that bottleneck of code I've been stuck on. I'll:
- Run PCIT up past 2 iterations of the importance sampler and save the state of all variables at that point and 
- Make a function that just loads those variables where it would normally run the importance sampler

That way I can keep working on the rest of the code without being so tormented by long runtimes.


## Outcome
I find a bug in the output of my `compute_weights` function: `p_theta_minus_q_theta` has the wrong shape, 20 x 100k instead of 1 x 100k. Obviously my hope is that this reflects an opportunity to further optimize compute weights, but it's probably a shape issue with a sum I did at some point in the calculation. Once I work that out (on line 131), I'll be able to move forward with my plan to sidestep runtime issues I might face with further code testing.
