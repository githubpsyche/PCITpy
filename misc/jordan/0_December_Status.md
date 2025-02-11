# Status Overview - December 2020
I'm trying to figure out where we are in this project as of December 2020 and
maybe introduce a better strategy for finally finishing it - notebook-driven
development. 

A lot of progress has already been made on PCITpy. But returning to the project,
I see a lot of isolated files and isolated Github issues that, while clear that
I have some things done and some things to do, don't really provide a clear
direction about which particular thing I should be doing now. So I'm reviewing
all that here. 

## A Try at Notebook-Driven Development

A problem I often face with larger projects that by virtue of the time they take
to finish have to slip in and out of attention as work goes on. Forgetting about
a project so you can spend some time on something else comes with a tax: on top
of all the work you still have left to do for the project, you also have to
spend some time reminding yourself what the work is, your approach to it, and so
on. The part of a project where you don't know what you're doing is maybe where
it's the most intimidating, and potentially the root of lots of procrastination
that can keep projects without strong deadlines driving them from ever getting
done. Efforts to document problems and goals with feature's like Github's Issue
tracker help a little with this, but they all present remaining work as a
discretized list of tasks, and hide or de-emphasize what's already been done. 

An approach I've found useful for avoiding this issue is to make all development
happen in the context of an explicit coherent narrative, initially encoded
within a computational notebook. That's pretty similar to the paradigm of
[literate programming](https://en.wikipedia.org/wiki/Literate_programming),
which uses computational notebooks to produce works that simultaneously serve as
source code and documentation. Code documentation is primarily supposed to
instruct: explain what a software does, is for, and how to use it. Rather than
combining source code development with documentation, I'm looking more for a
record that organizes and contextualizes work on a project, making it easy to
grasp its overall course and the mindsets/ideas/strategies we've picked up along
the way to define that course - much like the [laboratory
notebooks](https://en.wikipedia.org/wiki/Lab_notebook) popular in the hard
sciences.

The final product of all our work will be refactored into a Python package that
makes it easy for anyone to use what we've made. But while we're working on all
this, we'll organize our code in a way optimized for development, rather than
distribution. New increments will be structured like discrete chapters or
subsections of chapteres embedded in a broader story. They'll be contextualized
with natural language, including clear explanations of how they connect with
what's already been done, and with what will come next. Later on, someone trying
to grasp where the project is at (probably one of us again after a long break)
should be able to check out the newest few chapters of the text and have a clear
idea, even if they aren't experts at interpreting code.

## P-CIT Roll Call

The original version of P-CIT comes with a pretty detailed manual for its use,
outlining the purpose of each function.

1. `run importance sampler.m`— This is the driver routine that you will use to
   load your data matrix and also set parameters for the curve-fitting
   procedure.
2. `preprocessing setup.m`— In this routine, we perform a whole suite of sanity
   checks on both the data matrix and the parameter settings, and we pre-process
   the data (drop outliers, z-score,etc).
3. `importance sampler.m`— This file implements the actual curve-fitting
   procedure.
4. `family of curves.m`— This function has curve-specific information like the
   number of curve parameters, individual curve parameter boundaries, theory
   consistency criteria, and the code to compute likelihoods i.e. P(Y|X, β, θ),
   where Y is dependent variable, X is predictor variable, β is the coefficient
   and θ is the set of curve parameters.
5. `family of distributions.m`— This function contains the probability density
   computation, the distribution-specific fminunc to optimize the objective
   function, and the associated partial derivatives for each of the
   dependent-variable distributions that are currently covered by the toolbox
   (i.e., Bernoulli, normal).
6. `common to all curves.m`— This function provides information that is common
   to all family of curves (in case you have multiple families of curves) like
   initial uniform sampling, checks to see if curve parameters exceed bounds,
   etc
7. `analyze outputs.m`— This function plots the weighted curve along with the
   credible interval. Refer to Figure 1 for an example plot. It gets all the
   analysis related information from the .mat file created by importance
   sampler.m. By default the credible interval is set to be 90%. It also plots
   results from bootstrap and scramble analyses. 8. `simulate data.m`— This
   function creates simulated data. This function gets the curve related
   information from family of curves.m and common to all curves.m. You can
   change the number of subjects, items per subject and other defaults in the
   simulate data.m to be similar to your dataset. This function generates
   simulated data from both Bernoulli and normal distributions. 
9. `logsumexp.m`— This function takes a sum of the log values while avoiding
   numerical overflows. We use this function in our toolbox for log likelihood
   computations. Code from Tom Minka, used here under the MIT license. 
10. `round to.m`— This function rounds scalars / vectors to d number of decimal
    places.
11. `scale data.m`— This function scales data between lower l and upper u
    bounds. We use this function in our toolbox to scale the predictor variable
    between 0 and 1. 
12. `truncated normal.m`— This function samples data from a truncated Gaussian
    distribution between lower l and upper u bounds with mean = µ and std
    deviation σ. We use this function in our toolbox to add noise from a
    truncated Gaussian distribution. 
13. `jbfill.m`— This functions shades the area between two lines on a plot. We
    use this function in our toolbox to plot credible intervals. Code from John
    Bockstege, used with permission. 
14. `savesamesize.m`— This function saves the figure at the same size as it was
    displayed on screen. We use this function in our toolbox to plot simulated
    data per subject over multiple subjects in a subplot setup. Code from
    Richard Quist, used here under the BSD license. 
15. `likratiotest.m`— This function implements the β1 likelihood-ratio test
    described in Section 4.10.

Which of these do we already have fully implemented and tested?

## Translation Status

### Basic Helper Functions
These came first; they're pretty simple and help out with operations across the
toolbox. The following scripts are all completed:
- `logsumexp`
- `round_to`
- `scale_data`
- `truncated_normal`
- `likratiotest`

Two more scripts are not completed, but likely don't need parallel
implementations in Python. They help with visualizing the outputs of PCIT;
Python's chief figure visualization package `matplotlib` may not need that extra
support. We won't really know until we make more progress translating the more
central output visualization function `analyze_outputs`:
- `jbfill`
- `savesamesize`

### Setup Functions

These functions run early in the P-CIT pipeline and do a lot of preprocessing to
the input data. We had to translate these early since their outputs are the
inputs to more core P-CIT functions. They're all done:
- `run_importance_sampler` 
- `preprocessing_setup`

### Major Helper Functions

These files provide more elaborate support to the core P-CIT functions. They're
basically libraries of functions themselves, so it might be better to think more
on the level of these sub-functions instead of as a few files. They should
probably be re-organized into discrete functions with their own documentation
for the Python translation, too.

#### common_to_all_curves

- `get_nParams`
- `get_bounds`
- `get_vertical_params_only`
- `get_horizontal_params_only`
- `compute_likelihood`
- `count_particles`
- `get_curve_xy_vals`

While the initial several of these subfunctions are trivial and have already
been translated, the latter three are longer and more tricky. Initial
translations have been developed, but `count_particles` and `get_curve_xy_vals`,
subfunctions specific to `analyze_outputs`, need further testing.

#### family_of_distributions

- `bernoulli_distribution`
- `fminunc_bernoulli_both`
- `normal_distribution`
- `fminunc_normal_both`

Has four subfunctions providing calculations related to Bernoulli and Normal
distributions. Functions related to Bernoulli distributions are apparently fully
implemented and tested, but functions related to Normal distributions have not
been validated yet.

### common_to_all_curves

Rather than distribution specific information as above, this function grabs
information common to all curves. That's apparently a lot of kinds of
information.

- `initial_sampling`
- `check_if_exceed_bounds`
- `curve_volumes`
- `flip_vertical_params`
- `sort_horizontal_params`
- `draw_bcm_params`
- `auto_generate`
- `weighted_curve`

`weighted_curve` has not been implemented yet. 

On the other hand, `initial_sampling`, `check_if_exceed_bounds`,
`sort_horizontal_params` all seem fully validated.

This leaves `curve_volumes`, `flip_vertical_params`, `draw_bcm_curve`,
`auto_generate` along with `weighted_curve` unfinished.

### Core Functions
These are the central routines carrying out most of what P-CIT is for. They
depend on the successful translation of most other functions in the toolbox and
are pretty long functions in their own right, so they take the longest to
translate and will be finished last. Neither is totally done yet. Some also
include multiple functions:
- `simulate_data` - Not even mentioned in the manual, but important for
  evaluating the accuracy of our translations.
- `importance_sampler` - The function that actually does P-CIT things.
- Also has subfunctions `compute_weights` 
- and `compute_trunc_likes`
- `analyze_outputs` - Turns the output of the importance sampler into
  interpretable analysis outcomes
- Also has a `plot_sim_data` subfunction.

## Translation Approach
So there's still a lot of translation and validation to do? Where does one
start, and after that how does one decide what to do next?

At some point along all this, I worked from the smallest functions up towards
the biggest ones, identifying test cases and writing code to pass them. This was
a lot of work, because it meant I had to both translate MATLAB code and write
more code testing the translations. 

Visual debuggers like those provided by Pycharm and MATLAB, though, make room
for a different approach that doesn't require developing new testing code.
Instead, as long as valid inputs to core P-CIT functions are available, program
behavior can be inspected at any point along its operation. This enables line by
line comparison of translated code to the original. Since all supporting
functions are used in one way or another by core P-CIT functions, they just
ended up being tested as core P-CIT functions are tested. These results don't
necessarily confirm that the codebases are equivalent, but over enough valid
inputs it at least offers a lot empirical reassurance that the code is properly
translated.

According to this scheme, we pick arbitrary data inputs and focus translation on
core functions and translate helper functions in the order that they are called
in these core functions. When co-working, we can work on separate core functions
(e.g. one on `importance_sampler` and another on `analyze_outputs`) as long as
appropriate inputs can be generated or found for them.

Progress on translation can thus be summarized with a particular input data file
and line number indicating how far along a translated function variable values
match those associated with corresponding MATLAB code. Any helper functions used
up to that point are considered at least partially validated.

## So Where Are We Now?

Across my old notes, I see the line number `143` singled out. That's the point
where beta weights are fitted based on the data and the current iteration of the
importance sampler, whatever that means. 

I had some issues with speed and memory then. I met an `OutOfMemory` error that
never happened at a corresponding point in the MATLAB code. And running time was
a lot higher than corresponding MATLAB code, too. That was a different kind of
problem than just making sure outputs matched for similar inputs, while at the
same time making further testing intolerably slower. Resolving that problem is
the next big step for this effort.