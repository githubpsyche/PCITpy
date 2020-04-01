<!--
marp: true
headingDivider: 2
-->

# PCITpy
The Probabilistic Curve Induction and Testing (P-CIT) toolbox, implemented in Python.

## File Organization

### `pcitpy` and `original`
The `pcitpy` directory includes Python implementation code while the `original` directory contains the original MATLAB code (potentially with additional code to serve testing.)

Both directories are necessary in order to execute tests helping to confirm equivalence between our Python and MATLAB implementations of P-CIT (or for troubleshooting when equivalence doesn't happen).

### `data` and `results`
Data and results directories contain `.m` files useful for testing equivalence between MATLAB and Python codebases on complex input. Current files are sourced from [the LewisPeacockLab's attempt to translate P-CIT to Python](https://github.com/LewisPeacockLab/PCITpy).

## Testing Framework
For every function in the MATLAB implementation of P-CIT, we make an equivalent one in Python within the `pcitpy` directory.

Within each function file, we include additional testing code - for example, `test_preprocessing_setup` for `preprocessing_setup.py`.

Test code generally instantiates the Python and MATLAB versions of the relevant function, runs the functions on reasonable inputs, and compares outputs. When randomness plays a role in function output, output is usually visualized using `matplotlib`. 

To control MATLAB functionality within Python, we leverage the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

# Current Progress as of 4/1/2020
After reviewing and developing a clearer testing framework for the Python codebase, it seems that PCITpy is around 70-80% complete.

### Helper Functions
- `round_to`, `logsumexp`, `likratiotest`, `scale_data`, `truncated_normal` are all fully implemented and tested in Python.

### Visualization Functions
- `analyze_outputs` and `jbfill` are not yet implemented.
- `savesamesize` probably doesn't need a Python analogue at all, though I may be wrong.

## Core Functions
- `run_importance_sampler` and `preprocessing_setup` are all fully implemented and tested in Python. 
- `importance_sampler`, `family_of_distributions`, `common_to_all_curves`, and `family_of_curves` are implemented but not fully tested and thus are certainly still quite buggy.
- `simulate_data` is not yet implemented.

## Summary
Core functions are implemented but still need to be substantially tested/bug-fixed; after that, visualization functions can be implemented and tested. Seems comfortably doable within 1-2 weeks?
