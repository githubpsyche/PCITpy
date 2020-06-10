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