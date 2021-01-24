# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PCITpy
#
# > The Probabilistic Curve Induction and Testing (P-CIT) toolbox, implemented in Python.

# %% [markdown]
# PCITpy is a Python-based reproduction of the the Probabilistic Curve Induction and Testing (P-CIT) toolbox. The toolbox was developed to estimate the shape of a curve relating a predictor variable to a dependent variable, such as fMRI classifier estimates of memory activation to subsequent memory recall). 
#
# P-CIT was initially designed specifcally to address a set of problems that came up in analyzing data from Detre et al (2013). Researchers wanted to estimate the shape of the function relating classifier output on no-think trials to performance on the final memory test, deriving an overall estimate of how well this function fit with their theory. However, in principle, the curve-fitting procedure they described can be used to estimate the shape of _any_ kind of function (so long as the function can be approximated by a piecewise linear function with 3 segments; though, even if it cannot, the code can be modified to work with more complex functions).
#
# A key benefit of the procedure is that it allows the user to estimate the posterior probability that the "true" underlying curve meets some (arbitrarily specified) set of conditions. There are already some existing ways to test for simple linear and quadratic relationships, but these methods fall short when the theory being tested predicts a more complex shape. For example, according to the theory being tested in the Detre et al. paper, the curve should dip below zero and then rise above zero; also, researchers are agnostic about the location of the rightmost point, so a curve is still theory-consistent if it dips back down after rising above zero. The method is capable of computing theory-consistency based on an arbitrarily complex set of criteria like these.
#
# While the original toolbox was realized as a collection of MATLAB files, we maintain PCITpy to make it easier for some researchers to take advantage of the toolbox's features. Along with the overall design of the toolbox, most of the language in this project's documentation is drawn directly from the [original project's docs](https://github.com/PrincetonUniversity/p-cit-toolbox) authored by Annamalai Natarajan, Samuel Gershman, Luis Piloto, Greg Detre, and Kenneth Norman with Princeton University. While our documentation aims to be self-sufficient to maximize toolbox accessibility, those docs represent the most comprehensive and accurate review of P-CIT's language-agnostic details.

# %% [markdown]
# ## Features

# %% [markdown]
# An explicit list of the tools P-CIT provides for researchers.

# %% [markdown]
# ## Example

# %% [markdown]
# A concise example illustrating  what P-CIT offers.

# %% [markdown]
# ## Installation

# %% [markdown]
# Explanation of how to install and set up P-CIT on a supported system.

# %% [markdown]
# ## Getting Started

# %% [markdown]
# Explaining how to quickly begin using the toolbox, including links to relevant docs.

# %% [markdown]
# ## Credits
# List contributors, references, license. Explain how to cite work.
