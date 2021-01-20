# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3.8 (XPython)
#     language: python
#     name: xpython
# ---

# +
# hide
# default_exp helper
# -

# # Helper Functions

# +
#exports
from scipy import stats


def likratiotest(l1, l0, k1, k0):
    D = -2 * (l0 - l1)  # deviance score
    df = k1 - k0  # degrees of freedom
    p = stats.chi2.sf(D, df)  # chi-square test
    return D, p
