# January 23, 2021
So I want to develop a workflow for debugging/extending what we have of PCITpy. The key difference between what we have here and ICMR is that much of the codebase already exists, while with ICMR we developed notebooks incrementally.

There is no linear structure to what we're doing like there is with ICMR.

But we can readily enforce one: move what we have to a thirdbasis subdirectory, and progressively add content as it fulfills ths requirements of the main workflow.

In the end, I suspect my pattern of weird circular imports has to do with my effort to combine modules rather than the particular sequencing of notebook script compilation. 

So what will I do?
- Reset current set of scripts to another set aside directory.
- Add run_importance_sampler

...No, that seems a bit dramatic. We'll put scripts back in the sequence that makes sense (though we will also keep track of function dependencies) and iteratively set them aside or otherwise revise until the circularity bug is gone.

![](2021-01-23-06-17-57.png)

# January 24
Okay, let's get started. 

I'll maintain a list of issues to eventually address:
- ~~Percent format for script-based notebooks~~
- _Overview notebook for helper subsection_
- ~~Apparent circular dependencies depending on order of notebook scripts.~~
- No clear workflow for iterative improvements
- A world of code to add or test once I have that set up.
- Maybe simulate data actually comes first along with a discussion of the data format: a data preparation notebook.

### Circular Dependencies Issue
We have already renamed each file. Let's reset the various libraries and htmls, and see if I can pull off a `make`. 

The error is reproduced for a few notebooks:

```
converting: C:\Users\gunnj\Google Drive\pcitpy\Miscellaneous_Helper_Functions.ipynb
An error occurred while executing the following cell:
------------------
from nbdev.showdoc import show_doc
from pcitpy.pcitpy import *
------------------

---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-1-f1058062f9de> in <module>
      1 from nbdev.showdoc import show_doc
----> 2 from pcitpy.pcitpy import *

~\Google Drive\pcitpy\pcitpy\pcitpy.py in <module>
      6 # Cell
      7 # hide
----> 8 from .pcitpy import importance_sampler
      9 from scipy.io import loadmat
     10 import os

ImportError: cannot import name 'importance_sampler' from partially initialized module 'pcitpy.pcitpy' (most likely due to a circular import) (C:\Users\gunnj\Google Drive\pcitpy\pcitpy\pcitpy.py)
ImportError: cannot import name 'importance_sampler' from partially initialized module 'pcitpy.pcitpy' (most likely due to a circular import) (C:\Users\gunnj\Google Drive\pcitpy\pcitpy\pcitpy.py)
An error occurred while executing the following cell:
------------------
from nbdev.showdoc import show_doc
from pcitpy.pcitpy import *
------------------
```

It's odd that the bug would start with `Miscellaneous_Helper_Functions` given that the notebook doesn't import pcitpy at all. What is the "cell" being referred to in the traceback? Quite a mystery. 

Anyway, my hypothesis about the errors is that some of my notebooks with within-module dependencies simply must be isolated according to the original conventions of PCITpy. It's a bit awkward (a separate module for every MATLAB file?) but might be totally adequate for the situation.

That seems to fix! And it has a decent semantics; I'd rather be able to import directly from PCIT though. Oh well.

### Percent format for script-based notebooks
Just a couple of jupytext commands and confirmation that the result works.

### Overview notebook for helper subsection
I'll put off documentation issues for later; for now I need to demonstrate progress on the translation. I can already show off how nbdev solves our documentation and packaging concerns layed out months ago.

### No clear workflow for iterative improvements
So I embraced the idea before that our work on tests for PCIT can be concentrated on tests for the main pipeline (Data Simulation -> Parameter Configuration -> Preprocessing -> Curve Fitting -> Results Analysis). If we can confirm that the pipeline produces the same outputs across all meaningfully different parameter configurations, then the entire toolbox is probably bug free. This in turn implies that development (and testing) can proceed line by line along this main pipeline. 

A big thing I need to enforce is not discarding tests once passed; they need to persist as proof that the codebase is sound. When I have to refactor bits of the codebase to support testing, I can either make that refactoring a permanent feature of the codebase (and include them in their respective notebooks, as hidden or explicit code cells), or I can include supplementary material reporting my results. But in the end, I definitely need a notebook that outlines and reports the results of my testing scheme. 

An odd feature of my testing framework is comparison with MATLAB outputs. Normally, I present my tests as demos in the documentation: they showcase the function while confirming it works. Adding MATLAB comparisons makes the documentation a running confirmation that the toolboxes are equivalent. I don't know if I really like that. These _can_ be hidden cells, but maybe I do want pages doing these MATLAB-based tests, even if they aren't the main testing framework. Maybe I can have a notebook that reports these tests too; it would also prove a helpful tutorial of that workflow. Okay.

In the meantime, though, hidden cells in the same contexts as the functions' specifications.

What do I do now? Start from notebook 0!