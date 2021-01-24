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

We have already renamed each file. Let's reset the various libraries and htmls, and see if I can pull off a `make`. 