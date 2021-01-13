# So Where Are We Now?
Across my old notes, I see the line number `143` singled out. That's the point where beta weights are fitted based on
the data and the current iteration of the importance sampler, whatever that means. 

I had some issues with speed and memory then. I met an `OutOfMemory` error that never happened at a corresponding point
in the MATLAB code. And running time was a lot higher than corresponding MATLAB code, too. That was a different kind of
problem than just making sure outputs matched for similar inputs, while at the same time making further testing
intolerably slower. Resolving that problem is the next big step for this effort.

## It's Actually `compute_weights`
The trouble doesn't necessarily seem to be the call to a fitting function. Instead, it's my `compute_weights` call in all successive em-iterations. A comparison of runtimes between MATLAB and Python versions observes a nearly 10x speed difference in runtime, something not really scalable to 20 em-iterations as prescribed. 

Even once I fix this, it seems likely that I'll also have the memory problem that I observed earlier. But this is the chief bottleneck for me now.

```python
    for idx in range(np.shape(reduced_nParticles_idx)[1]):
        prob_grp_lvl_curve = np.zeros((nParticles, reduced_nParticles))
        target_indices = np.arange(reduced_nParticles_idx[0, idx], reduced_nParticles_idx[1, idx])
        for npm in range(nParam):
            which_param = npm
            nth_grp_lvl_param = np.tile(param[:, npm].reshape(-1, 1), (1, reduced_nParticles))
            nth_prev_iter_curve_param = prev_iter_curve_param[target_indices, npm]
            trunc_likes = np.array([compute_trunc_likes(nth_grp_lvl_param[:, i], nth_prev_iter_curve_param[i])
                                    for i in range(len(nth_prev_iter_curve_param))]).T
            prob_grp_lvl_curve = np.add(prob_grp_lvl_curve, trunc_likes)

            if np.any(np.isnan(prob_grp_lvl_curve)):
                raise ValueError('NaNs in probability of group level curves matrix!')

        q_theta = np.add(q_theta, np.exp(prob_grp_lvl_curve) * normalized_w[target_indices])
```

VS

```m
for idx = 1:size(reduced_nParticles_idx, 2)
	prob_grp_lvl_curve = zeros(nParticles, reduced_nParticles);
	target_indices = reduced_nParticles_idx(1, idx):reduced_nParticles_idx(2, idx);
	for npm = 1:nParam
		which_param = npm;
		nth_grp_lvl_param = repmat(param(:, npm), 1, reduced_nParticles);
		nth_prev_iter_curve_param = prev_iter_curve_param(target_indices, npm)';
		prob_grp_lvl_curve = bsxfun(@plus, prob_grp_lvl_curve, bsxfun(@compute_trunc_likes, nth_grp_lvl_param, nth_prev_iter_curve_param));
	end
	if any(isnan(prob_grp_lvl_curve(:))), error('NaNs in probability of group level curves matrix!'); end
	q_theta = bsxfun(@plus, q_theta, (exp(prob_grp_lvl_curve) * normalized_w(target_indices)'));
	clear prob_grp_lvl_curve; clear target_indices; clear nth_grp_lvl_param; clear nth_prev_iter_curve_param;
end
```

The suggestion has been raised that I might not be doing the numpy broadcasting right? That seems dubious, but it suggests I could bear to try rewriting my calls to `compute_weights` and `compute_trunc_likes` super carefully. We'll see.

![](2020-12-16-21-43-37.png)

```python
import time
start = time.time()
for iteration in range(10):
    test = np.apply_along_axis(compute_trunc_likes, 1, nth_grp_lvl_param, mu=).T
end = time.time()
print(end-start)
```