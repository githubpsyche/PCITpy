 [raw_data, analysis_settings] = eval_run_importance_sampler();
 
 % Resetting the random number seed based on the clock
rand('twister', sum(100 * clock));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing the data matrix and updating the analysis_settings struct with additional/missing information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[preprocessed_data, ana_opt] = preprocessing_setup(raw_data, analysis_settings);
% Removing away the old information
clear raw_data; clear analysis_settings;

%%%%%%%%%%%%%%%
% House keeping
%%%%%%%%%%%%%%%
importance_sampler = struct(); % Creating the output struct
hold_betas_per_iter = NaN(ana_opt.em_iterations+1, 2); % Matrix to hold the betas over em iterations
exp_max_fval = NaN(ana_opt.em_iterations, 1); % Matrix to hold the fvals over em iterations
normalized_w = NaN(ana_opt.em_iterations+1, ana_opt.particles);  % Vector to hold the normalized weights
global tau
global bounds
global w
global net_effects
global dependent_var

%%%%%%%%%%%%%%%%%%
% Fetch parameters
%%%%%%%%%%%%%%%%%%
tau = ana_opt.tau; % Store the tau for convenience
bounds = family_of_curves(ana_opt.curve_type, 'get_bounds'); % Get the curve parameter absolute bounds
nParam = family_of_curves(ana_opt.curve_type, 'get_nParams'); % Get the number of curve parameters
hold_betas = [ana_opt.beta_0, ana_opt.beta_1]; % Store the betas into a vector

em = 1

hold_betas_per_iter(em, :) = hold_betas; % Store the logreg betas over em iterations
disp(sprintf('Betas: %0.4f, %0.4f', hold_betas(1), hold_betas(2)));
disp(sprintf('EM Iteration: %d', em));

% Initialize the previous iteration curve parameters, weight vector, net_effects and dependent_var matrices
prev_iter_curve_param = NaN(ana_opt.particles, family_of_curves(ana_opt.curve_type, 'get_nParams')); % Matrix to hold the previous iteration curve parameters
w = NaN(1, ana_opt.particles); % Vector to hold normalized weights
net_effects = NaN(length(ana_opt.net_effect_clusters), ana_opt.particles); % Matrix to hold the predictor variables (taking net effects if relevant) over all particles
dependent_var = []; % This vector cannot be initialized in advance since we don't know the length of this vector (dropping outliers)

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sampling curve parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%
if em == 1 % only for the first em iteration
    param = common_to_all_curves(ana_opt.curve_type, 'initial_sampling', ana_opt.particles, ana_opt.resolution); % Good old uniform sampling
else % for em iterations 2, 3, etc
    % Sample curve parameters from previous iteration's curve parameters based on normalized weights
    prev_iter_curve_param = param; % storing the previous iteration's curve parameters since we need them to compute likelihood
    % Here we sample curves (with repetitions) based on the weights
    param = prev_iter_curve_param(randsample([1:ana_opt.particles], ana_opt.particles, true, normalized_w(em-1, :)), :);

    % Add Gaussian noise since some curves are going to be identical due to the repetitions
    % NOISE: Sample from truncated normal distribution using individual curve parameter bounds, mean = sampled curve parameters and sigma = tau
    for npm = 1:nParam
        param(:, npm) = truncated_normal(bounds(npm, 1), bounds(npm, 2), param(:, npm), tau, ana_opt.particles);
    end
end
param = common_to_all_curves(ana_opt.curve_type, 'check_if_exceed_bounds', param); % Check whether curve parameters lie within the upper and lower bounds
if strcmp(ana_opt.curve_type, 'horz_indpnt')
    param = common_to_all_curves(ana_opt.curve_type, 'sort_horizontal_params', param); % Check if the horizontal curve parameters are following the right trend i.e. x1 < x2
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Compute the likelihood over all subjects (i.e. log probability mass function if logistic regression)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% This is where we use the chunking trick II
ptl_idx = 1;
output_struct = family_of_curves(ana_opt.curve_type, 'compute_likelihood', ana_opt.net_effect_clusters, ana_opt.ptl_chunk_idx(ptl_idx, 3),...
        param(ana_opt.ptl_chunk_idx(ptl_idx, 1):ana_opt.ptl_chunk_idx(ptl_idx, 2), :), hold_betas, preprocessed_data,...
        ana_opt.distribution, ana_opt.dist_specific_params, ana_opt.data_matrix_columns);

    