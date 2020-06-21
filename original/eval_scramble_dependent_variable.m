function[scrambled_vector] = eval_scramble_dependent_variable(target_dependent_variables, target_net_effect_clusters)

% [SCRAMBLED_VECTOR] = SCRAMBLE_DEPENDENT_VARIABLE(TARGET_DEPENDENT_VARIABLES, TARGET_NET_EFFECT_CLUSTERS)
%
% Purpose
%
% To test code taking a dependent variable vector and scrambling it such that the net effect cluster groupings are NOT violated. 
%
% Input
%
% --target_dependent_variables: The vector you would like scrambled
% --target_net_effect_clusters: The groupings that you would like to NOT violate. Follow the example below
%
% Output
%
% --scrambled_vector: A scrambled vector
%
% Example usage:
%
% scramble_dependent_variable(target_dependent_variables, target_net_effect_clusters)
%

if ~isequal(size(target_dependent_variables), size(target_net_effect_clusters))
	error('Size of input vectors must be the same!');
end

% Detailed example
% example data matrix: target_dependent_variables = [1, 0, 1, 0, 0, 0, 1] and target_net_effect_clusters = [3, 5, 3, 7, 7, 5, 8]

% Fetch the sorted list of net effect clusters and their respective locations
% e.g. for [3, 5, 3, 7, 7, 5, 8] will return [3, 3, 5, 5, 7, 7, 8] and [1, 3, 2, 6, 4, 5, 7]
[sorted_neteff_clusters, sorted_neteff_clusters_indices] = sort(target_net_effect_clusters);
just_ones = ones(size(sorted_neteff_clusters)); % Populate a vector full of ones
% compute the length of each net effect cluster
% e.g. for [3, 5, 3, 7, 7, 5, 8] will return [2, 2, 2, 1] i.e. 3 is repeated twice and so on
length_each_neteff_cluster = accumarray(sorted_neteff_clusters(:), just_ones(:), [], @sum)';
length_each_neteff_cluster = length_each_neteff_cluster(logical(length_each_neteff_cluster));

% Get the unique list of clusters (i.e. excluding repetitions if any) e.g. [3, 5, 7, 8]
[unique_neteff_clusters, unique_indices] = unique(target_net_effect_clusters);
% Get the associated dependent variables (one per cluster; recall it is unique within a cluster) e.g. [1, 0, 0, 1]
associated_dependent_variables = target_dependent_variables(unique_indices);
% scramble the dependent variables e.g. [0, 0, 1, 1]
scrambled_indices = randperm(length(associated_dependent_variables));
% scrambled_dependent_variables = associated_dependent_variables(scrambled_indices);
scrambled_dependent_variables = [0, 0, 1, 1]

% Now we will need to repeat each scrambled dependent variable for the length of that net effect cluster. The next three lines will result in
% [0, 0, 0, 0, 1, 1, 1] corresponding to [3, 3, 5, 5, 7, 7, 8] since the scrambled dependent variable looks like [0, 0, 1, 1] for [3, 5, 7, 8]
cumsum_clutsers = cumsum(length_each_neteff_cluster);
indicator_vector = zeros(1, cumsum_clutsers(end));
indicator_vector([1, cumsum_clutsers(1:end-1)+1]) = 1;

% Store the scrambled dependent variable in the respective cluster locations
% The original vector looked like [3, 5, 3, 7, 7, 5, 8] so the scrambled vector will look like [0, 0, 0, 1, 1, 0, 1]
scrambled_vector = NaN(size(sorted_neteff_clusters_indices));
scrambled_vector(sorted_neteff_clusters_indices) = scrambled_dependent_variables(cumsum(indicator_vector));

if any(isnan(scrambled_vector(:))), error('Nan''s in scrambled dependent variable vector!'); end
