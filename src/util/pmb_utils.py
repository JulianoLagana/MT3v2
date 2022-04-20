import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.optimize import linear_sum_assignment
from util.misc import AnnotatedValue, AnnotatedValueSum


def compute_nll_for_pmb(predictions, targets, target_infos=None):
    if target_infos is None:
        dummy_ids = [0 for _ in range(len(targets))]
        dummy_trajectories = [[[-1, -1, -1, -1, -1]]]
        target_infos = [dummy_ids, dummy_trajectories]
    target_ids, all_trajectories = target_infos

    n_predictions = len(predictions['means'])
    n_targets = len(targets)

    if len(predictions['covs'].shape) == 3:
        distribution_type = MultivariateNormal
        scale_params = predictions['covs']
    else:
        distribution_type = Normal
        scale_params = predictions['covs'].sqrt()

    cost_matrix = np.ones((n_predictions + n_targets, n_targets)) * np.inf
    for i_prediction in range(n_predictions):
        p_existence = predictions['p_exs'][i_prediction].item()
        dist = distribution_type(predictions['means'][i_prediction],scale_params[i_prediction])
        for i_target in range(n_targets):
            cost_matrix[i_prediction, i_target] = -(np.log(p_existence) + dist.log_prob(targets[i_target]).sum() - np.log(1-p_existence))

    # Fill in diagonal of sub-matrix corresponding to PPP matches
    for i_target in range(n_targets):
        cost_matrix[n_predictions + i_target, i_target] = -predictions['ppp_log_prob_func'](targets[i_target])

    # Find optimal match using Hungarian algorithm
    optimal_match = linear_sum_assignment(cost_matrix)

    # Compute likelihood and decompositions
    annotated_cost = AnnotatedValueSum()
    annotated_cost.add(AnnotatedValue(predictions['ppp_lambda'], {'type': 'miss'}))
    for i_prediction, i_target in zip(optimal_match[0], optimal_match[1]):
        birth_time_annotation = all_trajectories[target_ids[i_target]][0][4]
        # For targets matched with predictions, add cost for localization and existence probability
        if i_prediction < n_predictions:
            p_existence = predictions['p_exs'][i_prediction].item()
            temp = -np.log(p_existence) + np.log(1-p_existence)
            annotated_cost.add(AnnotatedValue(cost_matrix[i_prediction, i_target] - temp,
                                              {'type': 'loc',
                                               'target_state': targets[i_target],
                                               'target_birth_time': birth_time_annotation}))
            annotated_cost.add(AnnotatedValue(-np.log(p_existence),
                                              {'type': 'p_true',
                                               'target_state': targets[i_target],
                                               'target_birth_time': birth_time_annotation}))

        # For targets matched with PPP, just add cost for explaining missed targets
        else:
            annotated_cost.add(AnnotatedValue(cost_matrix[i_prediction, i_target],
                                              {'type': 'miss',
                                               'target_state': targets[i_target],
                                               'target_birth_time': birth_time_annotation}))
    # Afterwards, add -log(1-p) for all predictions false predictions.
    for i_prediction in range(n_predictions):
        if i_prediction not in optimal_match[0]:
            p_existence = predictions['p_exs'][i_prediction].item()
            annotated_cost.add(AnnotatedValue(-np.log(1-p_existence),
                                              {'type': 'p_false', 'pred_state': predictions['means'][i_prediction]}))

    negative_log_likelihood = annotated_cost.get_total_value()
    loc_cost = sum(annotated_cost.get_filtered_values(lambda x: x.get('type')=='loc'))
    p_true_cost = sum(annotated_cost.get_filtered_values(lambda x: x.get('type')=='p_true'))
    p_false_cost = sum(annotated_cost.get_filtered_values(lambda x: x.get('type')=='p_false'))
    p_miss_cost = sum(annotated_cost.get_filtered_values(lambda x: x.get('type')=='miss'))

    return negative_log_likelihood, (loc_cost, p_true_cost, p_false_cost, p_miss_cost), optimal_match, annotated_cost


class UnnormalizedGaussianMixture:
    def __init__(self, weights, means, covs):
        self.weights = weights
        self.components = MultivariateNormal(means, covs)

    def log_prob(self, x):
        log_probs_for_each_component = self.components.log_prob(x)
        return torch.logsumexp(log_probs_for_each_component + self.weights.log(), 0).item()

    def get_lambda(self):
        return self.weights.sum().item()
