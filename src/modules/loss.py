import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def check_gospa_parameters(c, p, alpha):
    """ Check parameter bounds.

    If the parameter values are outside the allowable range specified in the
    definition of GOSPA, a ValueError is raised.
    """
    if alpha <= 0 or alpha > 2:
        raise ValueError("The value of alpha is outside the range (0, 2]")
    if c <= 0:
        raise ValueError("The cutoff distance c is outside the range (0, inf)")
    if p < 1:
        raise ValueError("The order p is outside the range [1, inf)")


class MotLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        if params.loss.type == 'gospa':
            check_gospa_parameters(params.loss.cutoff_distance, params.loss.order, params.loss.alpha)
            self.order = params.loss.order
            self.cutoff_distance = params.loss.cutoff_distance
            self.alpha = params.loss.alpha
            self.miss_cost = self.cutoff_distance ** self.order
        self.params = params
        self.device = torch.device(params.training.device)
        self.to(self.device)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def compute_hungarian_matching(self, predicted_states, predicted_logits, targets, distance='detr', scaling=1):
        """ Performs the matching

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = predicted_states.shape[:2]
        predicted_probabilities = predicted_logits.sigmoid()

        indices = []
        for i in range(bs):
            # Compute cost matrix for this batch position
            cost = torch.cdist(predicted_states[i], targets[i], p=2)
            cost -= predicted_probabilities[i].log()

            # Compute minimum cost assignment and save it
            with torch.no_grad():
                indices.append(linear_sum_assignment(cost.cpu()))

        permutation_idx = [(torch.as_tensor(i, dtype=torch.int64).to(self.device),
                            torch.as_tensor(j, dtype=torch.int64).to(self.device)) for i, j in indices]

        return permutation_idx, cost.to(self.device)

    def compute_orig_gospa_matching(self, outputs, targets, existence_threshold):
        """ Performs the matching. Note that this can NOT be used as a loss function

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

            existence_threshold: Float in range (0,1) that decides which object are considered alive and which are not.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"
        assert self.order == 1, 'This code does not work for loss.order != 1'
        assert self.alpha == 2, 'The permutation -> assignment relation used to decompose GOSPA might require that loss.alpha == 2'

        output_state = outputs['state'].detach()
        output_existence_probabilities = outputs['logits'].sigmoid().detach()

        bs, num_queries = output_state.shape[:2]
        dim_predictions = output_state.shape[2]
        dim_targets = targets[0].shape[1]
        assert dim_predictions == dim_targets

        loss = torch.zeros(size=(1,))
        localization_cost = 0
        missed_target_cost = 0
        false_target_cost = 0
        indices = []

        for i in range(bs):
            alive_idx = output_existence_probabilities[i, :].squeeze(-1) > existence_threshold
            alive_output = output_state[i, alive_idx, :]
            current_targets = targets[i]
            permutation_length = 0

            if len(current_targets) == 0:
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(alive_output)])
                false_target_cost = self.miss_cost/self.alpha * len(alive_output)
            elif len(alive_output) == 0:
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(current_targets)])
                missed_target_cost = self.miss_cost / self.alpha * len(current_targets)
            else:
                dist = torch.cdist(alive_output, current_targets, p=2)
                dist = dist.clamp_max(self.cutoff_distance)
                c = torch.pow(input=dist, exponent=self.order)
                c = c.cpu()
                output_idx, target_idx = linear_sum_assignment(c)
                indices.append((output_idx, target_idx))

                for t, o in zip(output_idx, target_idx):
                    loss += c[t,o]
                    if c[t, o] < self.cutoff_distance:
                        localization_cost += c[t, o].item()
                        permutation_length += 1
                
                cardinality_error = abs(len(alive_output) - len(current_targets))
                loss += self.miss_cost/self.alpha * cardinality_error

                missed_target_cost += (len(current_targets) - permutation_length) * (self.miss_cost/self.alpha)
                false_target_cost += (len(alive_output) - permutation_length) * (self.miss_cost/self.alpha)

        decomposition = {'localization': localization_cost, 'missed': missed_target_cost, 'false': false_target_cost,
                         'n_matched_objs': permutation_length}
        return loss, indices, decomposition

    def compute_orig_gospa_matching_with_uncertainties(self, predictions, targets, existence_threshold):

        assert self.order == 1, 'This code does not work for loss.order != 1'
        assert self.alpha == 2, 'The permutation -> assignment relation used to decompose GOSPA might require that loss.alpha == 2'

        batch_size, _, dim_predictions = predictions['state'].shape
        n_targets, dim_targets = targets[0].shape
        assert dim_predictions == dim_targets
        assert batch_size == 1, 'GOSPA matching with uncertainties currently only works with batch size = 1'

        existence_probabilities = predictions['logits'][0].sigmoid().detach()
        alive_idx = existence_probabilities.squeeze(-1) > existence_threshold

        predicted_distributions = {'states': predictions['state'][0, alive_idx].detach(),
                                   'state_covariances': predictions['state_covariances'][0, alive_idx].detach()}
        targets = targets[0]
        n_predictions = len(predicted_distributions['states'])

        loss = torch.zeros(size=(1,))
        localization_cost = 0
        missed_target_cost = 0
        false_target_cost = 0
        indices = []
        permutation_length = 0

        if n_targets == 0:
            indices.append(([], []))
            loss += torch.Tensor([self.miss_cost / self.alpha * n_predictions])
            false_target_cost = self.miss_cost / self.alpha * n_predictions
        elif n_predictions == 0:
            indices.append(([], []))
            loss += torch.Tensor([self.miss_cost / self.alpha * n_targets])
            missed_target_cost = self.miss_cost / self.alpha * n_targets
        else:
            dist = compute_pairwise_crossentropy(predicted_distributions, targets)
            dist = dist.clamp_max(self.cutoff_distance)
            c = torch.pow(input=dist, exponent=self.order)
            c = c.cpu()
            target_idx, output_idx = linear_sum_assignment(c)
            indices.append((target_idx, output_idx))

            for t, o in zip(target_idx, output_idx):
                loss += c[t, o]
                if c[t, o] < self.cutoff_distance:
                    localization_cost += c[t, o].item()
                    permutation_length += 1

            cardinality_error = abs(n_predictions - n_targets)
            loss += self.miss_cost / self.alpha * cardinality_error

            missed_target_cost += (n_targets - permutation_length) * (self.miss_cost / self.alpha)
            false_target_cost += (n_predictions - permutation_length) * (self.miss_cost / self.alpha)

        decomposition = {'localization': localization_cost, 'missed': missed_target_cost, 'false': false_target_cost,
                         'n_matched_objs': permutation_length}
        return loss, indices, decomposition

    def gospa_forward(self, outputs, targets, probabilistic=True, existence_threshold=0.75):

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"

        output_state = outputs['state']
        output_logits = outputs['logits'].sigmoid()
        # List with num_objects for each training-example
        sizes = [len(v) for v in targets]

        bs = output_state.shape[0]
        if probabilistic:
            indices, cost_matrix, unmatched_x = self.compute_prob_gospa_matching(outputs, targets)
            cost_matrix = cost_matrix.split(sizes, -1)
            loss = 0
            for i in range(bs):
                batch_idx = indices[i]
                batch_cost = cost_matrix[i][i][batch_idx].sum()
                batch_cost = batch_cost + output_logits[i][unmatched_x[i]].sum() * self.miss_cost/2.0
                loss = loss + batch_cost
            loss = loss/sum(sizes)
            return loss, indices
        else:
            assert 0 <= existence_threshold < 1, "'existence_threshold' should be in range (0,1)"
            loss, indices, decomposition = self.compute_orig_gospa_matching(outputs, targets, existence_threshold)
            loss = loss / bs
            return loss, indices, decomposition

    def state_loss(self, predicted_states, targets, indices, uncertainties=None):
        idx = self._get_src_permutation_idx(indices)
        matched_predicted_states = predicted_states[idx]
        target = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        if uncertainties is not None:
            matched_uncertainties = uncertainties[idx]
            prediction_distribution = torch.distributions.normal.Normal(matched_predicted_states, matched_uncertainties)
            loss = -prediction_distribution.log_prob(target).mean()
        else:
            loss = F.l1_loss(matched_predicted_states, target, reduction='none').sum(-1).mean()

        return loss

    def logits_loss(self, predicted_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.zeros_like(predicted_logits, device=predicted_logits.device)
        target_classes[idx] = 1.0  # this is representation of an object
        loss = F.binary_cross_entropy_with_logits(predicted_logits.squeeze(-1).permute(1,0), target_classes.squeeze(-1).permute(1,0))

        return loss

    def get_loss(self, prediction, targets, loss_type, existence_threshold=None):
        # Create state vectors for the predictions, based on prediction target specified by user
        if self.params.data_generation.prediction_target == 'position':
            predicted_states = prediction.positions
        elif self.params.data_generation.prediction_target == 'position_and_velocity':
            predicted_states = torch.cat((prediction.positions, prediction.velocities), dim=2)
        else:
            raise NotImplementedError(f'Hungarian matching not implemented for prediction target '
                                      f'{self.params.data_generation.prediction_target}')

        if loss_type == 'gospa':
            loss, indices = self.gospa_forward(prediction, targets, probabilistic=True)
            loss = {f'{loss_type}_state': loss, f'{loss_type}_logits': 0}
        elif loss_type == 'gospa_eval':
            loss,_ = self.gospa_forward(prediction, targets, probabilistic=False, existence_threshold=existence_threshold)
            indices = None
            loss = {f'{loss_type}_state': loss, f'{loss_type}_logits': 0}
        elif loss_type == 'detr':
            indices, _ = self.compute_hungarian_matching(predicted_states, prediction.logits, targets)
            log_loss = self.logits_loss(prediction.logits, targets, indices)
            if hasattr(prediction, 'uncertainties'):
                state_loss = self.state_loss(predicted_states, targets, indices, uncertainties=prediction.uncertainties)
            else:
                state_loss = self.state_loss(predicted_states, targets, indices)
            loss = {f'{loss_type}_state': state_loss, f'{loss_type}_logits': log_loss}
        
        return loss, indices
    
    def forward(self, targets, prediction, intermediate_predictions=None, encoder_prediction=None, loss_type='detr',
                existence_threshold=0.75):
        if loss_type not in ['gospa', 'gospa_eval', 'detr']:
            raise NotImplementedError(f"The loss type '{loss_type}' was not implemented.'")

        losses = {}
        loss, indices = self.get_loss(prediction, targets, loss_type, existence_threshold)
        losses.update(loss)

        if intermediate_predictions is not None:
            for i, intermediate_prediction in enumerate(intermediate_predictions):
                aux_loss, _ = self.get_loss(intermediate_prediction, targets, loss_type, existence_threshold)
                aux_loss = {f'{k}_{i}': v for k, v in aux_loss.items()}
                losses.update(aux_loss)

        if encoder_prediction is not None:
            enc_loss, _ = self.get_loss(encoder_prediction, targets, loss_type, existence_threshold)
            enc_loss = {f'{k}_enc': v for k, v in enc_loss.items()}
            losses.update(enc_loss)

        return losses, indices
