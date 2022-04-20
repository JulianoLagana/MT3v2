from typing import Optional, List
import math
import os
import sys

import torch
from torch import Tensor

from util.load_config_files import load_yaml_into_dotdict, dotdict


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(meas.shape) for meas in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for meas, pad_meas, m in zip(tensor_list, tensor, mask):
            pad_meas[: meas.shape[0], : meas.shape[1],
                     : meas.shape[2]].copy_(meas)
            m[: meas.shape[1], :meas.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def save_checkpoint(folder, filename, model, optimizer, scheduler):
    print(f"[INFO] Saving checkpoint in {folder}/{filename}")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(folder, filename))

def update_logs(logs, key, value):
    if not key in logs:
        logs[key] = [value]
    else:
        logs[key].append(value)
    return logs

def factor_int(n):
    """
    Given an integer n, compute a factorization into two integers such that they're close as possible to each other (like
    a square root). E.g. factor_int(16)=(4,4), but factor_int(15)=(3,5).
    """
    nsqrt = math.ceil(math.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n/val)
        if val2 * val == float(n):
            solution = True
        else:
            val-=1
    return val, val2


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


@torch.no_grad()
def split_batch(batch, unique_ids, params):
    bs = batch.tensors.shape[0]
    batch = batch.tensors
    first_batch = []
    first_ids = []
    second_batch = []
    second_ids = []

    mapped_time_idx = batch[:,:,-1] / params.data_generation.dt
    
    for i in range(bs):
        # Take out all measurements that are in the first batch that are not padded
        first_batch_idx = mapped_time_idx[i] < params.data_generation.n_timesteps
        first_batch_idx = torch.logical_and(first_batch_idx, unique_ids[i]!=-2)

        second_batch_idx = 1 <= mapped_time_idx[i]
        second_batch_idx = torch.logical_and(second_batch_idx, unique_ids[i]!=-2)

        first_batch.append(batch[i][first_batch_idx])
        first_ids.append(unique_ids[i][first_batch_idx])

        second_batch.append(batch[i][second_batch_idx])
        second_ids.append(unique_ids[i][second_batch_idx])

        # Shift timestep
        second_batch[i][:,-1] = second_batch[i][:,-1] - params.data_generation.dt
    
    
    first, first_ids = pad_and_nest(first_batch, first_ids)
    second, second_ids = pad_and_nest(second_batch, second_ids)
                   
    return first, second, first_ids, second_ids


def pad_and_nest(batch, ids):
    max_len = max(list(map(len, batch)))
    batch, mask, ids = pad_to_batch_max(batch, ids, max_len)
    nested = NestedTensor(batch, mask.bool())

    return nested, ids


def pad_to_batch_max(batch, unique_ids, max_len):
    batch_size = len(batch)
    dev = batch[0].device
    d_meas = batch[0].shape[1]
    training_data_padded = torch.zeros((batch_size, max_len, d_meas), device=dev)
    mask = torch.ones((batch_size, max_len), device=dev)
    ids = -2 * torch.ones((batch_size, max_len), device=dev)
    for i, ex in enumerate(batch):
        training_data_padded[i,:len(ex),:] = ex
        mask[i,:len(ex)] = 0
        ids[i,:len(ex)] = unique_ids[i]

    return training_data_padded, mask, ids


def extract_batch(batch,unique_ids, lower_time_idx, upper_time_idx, dt,  batch_id=0):
    bt = batch.tensors.clone().detach()
    bm = batch.mask.clone().detach()
    u = unique_ids.clone().detach()
    b = NestedTensor(bt,bm)
    times = torch.round(b.tensors[batch_id,:,-1] / dt)
    idx = torch.logical_and(lower_time_idx <= times, times < upper_time_idx)
    b.tensors = b.tensors[batch_id, idx].unsqueeze(0)
    b.tensors[:,:,-1] = b.tensors[:,:,-1] - lower_time_idx*dt
    b.mask = batch.mask[batch_id, idx].unsqueeze(0)
    u = u[:,idx]

    return b, u


def recursive_loss_sum(loss_dict):
    loss = 0
    if type(loss_dict) is not dict:
        return loss_dict
    else:
        for k,v in loss_dict.items():
            loss += recursive_loss_sum(v)
    return loss


def compute_median_absolute_deviation(x):
    median = x.median(dim=0)[0]
    mad = (x - median).norm(p=2, dim=1).median()
    return mad.item()


class Prediction:
    def __init__(self, positions=None, velocities=None, shapes=None, logits=None, uncertainties=None):
        if positions is not None:
            self.positions = positions
        if velocities is not None:
            self.velocities = velocities
        if shapes is not None:
            self.shapes = shapes
        if logits is not None:
            self.logits = logits
        if uncertainties is not None:
            self.uncertainties = uncertainties

        self._states = None

    @property
    def states(self):
        if self.positions is not None and self.velocities is not None:
            return torch.cat((self.positions, self.velocities), dim=2)
        elif self.positions is not None and self.velocities is None:
            return self.positions
        else:
            raise NotImplementedError(f'`states` attribute not implemented for positions {self.positions} and '
                                      f'velocities {self.velocities}.')


class AnnotatedValue:
    def __init__(self, value, annotation):
        self.value = value
        self.annotation = annotation


class AnnotatedValueSum:
    def __init__(self, *annotated_values):
        self.values = []
        self.annotations = []
        for annotated_value in annotated_values:
            self.values.append(annotated_value.value)
            self.annotations.append(annotated_value.annotation)

    def get_total_value(self):
        return sum(self.values)

    def get_filtered_values(self, filter_condition):
        values = []
        for value, annotation in zip(self.values, self.annotations):
            if filter_condition(annotation):
                values.append(value)
        return values

    def add(self, annotated_value: AnnotatedValue):
        self.values.append(annotated_value.value)
        self.annotations.append(annotated_value.annotation)

    def extend(self, other):
        self.values.extend(other.values)
        self.annotations.extend(other.annotations)



