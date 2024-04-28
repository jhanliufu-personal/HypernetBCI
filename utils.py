from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.base import EEGWindowsDataset
from importlib import import_module
import random
from numbers import Integral
import numpy as np


def generate_non_repeating_integers(x, y):
    # Check if y is greater than x
    if y < x:
        raise ValueError(f"y must be greater than or equal to x; Got y={y}, x={x}")
    
    # Generate x non-repeating integers between 0 and y
    return random.sample(range(y), x)


def sample_integers_sum_to_x(x, k):
    '''
    k >= 2
    '''
    # Generate k-1 random integers between 1 and x
    parts = sorted(random.randint(1, x) for _ in range(k-1))
    
    # Calculate the differences between consecutive numbers
    differences = [parts[0]] + [parts[i] - parts[i-1] for i in range(1, k-1)] + [x - parts[-1]]
    
    return differences


def get_subset(input_set, target_trial_num, random_sample=False, from_back=False):
    # check inputs
    assert isinstance(input_set, BaseConcatDataset)
    assert isinstance(target_trial_num, int)
    
    new_ds_lst = []

    if random_sample:
        
        base_ds_cnt = len(input_set.datasets)

        if base_ds_cnt == 1:
            cur_ds = input_set.datasets[0]
            trial_idx = generate_non_repeating_integers(target_trial_num, len(cur_ds))
            new_ds_lst.append(EEGWindowsDataset(cur_ds.raw, cur_ds.metadata.iloc[trial_idx], 
                                                        description=cur_ds.description))

        elif base_ds_cnt > 1:
            trial_cnt_from_each_base_ds = sample_integers_sum_to_x(target_trial_num, base_ds_cnt)
            for i, cnt in enumerate(trial_cnt_from_each_base_ds):
                if not cnt:
                    # no sampling in current base dataset
                    continue
            
                # Access current base dataset
                cur_ds = input_set.datasets[i]
                assert isinstance(cur_ds, EEGWindowsDataset)
                # Randomly sample trial index
                try:
                    trial_idx = generate_non_repeating_integers(cnt, len(cur_ds))
                    new_ds_lst.append(EEGWindowsDataset(cur_ds.raw, cur_ds.metadata.iloc[trial_idx], 
                                                        description=cur_ds.description))
                except ValueError:
                    try:
                        # If trying to sample more trials in current ds than there are
                        # Get entire cur_ds, and get what's missing fromt the next ds
                        new_ds_lst.append(cur_ds)
                        trial_cnt_from_each_base_ds[i+1] += (cnt - len(cur_ds))
                    except IndexError:
                        pass

    else:
    
        if from_back:
            for ds in reversed(input_set.datasets):
                assert isinstance(ds, EEGWindowsDataset)
                cur_run_trial_num = len(ds.metadata)
                if target_trial_num > cur_run_trial_num:
                    new_ds_lst.append(ds)
                    target_trial_num -= cur_run_trial_num
                else:
                    new_ds_lst.append(EEGWindowsDataset(ds.raw, ds.metadata[-target_trial_num:], description=ds.description))
                    break

        else:
            for ds in input_set.datasets:
                assert isinstance(ds, EEGWindowsDataset)
                cur_run_trial_num = len(ds.metadata)
                if target_trial_num > cur_run_trial_num:
                    new_ds_lst.append(ds)
                    target_trial_num -= cur_run_trial_num
                else:
                    new_ds_lst.append(EEGWindowsDataset(ds.raw, ds.metadata[:target_trial_num], description=ds.description))
                    break

    return BaseConcatDataset(new_ds_lst)


def import_model(model_name: str) -> object:
    # try import from braindecode models first
    try:
        braindecode_model_module_path = "braindecode.models"
        braindecode_model_module = import_module(braindecode_model_module_path)
        model_object = getattr(braindecode_model_module, model_name)
        return model_object

    except AttributeError:
        # if braindecode doesn't have it, check locally defined models
        pass


def get_center_label(x):
    # Use label of center window in the sequence as sequence target
    if isinstance(x, Integral):
        return x
    return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x