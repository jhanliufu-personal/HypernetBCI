from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.base import EEGWindowsDataset
from sklearn.metrics import balanced_accuracy_score
from importlib import import_module
import random
from numbers import Integral
import numpy as np
import argparse
import json


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


def balanced_accuracy_multi(model, X, y):
    y_pred = model.predict(X)
    return balanced_accuracy_score(y.flatten(), y_pred.flatten())


def clf_predict_on_set(clf, dataset):
    predicted_labels = clf.predict(dataset)
    true_labels = np.array(dataset.get_metadata().target)
    predicted_correct = np.equal(predicted_labels, true_labels)
    prediction_acc = np.sum(predicted_correct) / len(predicted_correct)
    return prediction_acc


def parse_training_config():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='HypernetBCI Training and Testing')
    parser.add_argument('--json', default=None, type=str, help='Path to JSON configuration file')
    parser.add_argument('--gpu_number', default='0', choices=range(4), type=str, help='BBQ cluster has 4 gpus')
    parser.add_argument('--experiment_version', default='1_1', type=str, help='1 -> train from scratch, 2 -> fine tune')
    parser.add_argument('--model_name', default='SleepStagerChambon2018', type=str)
    parser.add_argument('--model_kwargs', default=None)
    parser.add_argument('--dataset_name', default='SleepPhysionet', type=str)
    parser.add_argument('--data_amount_start', default=0, type=int)
    parser.add_argument('--data_amount_step', default=20, type=int, 
                        help='Increment training set size by this much each time. \
                            For testing purpose use super big data_amount_step')
    parser.add_argument('--trial_len_sec', default=30, type=float)
    parser.add_argument('--data_amount_unit', default='min', type=str)
    parser.add_argument('--only_pretrain', default=False, type=bool, help='If only_pretrain, the finetune step is skipped')
    parser.add_argument('--repetition', default=5, type=int, 
                        help="Repeat for this many times for each training_data_amount")
    parser.add_argument('--n_classes', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--significance_level', default=0.95, type=float)
    parser.add_argument('--fine_tune_lr', default=1e-3, type=float, help='fine tune learning rate')
    parser.add_argument('--weight_decay', default=0, type=int)
    parser.add_argument('--finetune_weight_decay', default=0, type=int)
    parser.add_argument('--fine_tune_n_epochs', default=30, type=int)
    parser.add_argument('--fine_tune_free_layer', default=None, type=list)

    args = parser.parse_args()
    with open(args.json, 'r') as f:
        args_from_json = json.load(f)

    # Use the dictionary to set the arguments
    for key, value in args_from_json.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # print(args)

    return args


def freeze_param(module_obj, parameter_name):
    """
    Freeze the specified layer of a model
    """
    param = module_obj
    for t in parameter_name.split('.'):
        param = getattr(param, t)
    param.requires_grad = False
    
