from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.base import EEGWindowsDataset
from sklearn.metrics import balanced_accuracy_score
from importlib import import_module
import random
from numbers import Integral
import numpy as np
import argparse
import json
from contextlib import nullcontext

import torch
from tqdm import tqdm
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

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
    parser.add_argument('--random_seed', default=20200220, type=int)

    parser.add_argument('--lr_warmup', default=False, type=bool, help='whether to warm up learning rate during training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--fine_tune_lr', default=1e-3, type=float, help='fine tune learning rate')

    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--pretrain_n_epochs', default=50, type=int)
    parser.add_argument('--adaptation_n_epochs', default=50, type=int)
    parser.add_argument('--fine_tune_n_epochs', default=30, type=int)

    parser.add_argument('--significance_level', default=0.95, type=float)

    parser.add_argument('--weight_decay', default=0, type=int)
    parser.add_argument('--fine_tune_weight_decay', default=0, type=int)
    
    parser.add_argument('--fine_tune_freeze_layer', default=None, type=list)
    parser.add_argument('--freeze_most_layers', default=False, type=bool)
    parser.add_argument('--fine_tune_freeze_layers_but', default=None, type=list)

    parser.add_argument('--forward_pass_kwargs', default=None)

    parser.add_argument('--optimize_for_acc', default=True, type=bool)
    parser.add_argument('--regularize_tensor_distance', default=True, type=bool)
    parser.add_argument('--regularization_coef', default=1, type=float)

    # For MAPU
    parser.add_argument('--scenarios', default=None, type=list)
    parser.add_argument('--add_tov_loss', default=True, type=bool)
    parser.add_argument('--pretrain_lr', default=1e-3, type=float, help='pretraining learning rate')
    parser.add_argument('--imputer_lr', default=1e-5, type=float, help='imputer learning rate')
    parser.add_argument('--adaptation_lr', default=1e-5, type=float, help='adaptation learning rate')
    parser.add_argument('--adaptation_lr_decay', default=1e-5, type=float)
    parser.add_argument('--adaptation_lr_step_size', default=1e-5, type=float)
    parser.add_argument('--ent_loss_wt', default=0.4216, type=float)
    parser.add_argument('--im', default=0.5514, type=float)
    parser.add_argument('--TOV_wt', default=0.6385, type=float)

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


def freeze_all_param_but(module_obj, parameter_name_lst):
    """
    Freeze all parameters of a model but the specified ones
    """
    for name, param in module_obj.named_parameters():
        if name in parameter_name_lst:
            continue
        param.requires_grad = False


'''
Define a method for training one epoch. Adapted from
https://braindecode.org/stable/auto_examples/model_building/plot_train_in_pure_pytorch_and_pytorch_lightning.html
'''
def train_one_epoch(
    dataloader: DataLoader, 
    model: Module, 
    loss_fn, 
    optimizer,
    scheduler: LRScheduler, 
    warmup_scheduler: None,
    epoch: int, 
    device="cuda", 
    print_batch_stats=False,
    optimize_for_acc=True,
    regularize_tensor_distance=False,
    regularization_coef=1,
    **forward_pass_kwargs
):
    # Have to include at least one loss term
    assert optimize_for_acc or regularization_coef, "Must include at least one loss term"

    # Set the model to training mode
    model.train()  
    train_loss, correct = 0, 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                        disable=not print_batch_stats)

    for batch_idx, (X, y, _) in progress_bar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X, **forward_pass_kwargs)
        
        if optimize_for_acc:
            acc_loss = loss_fn(pred, y)
        else:
            acc_loss = 0
        if regularize_tensor_distance:
            distance = model.calculate_tensor_distance()
            distance_loss = regularization_coef * distance
        else:
            distance_loss = 0
        loss = acc_loss + distance_loss

        loss.backward()
        # update the model weights
        optimizer.step()  
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Epoch {epoch}, "
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.6f}"
            )

    # Update the learning rate
    with warmup_scheduler.dampening() if warmup_scheduler is not None else nullcontext():
        scheduler.step()


    correct /= len(dataloader.dataset)
    return train_loss / len(dataloader), correct


'''
Evaluate model with no backprop. Adapted from
https://braindecode.org/stable/auto_examples/model_building/plot_train_in_pure_pytorch_and_pytorch_lightning.html
'''
@torch.no_grad()
def test_model(
    dataloader: DataLoader, 
    model: Module, 
    loss_fn, 
    print_batch_stats=True, 
    optimize_for_acc=True,
    regularize_tensor_distance=False,
    regularization_coef=1,
    device="cuda", 
    **forward_pass_kwargs
):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    # Switch to evaluation mode
    model.eval()  
    test_loss, correct = 0, 0

    if print_batch_stats:
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        progress_bar = enumerate(dataloader)

    for batch_idx, (X, y, _) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X, **forward_pass_kwargs)

        if optimize_for_acc:
            acc_loss = loss_fn(pred, y).item()
        else:
            acc_loss = 0
        if regularize_tensor_distance:
            distance = model.calculate_tensor_distance()
            distance_loss = regularization_coef * distance
        else:
            distance_loss = 0
        batch_loss = acc_loss + distance_loss

        # batch_loss = loss_fn(pred, y).item()
        # if regularize_tensor_distance:
        #     distance = model.calculate_tensor_distance()
        #     distance_loss = regularization_coef * distance
        #     # print(f'Tensor distance loss to reference tensor is {distance_loss:.6f}')
        #     batch_loss += distance_loss

        test_loss += batch_loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if print_batch_stats:
            progress_bar.set_description(
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {batch_loss:.6f}"
            )

    test_loss /= n_batches
    correct /= size

    print(
        f"Test Accuracy: {100 * correct:.1f}%, Test Loss: {test_loss:.6f}\n"
    )
    return test_loss, correct

    
