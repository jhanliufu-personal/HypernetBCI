import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl
from scipy import stats

import torch
from torch import nn

from braindecode import EEGClassifier
from braindecode.datasets import SleepPhysionet, BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_windows_from_events
from sklearn.preprocessing import scale as standard_scale
from braindecode.util import set_random_seeds
from braindecode.models import TimeDistributed
from braindecode.samplers import SequenceSampler

from sklearn.utils import compute_class_weight
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring

from utils import (
    get_subset, import_model, get_center_label, 
    balanced_accuracy_multi, parse_training_config, 
    freeze_all_param_but, clf_predict_on_set, freeze_param
)

import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
model_name = args.model_name
model_object = import_model(model_name)
dataset_name = args.dataset_name
# dataset = SleepPhysionet(subject_ids=range(79), recording_ids=[1, 2,], crop_wake_mins=30)
# Test with a few subjects first
dataset = SleepPhysionet(subject_ids=[0, 1, 2], recording_ids=[2,], crop_wake_mins=30)

print('Data loaded')

experiment_version = args.experiment_version
results_file_name = f'{model_name}_{dataset_name}_fine_tune_{experiment_version}'
# used to store pre-trained model parameters
temp_exp_name = f'baseline_2_{experiment_version}_pretrain'
dir_results = 'results/'
file_path = os.path.join(dir_results, f'{results_file_name}.pkl')

### ----------------------------- Training parameters -----------------------------
data_amount_start = args.data_amount_start
data_amount_step = args.data_amount_step
repetition = args.repetition
n_classes = args.n_classes
lr = args.lr
batch_size = args.batch_size
n_epochs = args.n_epochs

### ----------------------------- Plotting parameters -----------------------------
data_amount_unit = args.data_amount_unit
trial_len_sec = args.trial_len_sec
if data_amount_unit == 'trial':
    unit_multiplier = 1
elif data_amount_unit == 'sec':
    unit_multiplier = trial_len_sec
elif data_amount_unit == 'min':
    unit_multiplier = trial_len_sec / 60

significance_level = args.significance_level

### ----------------------------- Preprocessing -----------------------------
high_cut_hz = 30
factor = 1e6

preprocessors = [
    # Convert from V to uV
    Preprocessor(lambda data: np.multiply(data, factor), apply_on_array=True), 
    # filtering 
    Preprocessor('filter', l_freq=None, h_freq=high_cut_hz)
]

# Transform the data
preprocess(dataset, preprocessors)

### ----------------------------- Extract trial windows -----------------------------
# We merge stages 3 and 4 following AASM standards.
mapping = {  
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

window_size_s = trial_len_sec
sfreq = 100
window_size_samples = window_size_s * sfreq

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    picks="Fpz-Cz" if model_name == 'SleepStagerEldele2021' else None,
    preload=True,
    mapping=mapping
)
# window preprocessing
preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])

### ----------------------------- Create model -----------------------------
# Specify which GPU to run on to avoid collisions
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

# check if GPU is available
cuda = torch.cuda.is_available()  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if cuda:
    print('CUDA is available')
    torch.backends.cudnn.benchmark = True
set_random_seeds(seed=31, cuda=cuda)

# Extract number of channels and time steps from dataset
n_channels, input_size_samples = windows_dataset[0][0].shape

### ----------------------------- Training -----------------------------
dict_results = {}

# Sequences of 3 consecutive windows
n_windows = 3 
# Maximally overlapping sequences
n_windows_stride = 2
# Use 'test_percentage' of the dataset for validation
test_percentage = 0.2

splitted_by_subj = windows_dataset.split('subject')
subject_ids_lst = list(splitted_by_subj.keys())
for holdout_subj_id in subject_ids_lst:
    
    print(f'Hold out data from subject {holdout_subj_id}')
    
    ### ----------------------------------------
    ### ---------- PRE TRAINING ----------------
    ### ----------------------------------------

    ### ---------- Split dataset into pre-train set and fine-tune (holdout) set ----------
    pre_train_set = BaseConcatDataset([splitted_by_subj.get(f'{i}') for i in subject_ids_lst if i != holdout_subj_id])
    fine_tune_set = BaseConcatDataset([splitted_by_subj.get(f'{holdout_subj_id}'),])

    # Split pre-train set into pre-train-train and pre-train-test
    pre_train_test_set_size = int(len(pre_train_set) * test_percentage)
    pre_train_train_set = get_subset(pre_train_set, target_trial_num=len(pre_train_set)-pre_train_test_set_size)
    pre_train_test_set = get_subset(pre_train_set, target_trial_num=pre_train_test_set_size, from_back=True)

    # Extract sequences
    pre_train_train_sampler = SequenceSampler(
        pre_train_train_set.get_metadata(), n_windows, n_windows_stride, randomize=True
    )
    pre_train_train_set.target_transform = get_center_label if model_name != 'USleep' else None
    pre_train_test_sampler = SequenceSampler(
        pre_train_test_set.get_metadata(), n_windows, n_windows_stride, randomize=True
    )
    pre_train_test_set.target_transform = get_center_label if model_name != 'USleep' else None

    if model_name != 'USleep':
        y_pre_train_test = [pre_train_test_set[idx][1] for idx in pre_train_test_sampler] 
    else:
        y_pre_train_test = [pre_train_test_set[idx][1][1] for idx in pre_train_test_sampler]

    new_class_weights = compute_class_weight('balanced', classes=np.unique(y_pre_train_test), y=y_pre_train_test)
    # Only update class_weights if generated weights for all classes
    if len(new_class_weights) == n_classes:
        class_weights = new_class_weights

    # Create model
    model_kwargs = args.model_kwargs
    pre_train_model = model_object(
        n_chans = n_channels,
        sfreq = sfreq,
        n_outputs = n_classes,
        n_times = input_size_samples,
        **model_kwargs
    )
    if model_name != 'USleep':
        pre_train_model = nn.Sequential(
            # apply model on each 30-s window
            TimeDistributed(pre_train_model),  
            nn.Sequential(  
                # apply linear layer on concatenated feature vectors
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(pre_train_model.len_last_layer * n_windows, n_classes)
            )
        )

    # Send model to GPU
    if cuda:
        pre_train_model.cuda()

    train_bal_acc = EpochScoring(
        scoring=balanced_accuracy_multi, on_train=True, name='train_bal_acc',
        lower_is_better=False)
    valid_bal_acc = EpochScoring(
        scoring=balanced_accuracy_multi, on_train=False, name='valid_bal_acc',
        lower_is_better=False)
    callbacks = [
        ('train_bal_acc', train_bal_acc),
        ('valid_bal_acc', valid_bal_acc)
    ]
    pre_train_clf = EEGClassifier(
        pre_train_model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=torch.Tensor(class_weights).to(device),
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=False,
        iterator_train__sampler=pre_train_train_sampler,
        iterator_valid__sampler=pre_train_test_sampler,
        # using valid_set for validation
        train_split=predefined_split(pre_train_test_set),  
        optimizer__lr=lr,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
        classes=np.unique(y_pre_train_test),
    )
    pre_train_clf.initialize()

    # check if a pretrained model exists
    model_exist = True
    # Shouldn't have hard coded this
    temp_exp_name = 'baseline_2_5_pretrain'
    for file_end in ['_model.pkl', '_opt.pkl', '_history.json']:
        cur_file_path = os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}{file_end}')
        model_exist = model_exist and os.path.exists(cur_file_path) and os.path.getsize(cur_file_path) > 0

    if model_exist:
        ### Load trained model
        print(f'A pre-trained model for holdout subject {holdout_subj_id} exists')
        pre_train_clf.load_params(f_params=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_model.pkl'), 
                            f_optimizer=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_opt.pkl'), 
                            f_history=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_history.json'))
    else:
        ### ---------- Pre-training ----------
        print(f'Currently pre-training model with data from all subjects {len(pre_train_train_sampler)} ' + 
              f'sequences = {len(pre_train_train_set)} ' +
              f'trials but holding out {holdout_subj_id}')
        # Deactivate the default valid_acc callback. USleep wouldn't work without this line
        pre_train_clf.set_params(callbacks__valid_acc=None)
        _ = pre_train_clf.fit(pre_train_train_set, y=None, epochs=n_epochs)
        pre_train_clf.save_params(f_params=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_model.pkl'), 
                            f_optimizer=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_opt.pkl'), 
                            f_history=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_history.json'))
        
    if args.only_pretrain:
        continue

    ### ----------------------------------------
    ### ---------- FINE TUNING -----------------
    ### ----------------------------------------

    ### ---------- Split fine tune set into fine tune-train set and fine tune-valid set ----------
    fine_tune_test_set_size = int(len(fine_tune_set) * test_percentage)
    fine_tune_train_set = get_subset(fine_tune_set, target_trial_num=len(fine_tune_set)-fine_tune_test_set_size)
    fine_tune_test_set = get_subset(fine_tune_set, target_trial_num=fine_tune_test_set_size, from_back=True)
    
    # Extract fine_tune_test_set sequences
    fine_tune_test_sampler = SequenceSampler(
        fine_tune_test_set.get_metadata(), n_windows, n_windows_stride, randomize=True
    )
    fine_tune_test_set.target_transform = get_center_label if model_name != 'USleep' else None

    if model_name != 'USleep':
        y_fine_tune_test = [fine_tune_test_set[idx][1] for idx in fine_tune_test_sampler] 
    else:
        y_fine_tune_test = [fine_tune_test_set[idx][1][1] for idx in fine_tune_test_sampler]

    new_class_weights = compute_class_weight('balanced', classes=np.unique(y_fine_tune_test), y=y_fine_tune_test)
    # Only update class_weights if generated weights for all classes
    if len(new_class_weights) == n_classes:
        class_weights = new_class_weights

    # Baseline accuracy on holdout subject before fine tuning
    # finetune_baseline_acc = clf_predict_on_set(pre_train_clf, fine_tune_test_set)

    ### ---------- Fine tuning ----------
    # dict_subj_results = {0: [finetune_baseline_acc,]}
    dict_subj_results = {}
    fine_tune_train_set_size = len(fine_tune_train_set.get_metadata())
    for fine_tune_data_amount in data_amount_start + np.arange(0, 1 + (fine_tune_train_set_size - data_amount_start) // data_amount_step) * data_amount_step:
        
        final_accuracy = []

        for i in range(repetition):
            # Finetune with a subset of fine_tune_train_set
            fine_tune_train_subset = get_subset(fine_tune_train_set, target_trial_num=int(fine_tune_data_amount), random_sample=False)

            # Extract fine_tune_test_set sequences
            fine_tune_train_subset_sampler = SequenceSampler(
                fine_tune_train_subset.get_metadata(), n_windows, n_windows_stride, randomize=True
            )
            fine_tune_train_subset.target_transform = get_center_label if model_name != 'USleep' else None

            fine_tune_model = model_object(
                n_chans = n_channels,
                sfreq = sfreq,
                n_outputs = n_classes,
                n_times = input_size_samples,
                **model_kwargs
                )
            if model_name != 'USleep':
                fine_tune_model = nn.Sequential(
                    # apply model on each 30-s window
                    TimeDistributed(fine_tune_model),  
                    nn.Sequential(  
                        # apply linear layer on concatenated feature vectors
                        nn.Flatten(start_dim=1),
                        nn.Dropout(0.5),
                        nn.Linear(fine_tune_model.len_last_layer * n_windows, n_classes)
                    )
                )

            # Send model to GPU
            if cuda:
                fine_tune_model.cuda()

            fine_tune_clf = EEGClassifier(
                fine_tune_model,
                criterion=torch.nn.CrossEntropyLoss,
                criterion__weight=torch.Tensor(class_weights).to(device),
                optimizer=torch.optim.Adam,
                iterator_train__shuffle=False,
                iterator_train__sampler=fine_tune_train_subset_sampler,
                iterator_valid__sampler=fine_tune_test_sampler,
                train_split=predefined_split(fine_tune_test_set),  
                optimizer__lr=args.fine_tune_lr,
                batch_size=int(min(batch_size, fine_tune_data_amount // 2)),
                callbacks=callbacks,
                device=device,
                classes=np.unique(y_fine_tune_test),
            )
            fine_tune_clf.initialize()

            # Shouldn't have hard coded this
            temp_exp_name = 'baseline_2_5_pretrain'
            # Load pretrained model
            fine_tune_clf.load_params(f_params=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_model.pkl'), 
                            f_optimizer=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_opt.pkl'), 
                            f_history=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_history.json'))
            
            ## Freeze layers
            if args.freeze_most_layers:
                if args.fine_tune_freeze_layers_but is not None:
                    print(f'Freezing all parameters but {args.fine_tune_freeze_layers_but}')
                    freeze_all_param_but(fine_tune_clf.module, args.fine_tune_freeze_layers_but)
            else:
                if args.fine_tune_freeze_layer is not None:
                    for param_name in args.fine_tune_freeze_layer:
                        print(f'Freezing parameter: {param_name}')
                        freeze_param(fine_tune_clf.module, param_name)

            # Continue training / finetuning
            print(f'Fine tuning model for subject {holdout_subj_id} ' +
                  f'with {len(fine_tune_train_subset_sampler)} sequences ' +
                  f'= {len(fine_tune_train_subset)} trials (repetition {i})')
            _ = fine_tune_clf.partial_fit(fine_tune_train_subset, y=None, epochs=args.fine_tune_n_epochs)

            # Get final accuracy
            results_columns = ['valid_bal_acc',]
            df = pd.DataFrame(fine_tune_clf.history[:, results_columns], columns=results_columns, 
                            index=fine_tune_clf.history[:, 'epoch'])
            df = df.assign(valid_misclass=100 - 100 * df.valid_bal_acc)
            cur_final_acc = np.mean(df.tail(5).valid_bal_acc)
            final_accuracy.append(cur_final_acc)

        dict_subj_results.update({fine_tune_data_amount: final_accuracy})

    dict_results.update({holdout_subj_id: dict_subj_results})
    ### ----------------------------- Save results -----------------------------
    # Save results after done with a subject, in case server crashes
    # remove existing results file if one exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # save the updated one
    with open(file_path, 'wb') as f:
        pkl.dump(dict_results, f)

# check if results are saved correctly
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, 'rb') as f:
        dummy = pkl.load(f)
    print("Data was saved successfully.")
else:
    print(f"Error: File '{file_path}' does not exist or is empty. The save was insuccesful")

### ----------------------------- Plot results -----------------------------
df_results = pd.DataFrame(dict_results)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

for col in df_results.columns:
    y_values = [np.mean(lst) for lst in df_results[col]]
    y_errors = [np.std(lst) for lst in df_results[col]]
    ax1.errorbar(df_results.index * unit_multiplier, y_values, yerr=y_errors, label=f'Subject {col}')
    ax2.plot(df_results.index * unit_multiplier, y_values, label=f'Subject {col}')

df_results_rep_avg = df_results.applymap(lambda x: np.mean(x))
subject_averaged_df = df_results_rep_avg.mean(axis=1)
std_err_df = df_results_rep_avg.sem(axis=1)
conf_interval_df = stats.t.interval(significance_level, len(df_results_rep_avg.columns) - 1, 
                                    loc=subject_averaged_df, scale=std_err_df)

ax3.plot(subject_averaged_df.index * unit_multiplier, subject_averaged_df, label='Subject averaged')
ax3.fill_between(subject_averaged_df.index * unit_multiplier, conf_interval_df[0], conf_interval_df[1], 
                 color='b', alpha=0.3, label=f'{significance_level*100}% CI')

for ax in [ax1, ax2, ax3]:
    # ax.legend()
    ax.set_xlabel(f'Training data amount ({data_amount_unit})')
    ax.set_ylabel('Accuracy')

plt.suptitle(f'{model_name} on {dataset_name} Dataset \n Fine tune model for each subject, {repetition} reps each point')
plt.savefig(os.path.join(dir_results, f'{results_file_name}.png'))