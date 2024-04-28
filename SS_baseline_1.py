import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl
from scipy import stats

import torch
from torch import nn

from braindecode import EEGClassifier
from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_windows_from_events
from sklearn.preprocessing import scale as standard_scale
from braindecode.util import set_random_seeds
from braindecode.models import TimeDistributed
from braindecode.samplers import SequenceSampler

from sklearn.utils import compute_class_weight
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring

from utils import get_subset, import_model, get_center_label

import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
model_name = 'SleepStagerChambon2018'
model_object = import_model(model_name)
dataset_name = 'SleepPhysionet'
# dataset = SleepPhysionet(subject_ids=range(79), recording_ids=[1, 2,], crop_wake_mins=30)
# Test with a few subjects first
dataset = SleepPhysionet(subject_ids=[0, 1, 2,], recording_ids=[1, 2,], crop_wake_mins=30)

print('Data loaded')

experiment_version = 4
results_file_name = f'{model_name}_{dataset_name}_from_scratch_{experiment_version}'
dir_results = 'results/'
file_path = os.path.join(dir_results, f'{results_file_name}.pkl')

### ----------------------------- Training parameters -----------------------------
# Increment training set size by 'data_amount_step' each time
data_amount_step = 50
# data_amount_step = 400 # for testing purpose use super big data_amount_step
# Repeat for 'repetition' times for each training_data_amount
repetition = 1 
# 'n_classes' class classification task
n_classes = 5
# learning rate
lr = 1e-3
batch_size = 32
n_epochs = 50

### ----------------------------- Plotting parameters -----------------------------
data_amount_unit = 'min'
trial_len_sec = 30
if data_amount_unit == 'trial':
    unit_multiplier = 1
elif data_amount_unit == 'sec':
    unit_multiplier = trial_len_sec
elif data_amount_unit == 'min':
    unit_multiplier = trial_len_sec / 60

significance_level = 0.95

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
    preload=True,
    mapping=mapping
)
# window preprocessing
preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])

### ----------------------------- Create model -----------------------------
# check if GPU is available
cuda = torch.cuda.is_available()  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if cuda:
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

for subject_id, subject_dataset in windows_dataset.split('subject').items():

    dict_subj_results = {}

    # Split subject dataset into train set and test set
    test_set_size = int(len(subject_dataset) * test_percentage)
    train_set = get_subset(subject_dataset, target_trial_num=len(subject_dataset)-test_set_size)
    test_set = get_subset(subject_dataset, target_trial_num=test_set_size, from_back=True)
    # Extract test sequences
    test_sampler = SequenceSampler(
        test_set.get_metadata(), n_windows, n_windows_stride, randomize=True
    )
    test_set.target_transform = get_center_label
    y_test = [test_set[idx][1] for idx in test_sampler]
    new_class_weights = compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)

    # Only update class_weights if generated weights for all classes
    if len(new_class_weights) == n_classes:
        class_weights = new_class_weights

    train_set_size = len(train_set)
    for training_data_amount in np.arange(1, train_set_size // data_amount_step) * data_amount_step:

        final_accuracy = []

        for i in range(repetition):
            
            # Get subset of train set
            print(f'training_data_amount={training_data_amount}, train_set_size={train_set_size}')
            train_subset = get_subset(train_set, target_trial_num=int(training_data_amount), random_sample=False)
            # Extract train sequences
            train_sampler = SequenceSampler(
                train_subset.get_metadata(), n_windows, n_windows_stride, randomize=True
            )
            train_subset.target_transform = get_center_label

            # Balance for imbalanced class representation
            # y_train = [train_subset[idx][1] for idx in train_sampler]
            # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            # print(np.unique(y_train))
            # print(class_weights)

            # SleepStagerChambon2018
            feat_extractor = model_object(
                n_channels,
                sfreq,
                n_outputs=n_classes,
                n_times=input_size_samples,
                return_feats=True
            )
            model = nn.Sequential(
                # apply model on each 30-s window
                TimeDistributed(feat_extractor),  
                nn.Sequential(  
                    # apply linear layer on concatenated feature vectors
                    nn.Flatten(start_dim=1),
                    nn.Dropout(0.5),
                    nn.Linear(feat_extractor.len_last_layer * n_windows, n_classes)
                )
            )

            # Send model to GPU
            if cuda:
                model.cuda()

            print(f'Currently training for subject {subject_id} with {len(train_sampler)} sequences = {training_data_amount} trials')
            
            batch_size = int(min(32, training_data_amount // 2))
            
            train_bal_acc = EpochScoring(
                scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
                lower_is_better=False)
            valid_bal_acc = EpochScoring(
                scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
                lower_is_better=False)
            callbacks = [
                ('train_bal_acc', train_bal_acc),
                ('valid_bal_acc', valid_bal_acc)
            ]

            clf = EEGClassifier(
                model,
                criterion=torch.nn.CrossEntropyLoss,
                criterion__weight=torch.Tensor(class_weights).to(device),
                optimizer=torch.optim.Adam,
                iterator_train__shuffle=False,
                iterator_train__sampler=train_sampler,
                iterator_valid__sampler=test_sampler,
                # using valid_set for validation
                train_split=predefined_split(test_set),  
                optimizer__lr=lr,
                batch_size=batch_size,
                callbacks=callbacks,
                device=device,
                classes=np.unique(y_test),
            )
            clf.fit(train_subset, y=None, epochs=n_epochs)

            # Get final accuracy
            results_columns = ['valid_bal_acc',]
            df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns, 
                            index=clf.history[:, 'epoch'])

            df = df.assign(valid_misclass=100 - 100 * df.valid_bal_acc)
            cur_final_acc = np.mean(df.tail(5).valid_bal_acc)
            final_accuracy.append(cur_final_acc)

        dict_subj_results.update({training_data_amount: final_accuracy})

    dict_results.update({subject_id: dict_subj_results})
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

plt.suptitle(f'{model_name} on {dataset_name} Dataset \n Train model from scratch for each subject, {repetition} reps each point')
plt.savefig(os.path.join(dir_results, f'{results_file_name}.png'))


