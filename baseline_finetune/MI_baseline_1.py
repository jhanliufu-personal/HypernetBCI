'''
baseline 1 refers to experiments that test the "train from scratch" approach.
Model is trained from scratch for each subject.
'''

import matplotlib.pyplot as plt
from braindecode.datasets import MOABBDataset
from numpy import multiply
from braindecode.preprocessing import (Preprocessor,
                                       exponential_moving_standardize,
                                       preprocess)
from braindecode.preprocessing import create_windows_from_events
import torch
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
import pickle
import numpy as np

from utils import get_subset, import_model

### ----------------------------- Experiment parameters -----------------------------
model_name = 'ShallowFBCSPNet'
model_object = import_model(model_name)

dataset_name = 'Schirrmeister2017'
subject_ids_lst = list(range(1, 14))
dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_ids_lst)

print('Data loaded')

experiment_version = 3
results_file_name = f'{model_name}_{dataset_name}_from_scratch_{experiment_version}'
dir_results = 'results/'

### ----------------------------- Training parameters -----------------------------
# Increment training set size by 'data_amount_step' each time
data_amount_step = 20
# Repeat training with a certain training set size for 'repetition' times
repetition = 10
# 'n_classes' class classification task
n_classes = 4
# learning rate
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
# training epochs
n_epochs = 40

### ----------------------------- Plotting parameters -----------------------------
data_amount_unit = 'min'
trial_len_sec = 4
if data_amount_unit == 'trial':
    unit_multiplier = 1
elif data_amount_unit == 'sec':
    unit_multiplier = trial_len_sec
elif data_amount_unit == 'min':
    unit_multiplier = trial_len_sec / 60

significance_level = 0.95

### ----------------------------- Preprocessing -----------------------------
low_cut_hz = 4.  
high_cut_hz = 38. 
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    # Keep EEG sensors
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  
    # Convert from V to uV
    Preprocessor(lambda data: multiply(data, factor)), 
    # Bandpass filter
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  
    # Exponential moving standardization
    Preprocessor(exponential_moving_standardize,  
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors, n_jobs=-1)

print('Preprocessing done')

### ----------------------------- Extract trial windows -----------------------------
trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

print('Trial windows extracted')

### ----------------------------- Create model -----------------------------
# check if GPU is available, if True choose to use it
cuda = torch.cuda.is_available()  
if cuda:
    print('CUDA available, use GPU for training')
    torch.backends.cudnn.benchmark = True
    device = 'cuda'
else:
    print('No CUDA available, use CPU for training')
    device = 'cpu'

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_chans = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

### ----------------------------- Training -----------------------------
dict_results = {}

for subj_id, subj_dataset in windows_dataset.split('subject').items():

    dict_subj_results = {}

    ### Split by train and test sessions
    splitted_by_run = subj_dataset.split('run')
    subj_train_set = splitted_by_run.get('0train')
    subj_valid_set = splitted_by_run.get('1test')

    ### Use the last "valid_set_size" number of sets for testing
    # splitted_lst_by_run = list(subj_dataset.split('run').values())
    # subj_train_set = BaseConcatDataset(splitted_lst_by_run[:-valid_set_size])
    # subj_valid_set = BaseConcatDataset(splitted_lst_by_run[-valid_set_size:])

    # splitted_by_run = subj_dataset.split('session')
    # subj_train_set = BaseConcatDataset(splitted_lst_by_run[:-valid_set_size])
    # subj_train_set = splitted_by_run.get('0')
    # subj_valid_set = BaseConcatDataset(splitted_lst_by_run[-valid_set_size:])
    # subj_valid_set = splitted_by_run.get('1')
    
    train_trials_num = len(subj_train_set.get_metadata())

    for training_data_amount in np.arange(1, train_trials_num // data_amount_step) * data_amount_step:
    
        final_accuracy = []

        for i in range(repetition):

            cur_model = model_object(
                n_chans,
                n_classes,
                input_window_samples=input_window_samples,
                final_conv_length='auto',
            )
            
            cur_batch_size = int(min(training_data_amount // 2, batch_size))
        
            # Initialize EEGClassifier
            cur_clf = EEGClassifier(
                cur_model,
                criterion=torch.nn.NLLLoss,
                optimizer=torch.optim.AdamW,
                train_split=predefined_split(subj_valid_set),  # using valid_set for validation
                optimizer__lr=lr,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                callbacks=[
                    "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
                ],
                device=device,
                classes=classes,
            )
        
            # Get current training set
            cur_train_set = get_subset(subj_train_set, int(training_data_amount), random_sample=True)
        
            # Fit model
            print(f'Training model for subject {subj_id} with {training_data_amount} = {len(cur_train_set)} trials (repetition {i})')
            _ = cur_clf.fit(cur_train_set, y=None, epochs=n_epochs)
        
            # results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
            results_columns = ['train_accuracy', 'valid_accuracy',]
            df = pd.DataFrame(cur_clf.history[:, results_columns], columns=results_columns,
                              index=cur_clf.history[:, 'epoch'])
            
            # get percent of misclass for better visual comparison to loss
            df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                           valid_misclass=100 - 100 * df.valid_accuracy)
        
            cur_final_acc = np.mean(df.tail(5).valid_accuracy)
            final_accuracy.append(cur_final_acc)
            dict_subj_results.update({training_data_amount: final_accuracy})

        dict_results.update({subj_id: dict_subj_results})

### ----------------------------- Save results -----------------------------
file_path = os.path.join(dir_results, f'{results_file_name}.pkl')

with open(file_path, 'wb') as f:
    pickle.dump(dict_results, f)

# check if results are saved correctly
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, 'rb') as f:
        dummy = pickle.load(f)
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
    ax.legend()
    ax.set_xlabel(f'Training data amount ({data_amount_unit})')
    ax.set_ylabel('Accuracy')

plt.suptitle(f'{model_name} on {dataset_name} Dataset \n Train model from scratch for each subject, {repetition} reps each point')
plt.savefig(os.path.join(dir_results, f'{results_file_name}.png'))