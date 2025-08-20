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
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader

from core.utils import (
    get_subset, import_model, train_one_epoch, test_model, parse_training_config
)

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
model_object = import_model(args.model_name)
subject_ids_lst = list(range(1, 14))
dataset = MOABBDataset(dataset_name=args.dataset_name, subject_ids=subject_ids_lst)

print('Data loaded')

results_file_name = f'{args.model_name}_{args.dataset_name}_from_scratch_{args.experiment_version}'
dir_results = 'results/'

### ----------------------------- Plotting parameters -----------------------------
if args.data_amount_unit == 'trial':
    unit_multiplier = 1
elif args.data_amount_unit == 'sec':
    unit_multiplier = args.trial_len_sec
elif args.data_amount_unit == 'min':
    unit_multiplier = args.trial_len_sec / 60

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
# Specify which GPU to run on to avoid collisions
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

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

classes = list(range(args.n_classes))
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
    cur_valid_loader = DataLoader(subj_valid_set, batch_size=args.batch_size)

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

    for training_data_amount in np.arange(1, train_trials_num // args.data_amount_step) * args.data_amount_step:
    
        final_accuracy = []

        for i in range(args.repetition):

            # Get current training set
            cur_train_set = get_subset(subj_train_set, int(training_data_amount), random_sample=True)

            cur_model = model_object(
                n_chans,
                args.n_classes,
                input_window_samples=input_window_samples,
                final_conv_length='auto',
            )
            
            cur_batch_size = int(min(training_data_amount // 2, args.batch_size))
        
            optimizer = torch.optim.AdamW(
                cur_model.parameters(),
                lr=args.lr, 
                weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.n_epochs - 1
            )
            loss_fn = torch.nn.NLLLoss()

            cur_train_loader = DataLoader(
                cur_train_set, 
                batch_size=args.batch_size, 
                shuffle=True
            )
            
            test_accuracy_lst = []
            for epoch in range(1, args.n_epochs + 1):
                print(f"Epoch {epoch}/{args.n_epochs}: ", end="")

                train_loss, train_accuracy = train_one_epoch(
                    cur_train_loader, 
                    cur_model, 
                    loss_fn, 
                    optimizer, 
                    scheduler, 
                    epoch, 
                    device,
                    print_batch_stats=False
                )

                test_loss, test_accuracy = test_model(
                    cur_valid_loader, 
                    cur_model, 
                    loss_fn,
                    print_batch_stats=False
                )
                test_accuracy_lst.append(test_accuracy)

                print(
                    f"Train Accuracy: {100 * train_accuracy:.2f}%, "
                    f"Average Train Loss: {train_loss:.6f}, "
                    f"Test Accuracy: {100 * test_accuracy:.1f}%, "
                    f"Average Test Loss: {test_loss:.6f}\n"
                )

            dict_subj_results.update({training_data_amount: np.mean(test_accuracy_lst[-5:])})

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
    ax1.errorbar(
        df_results.index * unit_multiplier, 
        y_values, 
        yerr=y_errors, 
        label=f'Subject {col}'
    )
    ax2.plot(
        df_results.index * unit_multiplier, 
        y_values, 
        label=f'Subject {col}'
    )

df_results_rep_avg = df_results.applymap(lambda x: np.mean(x))
subject_averaged_df = df_results_rep_avg.mean(axis=1)
std_err_df = df_results_rep_avg.sem(axis=1)
conf_interval_df = stats.t.interval(
    args.significance_level, 
    len(df_results_rep_avg.columns) - 1, 
    loc=subject_averaged_df, 
    scale=std_err_df)

ax3.plot(
    subject_averaged_df.index * unit_multiplier, 
    subject_averaged_df, 
    label='Subject averaged'
)
ax3.fill_between(
    subject_averaged_df.index * unit_multiplier, 
    conf_interval_df[0], 
    conf_interval_df[1], 
    color='b', 
    alpha=0.3, 
    label=f'{args.significance_level*100}% CI'
)

for ax in [ax1, ax2, ax3]:
    ax.legend()
    ax.set_xlabel(f'Training data amount ({args.data_amount_unit})')
    ax.set_ylabel('Accuracy')

plt.suptitle(
    f'{args.model_name} on {args.dataset_name} Dataset \n , ' + 
    'Train model from scratch for each subject ' +
    f'{args.repetition} reps each point'
)
plt.savefig(os.path.join(dir_results, f'{results_file_name}.png'))