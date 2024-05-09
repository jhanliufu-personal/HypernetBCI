'''
baseline 2 refers to experiments that test the "fine tune" approach.
Before fine tuning a model for a subject, the model is pre-trained on
all other subjects.

This is a very head-empty approach, since no distinction bewteen subjects
is made within the pre-train dataset.

For other approaches, check out the meta learning papers
'''

import matplotlib.pyplot as plt
from braindecode.datasets import MOABBDataset, BaseConcatDataset
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

from utils import get_subset, import_model, clf_predict_on_set, parse_training_config, freeze_param

import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
model_object = import_model(args.model_name)
subject_ids_lst = list(range(1, 3))
dataset = MOABBDataset(dataset_name=args.dataset_name, subject_ids=subject_ids_lst)

print('Data loaded')

results_file_name = f'{args.model_name}_{args.dataset_name}_finetune_{args.experiment_version}'
dir_results = 'results/'
# used to store pre-trained model parameters
temp_exp_name = f'baseline_2_{args.experiment_version}_pretrain'
file_path = os.path.join(dir_results, f'{results_file_name}.pkl')

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

### ----------------------------- Create model -----------------------------
# Specify which GPU to run on to avoid collisions
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

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

### ----------------------------- Training -----------------------------

classes = list(range(args.n_classes))
n_chans = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

splitted_by_subj = windows_dataset.split('subject')

dict_results = {}
results_columns = ['valid_accuracy',]

for holdout_subj_id in subject_ids_lst:
    
    print(f'Hold out data from subject {holdout_subj_id}')
    
    ### ---------- Split dataset into pre-train set and fine-tune (holdout) set ----------
    pre_train_set = BaseConcatDataset([splitted_by_subj.get(f'{i}') for i in subject_ids_lst if i != holdout_subj_id])
    fine_tune_set = BaseConcatDataset([splitted_by_subj.get(f'{holdout_subj_id}'),])

    ### ---------- Split pre-train set into pre-train-train set and pre-train-test set ----------
    ### THIS PART IS FOR BCNI2014001
    if args.dataset_name == 'BCNI2014001':
        pre_train_train_set_lst = []
        pre_train_test_set_lst = []
        pre_train_test_set_size = 1 # runs
        for key, val in pre_train_set.split('subject').items():
            subj_splitted_lst_by_run = list(val.split('run').values())
            pre_train_train_set_lst.extend(subj_splitted_lst_by_run[:-pre_train_test_set_size])
            pre_train_test_set_lst.extend(subj_splitted_lst_by_run[-pre_train_test_set_size:])
    
    ### THIS PART IS FOR SHCIRRMEISTER 2017
    elif args.dataset_name == 'Schirrmeister2017':
        pre_train_train_set_lst = []
        pre_train_test_set_lst = []
        for key, val in pre_train_set.split('subject').items():
            # print(f'Splitting data of subject {key}')
            subj_splitted_by_run = val.split('run')

            cur_train_set = subj_splitted_by_run.get('0train')
            # pre_train_train_set_lst.extend(cur_train_set)
            pre_train_train_set_lst.append(cur_train_set)

            cur_test_set = subj_splitted_by_run.get('1test')
            # pre_train_test_set_lst.extend(cur_test_set)
            pre_train_test_set_lst.append(cur_test_set)
    
    pre_train_train_set = BaseConcatDataset(pre_train_train_set_lst)
    pre_train_test_set = BaseConcatDataset(pre_train_test_set_lst)
    ### ------------------------------

    cur_model = model_object(
            n_chans,
            args.n_classes,
            input_window_samples=input_window_samples,
            **(args.model_kwargs)
        )
        
    cur_clf = EEGClassifier(
        cur_model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(pre_train_test_set), 
        optimizer__lr=args.lr,
        optimizer__weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=args.n_epochs - 1)),
        ],
        device=device,
        classes=classes,
        warm_start=False
    )
    cur_clf.initialize()

    # check if a pretrained model exists
    model_exist = True
    for file_end in ['_model.pkl', '_opt.pkl', '_history.json']:
        cur_file_path = os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}{file_end}')
        model_exist = model_exist and os.path.exists(cur_file_path) and os.path.getsize(cur_file_path) > 0

    if model_exist:
        ### Load trained model
        print(f'A pre-trained model for holdout subject {holdout_subj_id} exists')
        # shouldn't have har coded it. Need to think of a way to use pretrained models from other experiment
        temp_exp_name = 'baseline_2_6_pretrain'
        cur_clf.load_params(f_params=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_model.pkl'), 
                            f_optimizer=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_opt.pkl'), 
                            f_history=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_history.json'))
    else:
        ### ---------- Pre-training ----------
        print(f'Pre-training model with data from all subjects ({len(pre_train_train_set)} trials) but subject {holdout_subj_id}')
        _ = cur_clf.fit(pre_train_train_set, y=None, epochs=args.n_epochs)
        cur_clf.save_params(f_params=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_model.pkl'), 
                            f_optimizer=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_opt.pkl'), 
                            f_history=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_history.json'))

    if args.only_pretrain:
        continue

    ### ---------- Split fine tune set into fine tune-train set and fine tune-valid set ----------
    ### THIS PART IS FOR BCNI2014001
    if args.dataset_name == 'BCNI2014001':
        finetune_splitted_lst_by_run = list(fine_tune_set.split('run').values())
        finetune_subj_train_set = BaseConcatDataset(finetune_splitted_lst_by_run[:-1])
        finetune_subj_valid_set = BaseConcatDataset(finetune_splitted_lst_by_run[-1:])
    ### THIS PART IS FOR SHCIRRMEISTER 2017
    elif args.dataset_name == 'Schirrmeister2017':
        finetune_splitted_by_run = fine_tune_set.split('run')
        finetune_subj_train_set = finetune_splitted_by_run.get('0train')
        finetune_subj_valid_set = finetune_splitted_by_run.get('1test')
    ### ------------------------------

    ### Baseline accuracy on the finetune_valid set
    finetune_baseline_acc = clf_predict_on_set(cur_clf, finetune_subj_valid_set)
    print(f'Before finetuning for subject {holdout_subj_id}, the baseline accuracy is {finetune_baseline_acc}')

    ### ---------- Fine tuning ----------
    dict_subj_results = {0: [finetune_baseline_acc,]}

    ### Finetune with different amount of new data
    finetune_trials_num = len(finetune_subj_train_set.get_metadata())
    for finetune_training_data_amount in np.arange(1, (finetune_trials_num // args.data_amount_step) + 1) * args.data_amount_step:

        final_accuracy = []
        
        ### Since we're sampling randomly, repeat for 'repetition' times
        for i in range(args.repetition):

            ## Get current finetune samples
            cur_finetune_subj_train_subset = get_subset(finetune_subj_train_set, int(finetune_training_data_amount), random_sample=True)
    
            finetune_model = model_object(
                n_chans,
                args.n_classes,
                input_window_samples=input_window_samples,
                **(args.model_kwargs)
            )
    
            cur_finetune_batch_size = int(min(finetune_training_data_amount // 2, args.batch_size))
            
            new_clf = EEGClassifier(
                finetune_model,
                criterion=torch.nn.NLLLoss,
                optimizer=torch.optim.AdamW,
                train_split=predefined_split(finetune_subj_valid_set), 
                optimizer__lr=args.fine_tune_lr,
                optimizer__weight_decay=args.fine_tune_weight_decay,
                batch_size=cur_finetune_batch_size,
                callbacks=[
                    "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=args.fine_tune_n_epochs - 1)),
                ],
                device=device,
                classes=classes,
            )
            new_clf.initialize()
            
            ## Load pretrained model
            # shouldn't have har coded it. Need to think of a way to use pretrained models from other experiment
            temp_exp_name = 'baseline_2_6_pretrain'
            new_clf.load_params(f_params=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_model.pkl'), 
                                f_optimizer=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_opt.pkl'), 
                                f_history=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_history.json'))
    
            ## Freeze specified layers
            if args.fine_tune_freeze_layer is not None:
                for param_name in args.fine_tune_freeze_layer:
                    print(f'Freezing parameter: {param_name}')
                    freeze_param(new_clf.module, param_name)

            ## Continue training / finetuning
            print(f'Fine tuning model for subject {holdout_subj_id} with {finetune_training_data_amount} = {len(cur_finetune_subj_train_subset)} trials (repetition {i})')
            _ = new_clf.partial_fit(cur_finetune_subj_train_subset, y=None, epochs=args.fine_tune_n_epochs)
    
            ## Get results after fine tuning
            df = pd.DataFrame(new_clf.history[:, results_columns], columns=results_columns,)
    
            cur_final_acc = np.mean(df.tail(5).valid_accuracy)
            final_accuracy.append(cur_final_acc)
        
        dict_subj_results.update({finetune_training_data_amount: final_accuracy})

    dict_results.update({holdout_subj_id: dict_subj_results})
    ### ----------------------------- Save results -----------------------------
    # Save results after done with a subject, in case server crashes
    # remove existing results file if one exists
    if os.path.exists(file_path):
        os.remove(file_path)
    # save the updated one
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
conf_interval_df = stats.t.interval(args.significance_level, len(df_results_rep_avg.columns) - 1, 
                                    loc=subject_averaged_df, scale=std_err_df)

ax3.plot(subject_averaged_df.index * unit_multiplier, subject_averaged_df, label='Subject averaged')
ax3.fill_between(subject_averaged_df.index * unit_multiplier, conf_interval_df[0], conf_interval_df[1], 
                 color='b', alpha=0.3, label=f'{args.significance_level*100}% CI')

for ax in [ax1, ax2, ax3]:
    ax.legend()
    ax.set_xlabel(f'Fine tune data amount ({args.data_amount_unit})')
    ax.set_ylabel('Accuracy')

plt.suptitle(f'{args.model_name} on {args.dataset_name} Dataset \n fine tune model for each subject, {args.repetition} reps each point')
plt.savefig(os.path.join(dir_results, f'{results_file_name}.png'))