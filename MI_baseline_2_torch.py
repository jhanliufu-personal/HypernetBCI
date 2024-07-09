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
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
import pickle
import numpy as np

from torch.utils.data import DataLoader

from utils import (
    get_subset, import_model, parse_training_config, 
    freeze_param, train_one_epoch, test_model
)

import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
model_object = import_model(args.model_name)
# subject_ids_lst = list(range(1, 14))
subject_ids_lst = [1,]
dataset = MOABBDataset(dataset_name=args.dataset_name, subject_ids=subject_ids_lst)

print('Data loaded')

dir_results = 'results/'
experiment_folder_name = f'{args.model_name}_{args.dataset_name}_finetune_{args.experiment_version}'
results_file_name = f'{experiment_folder_name}_results'
intermediate_outputs_file_name = f'{experiment_folder_name}_intermediate_outputs'
accuracy_figure_file_name = f'{experiment_folder_name}_accuracy'
results_file_path = os.path.join(
    dir_results, 
    f'{experiment_folder_name}/', 
    f'{results_file_name}.pkl'
)
intermediate_outputs_file_path = os.path.join(
    dir_results, 
    f'{experiment_folder_name}/', 
    f'{intermediate_outputs_file_name}.pkl'
)
accuracy_figure_file_path = os.path.join(
    dir_results, 
    f'{experiment_folder_name}/', 
    f'{accuracy_figure_file_name}.png'
)
print(f'Saving results at {results_file_path}')
print(f'Saving intermediate outputs at {intermediate_outputs_file_path}')
print(f'Saving accuracy figure at {accuracy_figure_file_path}')

# used to store pre-trained model parameters
temp_exp_name = f'baseline_2_{args.experiment_version}_pretrain'

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

dict_intermediate_outputs = {}

for holdout_subj_id in subject_ids_lst:
    
    print(f'Hold out data from subject {holdout_subj_id}')
    
    ### ---------- Split dataset into pre-train set and fine-tune (holdout) set ----------
    fine_tune_set = BaseConcatDataset([splitted_by_subj.get(f'{holdout_subj_id}'),])

    ### -----------------------------------------------------------------------------------------
    ### ---------------------------------------- PRETRAINING ------------------------------------
    ### -----------------------------------------------------------------------------------------

    # Check if a pre-trained model exists
    # shouldn't have har coded it. Need to think of a better way to use pretrained models from other experiment
    # temp_exp_name = 'baseline_2_6_pretrain'
    # check if a pretrained model exists
    model_param_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{temp_exp_name}_without_subj_{holdout_subj_id}_model_params.pth'
    )
    pretrain_curve_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{temp_exp_name}_without_subj_{holdout_subj_id}_pretrain_curve.png'
    )
    model_exist = os.path.exists(model_param_path) and os.path.getsize(model_param_path) > 0

    if model_exist:
        if args.only_pretrain:
            continue
    else:
        print(f'Pretraining model for subject {holdout_subj_id}')
        print(f'Hold out data from subject {holdout_subj_id}')

        ### ---------- Split pre-train set into pre-train-train set and pre-train-test set ----------
        pre_train_set = BaseConcatDataset([splitted_by_subj.get(f'{i}') for i in subject_ids_lst if i != holdout_subj_id])
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
        
        # Send model to GPU
        if cuda:
            cur_model.cuda()

        optimizer = torch.optim.AdamW(
            cur_model.parameters(),
            lr=args.lr, 
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.n_epochs - 1
        )
        loss_fn = torch.nn.NLLLoss()

        pre_train_train_loader = DataLoader(pre_train_train_set, batch_size=args.batch_size, shuffle=True)
        pre_train_test_loader = DataLoader(pre_train_test_set, batch_size=args.batch_size)

        pretrain_train_acc_lst = []
        pretrain_test_acc_lst = []
        for epoch in range(1, args.n_epochs + 1):
            print(f"Epoch {epoch}/{args.n_epochs}: ", end="")

            train_loss, train_accuracy = train_one_epoch(
                pre_train_train_loader, 
                cur_model, 
                loss_fn, 
                optimizer, 
                scheduler, 
                epoch, 
                device,
                print_batch_stats=False
            )

            test_loss, test_accuracy = test_model(
                pre_train_test_loader, 
                cur_model, 
                loss_fn,
                print_batch_stats=False
            )
            print(
                f"Train Accuracy: {100 * train_accuracy:.2f}%, "
                f"Average Train Loss: {train_loss:.6f}, "
                f"Test Accuracy: {100 * test_accuracy:.1f}%, "
                f"Average Test Loss: {test_loss:.6f}\n"
            )

            pretrain_train_acc_lst.append(train_accuracy)
            pretrain_test_acc_lst.append(test_accuracy)

        # Plot and save the pretraining curve
        plt.figure()
        plt.plot(pretrain_train_acc_lst, label='Training accuracy')
        plt.plot(pretrain_test_acc_lst, label='Test accuracy')
        plt.legend()
        plt.xlabel('Training epochs')
        plt.ylabel('Accuracy')
        plt.title(f'{temp_exp_name}_without_subj_{holdout_subj_id}_pretrain_curve')
        plt.savefig(pretrain_curve_path)

        # Save the model weights to a file
        torch.save(
            cur_model.state_dict(), 
            model_param_path
        )

    ### -----------------------------------------------------------------------------------------
    ### ---------------------------------------- FINE TUNING ------------------------------------
    ### -----------------------------------------------------------------------------------------

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
    finetune_subj_valid_loader = DataLoader(finetune_subj_valid_set, batch_size=args.batch_size)

    finetune_model = model_object(
        n_chans,
        args.n_classes,
        input_window_samples=input_window_samples,
        **(args.model_kwargs)
    )
    finetune_model.load_state_dict(torch.load(model_param_path))
    # Send model to GPU
    if cuda:
        finetune_model.cuda()

    _, finetune_baseline_acc = test_model(finetune_subj_valid_loader, finetune_model, loss_fn)
    print(f'Before fine tuning for subject {holdout_subj_id}, the baseline accuracy is {finetune_baseline_acc}')

    ### Finetune with different amount of new data
    dict_subj_results = {0: [finetune_baseline_acc,]}
    dict_subj_intermediate_outputs = {}
    finetune_trials_num = len(finetune_subj_train_set.get_metadata())
    for finetune_training_data_amount in np.arange(1, (finetune_trials_num // args.data_amount_step) + 1) * args.data_amount_step:

        final_accuracy_lst = []
        final_tensor_lst = []

        ### Since we're sampling randomly, repeat for 'repetition' times
        for i in range(args.repetition):

            ## Get current finetune samples
            cur_finetune_subj_train_subset = get_subset(
                finetune_subj_train_set, 
                int(finetune_training_data_amount), 
                random_sample=True
            )
            cur_finetune_batch_size = int(min(finetune_training_data_amount // 2, args.batch_size))
            cur_finetune_subj_train_subset_loader = DataLoader(
                cur_finetune_subj_train_subset, 
                batch_size=cur_finetune_batch_size, 
                shuffle=True
            )

            # Restore to the pre-trained state
            finetune_model.load_state_dict(torch.load(model_param_path))
            # Send model to GPU
            if cuda:
                finetune_model.cuda()
    
            # Freeze specified layers
            if args.fine_tune_freeze_layer is not None:
                for param_name in args.fine_tune_freeze_layer:
                    print(f'Freezing parameter: {param_name}')
                    freeze_param(finetune_model, param_name)

            # Continue training / fine tuning
            print(
                f'Fine tuning model for subject {holdout_subj_id} ' +
                f'with {len(cur_finetune_subj_train_subset)} trials (repetition {i})' +
                f'with lr = {args.fine_tune_lr:.5f}'
            )

            finetune_optimizer = torch.optim.AdamW(
                finetune_model.parameters(),
                lr=args.fine_tune_lr, 
                weight_decay=args.fine_tune_weight_decay)
            finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                finetune_optimizer,
                T_max=args.fine_tune_n_epochs - 1
            )

            test_accuracy_lst = []
            for epoch in range(1, args.fine_tune_n_epochs + 1):
                print(f"Epoch {epoch}/{args.fine_tune_n_epochs}: ", end="")

                train_loss, train_accuracy = train_one_epoch(
                    cur_finetune_subj_train_subset_loader, 
                    finetune_model, 
                    loss_fn, 
                    finetune_optimizer, 
                    finetune_scheduler, 
                    epoch, 
                    device
                )
                test_loss, test_accuracy = test_model(finetune_subj_valid_loader, finetune_model, loss_fn)
                test_accuracy_lst.append(test_accuracy)

                print(
                    f"Train Accuracy: {100 * train_accuracy:.2f}%, "
                    f"Average Train Loss: {train_loss:.6f}, "
                    f"Test Accuracy: {100 * test_accuracy:.1f}%, "
                    f"Average Test Loss: {test_loss:.6f}\n"
                )
        
            final_accuracy_lst.append(np.mean(test_accuracy_lst[-5:]))
            # Save weights of the classifier after fine tuning
            final_tensor_lst.append(finetune_model.final_layer.conv_classifier.weight.clone().detach())

        dict_subj_results.update(
            {
                finetune_training_data_amount: final_accuracy_lst
            }
        )

        dict_subj_intermediate_outputs.update(
            {
                finetune_training_data_amount: {
                    'final_tensor': final_tensor_lst
                }
            }
        )

    dict_results.update(
        {
            holdout_subj_id: dict_subj_results
        }
    )

    dict_intermediate_outputs.update(
        {
            holdout_subj_id: dict_subj_intermediate_outputs
        }
    )

    ### ----------------------------- Save results -----------------------------
    # Save results after done with a subject, in case server crashes
    # remove existing pickle file if one exists
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
    # save the updated one
    with open(results_file_path, 'wb') as f:
        pickle.dump(dict_results, f)

    if os.path.exists(intermediate_outputs_file_path):
        os.remove(intermediate_outputs_file_path)
    # save the updated one
    with open(intermediate_outputs_file_path, 'wb') as f:
        pickle.dump(dict_intermediate_outputs, f)

# check if results are saved correctly
if os.path.exists(results_file_path) and os.path.getsize(results_file_path) > 0:
    with open(results_file_path, 'rb') as f:
        dummy = pickle.load(f)
    print("Results were saved successfully.")
else:
    print(f"Error: File '{results_file_path}' does not exist or is empty. The save was insuccesful")

# check if intermediate outputs are saved correctly
if os.path.exists(intermediate_outputs_file_path) and os.path.getsize(intermediate_outputs_file_path) > 0:
    with open(intermediate_outputs_file_path, 'rb') as f:
        dummy = pickle.load(f)
    print("Intermediate outputs were saved successfully.")
else:
    print(f"Error: File '{intermediate_outputs_file_path}' does not exist or is empty. The save was insuccesful")

### -----------------------------------------------------------------------------------------
### ---------------------------------------- PLOTTING ---------------------------------------
### -----------------------------------------------------------------------------------------
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
    scale=std_err_df
)

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
    ax.set_xlabel(f'Fine tune data amount ({args.data_amount_unit})')
    ax.set_ylabel('Accuracy')

plt.suptitle(
    f'{args.model_name} on {args.dataset_name} Dataset \n, ' 
    f'fine tune model for each subject, {args.repetition} reps each point'
)
plt.savefig(accuracy_figure_file_path)