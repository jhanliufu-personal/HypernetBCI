'''
HN cross-subject calibration experiment: hold out each person as the new arrival, and pre-train
the HyperBCI on everyone else put together as the pre-train pool. For the new arrival person, 
their data is splitted into calibration set and validation set. Varying amount of data is drawn
from the calibration set for calibration, then the calibrated model is evaluated using the test set.

The calibration process is unsupervised; the HN is expected to pick up relevant info from the
calibration set.
'''

import matplotlib.pyplot as plt
from braindecode.datasets import MOABBDataset, BaseConcatDataset
from numpy import multiply
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
    create_windows_from_events
)
from braindecode.datautil import load_concat_dataset
from braindecode.util import set_random_seeds

import torch
from torch.utils.data import DataLoader
import pandas as pd
from scipy import stats
import os
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain

from utils import (
    get_subset, import_model, parse_training_config, 
    train_one_epoch, test_model
)
from models.HypernetBCI import HyperBCINet
from models.Embedder import Conv1dEmbedder, ShallowFBCSPEmbedder, EEGConformerEmbedder
from models.Hypernet import LinearHypernet

import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
model_object = import_model(args.model_name)
# subject_ids_lst = list(range(1, 14))
subject_ids_lst = [1, 2, 3,]

preprocessed_dir = 'data/Schirrmeister2017_preprocessed'
if os.path.exists(preprocessed_dir) and os.listdir(preprocessed_dir):
    print('Preprocessed dataset exists')
    # If a preprocessed dataset exists
    windows_dataset = load_concat_dataset(
        path = preprocessed_dir,
        preload = True,
        ids_to_load = subject_ids_lst,
        target_name = None,
    )
    sfreq = 500
    print('Preprocessed dataset loaded')
else:
    dataset = MOABBDataset(dataset_name=args.dataset_name, subject_ids=subject_ids_lst)
    print('Raw dataset loaded')

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
    print('Dataset preprocessed')

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

    # Save preprocessed dataset
    windows_dataset.save(
        path=preprocessed_dir,
        overwrite=True,
    )
    print(f'Dataset saved to {preprocessed_dir}')

dir_results = 'results/'
experiment_folder_name = f'HYPER{args.model_name}_{args.dataset_name}_xsubj_calib_{args.experiment_version}'
# Create expriment folder
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)

pretrain_file_name = f'{experiment_folder_name}_pretrain_acc'
results_file_name = f'{experiment_folder_name}_results'
intermediate_outputs_file_name = f'{experiment_folder_name}_intermediate_outputs'
accuracy_figure_file_name = f'{experiment_folder_name}_accuracy'
pretrain_file_path = os.path.join(
    dir_results, 
    f'{experiment_folder_name}/', 
    f'{pretrain_file_name}.pkl'
)
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
print(f'Saving pretrain accuracy at {pretrain_file_path}')
print(f'Saving results at {results_file_path}')
print(f'Saving intermediate outputs at {intermediate_outputs_file_path}')
print(f'Saving accuracy figure at {accuracy_figure_file_path}')

# used to store pre-trained model parameters
temp_exp_name = f'HN_xsubj_calibration_{args.experiment_version}_pretrain'

### ----------------------------- Plotting parameters -----------------------------
match args.data_amount_unit:
    case 'trial':
        unit_multiplier = 1
    case 'sec':
        unit_multiplier = args.trial_len_sec
    case 'min':
        unit_multiplier = args.trial_len_sec / 60
    case _:
        unit_multiplier = args.trial_len_sec

### ----------------------------- Create model -----------------------------
# Specify which GPU to run on to avoid collisions
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

cuda = torch.cuda.is_available()
device_count = torch.cuda.device_count()
if cuda:
    print(f'{device_count} CUDA devices available, use GPU for training')
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
print('here')
print(splitted_by_subj.keys())

dict_pretrain = {}
dict_results = {}
results_columns = ['valid_accuracy',]
dict_intermediate_outputs = {}

# Load existing outputs if they exist
if os.path.exists(pretrain_file_path):
    with open(pretrain_file_path, 'rb') as f:
        dict_pretrain = pickle.load(f)

if os.path.exists(results_file_path):
    with open(results_file_path, 'rb') as f:
        dict_results = pickle.load(f)

if os.path.exists(intermediate_outputs_file_path):
    with open(intermediate_outputs_file_path, 'rb') as f:
        dict_intermediate_outputs = pickle.load(f)

for holdout_subj_id in subject_ids_lst:

    if (dict_results.get(holdout_subj_id) is not None 
        and dict_intermediate_outputs.get(holdout_subj_id) is not None):
        print(f'Experiment for subject {holdout_subj_id} already done.')
        continue

    ### -----------------------------------------------------------------------------------------
    ### ---------------------------------------- PRETRAINING ------------------------------------
    ### -----------------------------------------------------------------------------------------

    # Check if a pre-trained model exists
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
    # Also check if the pretrain accuracy has been saved
    model_exist = model_exist and (dict_pretrain.get(holdout_subj_id) is not None)

    sample_shape = torch.Size([n_chans, input_window_samples])

    # For conv1d embedder
    # embedding length = 729 when conv1d kernel size = 5, stide = 3, input_window_samples = 2250
    # embedding_shape = torch.Size([1, 749])
    # pretrain_embedder = Conv1dEmbedder(sample_shape, embedding_shape)

    # For ShallowFBCSP-based embedder
    # this is the input shape of the final layer of ShallowFBCSPNet
    # embedding_shape = torch.Size([40, 144, 1])
    # pretrain_embedder = ShallowFBCSPEmbedder(sample_shape, embedding_shape, 'drop', args.n_classes)
    
    # For EEGConformer-based embedder
    embedding_shape = torch.Size([32,])
    pretrain_embedder = EEGConformerEmbedder(sample_shape, embedding_shape, args.n_classes, sfreq)
    
    loss_fn = torch.nn.NLLLoss()

    if not model_exist:
        print(f'Pretraining model for subject {holdout_subj_id}')
        print(f'Hold out data from subject {holdout_subj_id}')
        ### ---------- Create pretrain dataset ----------
        pre_train_set = BaseConcatDataset([splitted_by_subj.get(f'{i}') for i in subject_ids_lst if i != holdout_subj_id])
        
        ### ---------------------------- CREATE PRIMARY NETWORK ----------------------------
        cur_model = model_object(
            n_chans,
            args.n_classes,
            input_window_samples=input_window_samples,
            **(args.model_kwargs)
        )
        # Load model parameters trained without hypernet
        cur_model.load_state_dict(
            torch.load(
                os.path.join(
                    dir_results, 
                    f'ShallowFBCSPNet_Schirrmeister2017_finetune_6/',
                    f'baseline_2_6_pretrain_without_subj_{holdout_subj_id}_model_params.pth'
                )
            )
        )
                    
        ### ----------------------------------- CREATE HYPERNET BCI -----------------------------------
        weight_shape = cur_model.final_layer.conv_classifier.weight.shape

        pretrain_hypernet = LinearHypernet(embedding_shape, weight_shape)

        pretrain_HNBCI = HyperBCINet(
            cur_model, 
            pretrain_embedder,
            embedding_shape, 
            sample_shape,
            pretrain_hypernet
        )
        # Send to GPU
        if cuda:
            # cur_model.cuda()
            pretrain_HNBCI.cuda()

        # optimizer = torch.optim.AdamW(
        #     pretrain_HNBCI.parameters(),
        #     lr=args.lr, 
        #     weight_decay=args.weight_decay,
        #     # This is for EEGConformer
        #     betas = (0.5, 0.999)
        # )
        optimizer = torch.optim.AdamW(
            # Only backprop to hypernet and embedder.
            chain(
                pretrain_HNBCI.hypernet.parameters(), 
                pretrain_HNBCI.embedder.parameters()
            ),
            lr=args.lr, 
            weight_decay=args.weight_decay,
            # This is for EEGConformer
            betas = (0.5, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.n_epochs - 1
        )

        ### ---------------------------- PREPARE PRETRAIN DATASETS ----------------------------
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
        pre_train_train_loader = DataLoader(pre_train_train_set, batch_size=args.batch_size, shuffle=True)
        pre_train_test_loader = DataLoader(pre_train_test_set, batch_size=args.batch_size)

        pretrain_train_acc_lst = []
        pretrain_test_acc_lst = []
        for epoch in range(1, args.n_epochs + 1):
            print(f"Epoch {epoch}/{args.n_epochs}: ", end="")

            train_loss, train_accuracy = train_one_epoch(
                pre_train_train_loader, 
                pretrain_HNBCI, 
                loss_fn, 
                optimizer, 
                scheduler, 
                epoch, 
                device,
                print_batch_stats=False,
                regularization_coef=args.regularization_coef,
                **(args.forward_pass_kwargs)
            )
            
            # Update weight tensor for each evaluation pass
            pretrain_HNBCI.calibrate()
            test_loss, test_accuracy = test_model(
                pre_train_test_loader, 
                pretrain_HNBCI, 
                loss_fn,
                print_batch_stats=False,
                regularization_coef=args.regularization_coef,
                **(args.forward_pass_kwargs)
            )
            pretrain_HNBCI.calibrating = False

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
        plt.close()

        # Save the pretrain accuracy
        dict_pretrain.update({
            holdout_subj_id: {
                'pretrain_test_acc': pretrain_test_acc_lst,
                'pretrain_train_acc': pretrain_train_acc_lst
            }
        })
        if os.path.exists(pretrain_file_path):
            os.remove(pretrain_file_path)
        with open(pretrain_file_path, 'wb') as f:
            pickle.dump(dict_pretrain, f)

        # Save the pre-trained model parameters to a file
        torch.save(
            {
                'HN_params_dict': pretrain_HNBCI.state_dict(), 
                'primary_params': pretrain_HNBCI.primary_params
            },
            model_param_path
        )

    else:
        print(f'A pretrained model for subject {holdout_subj_id} exists')

    if args.only_pretrain:
        continue

    ### -----------------------------------------------------------------------------------------
    ### ---------------------------------------- CALIBRATION ------------------------------------
    ### -----------------------------------------------------------------------------------------

    ### ----------------------------------- PREPARE CALIBRATION DATASETS -----------------------------------
    calibrate_set = BaseConcatDataset([splitted_by_subj.get(f'{holdout_subj_id}'),])
    ### THIS PART IS FOR BCNI2014001
    if args.dataset_name == 'BCNI2014001':
        calibrate_splitted_lst_by_run = list(calibrate_set.split('run').values())
        subj_calibrate_set = BaseConcatDataset(calibrate_splitted_lst_by_run[:-1])
        subj_valid_set = BaseConcatDataset(calibrate_splitted_lst_by_run[-1:])
    ### THIS PART IS FOR SHCIRRMEISTER 2017
    elif args.dataset_name == 'Schirrmeister2017':
        calibrate_splitted_lst_by_run = calibrate_set.split('run')
        subj_calibrate_set = calibrate_splitted_lst_by_run.get('0train')
        subj_valid_set = calibrate_splitted_lst_by_run.get('1test')

    ### Resume pretrained model
    calibrate_model = model_object(
        n_chans,
        args.n_classes,
        input_window_samples=input_window_samples,
        **(args.model_kwargs)
    )

    weight_shape = calibrate_model.final_layer.conv_classifier.weight.shape

    # calibrate_embedder = Conv1dEmbedder(sample_shape, embedding_shape)
    # calibrate_embedder = ShallowFBCSPEmbedder(sample_shape, embedding_shape, 'drop', args.n_classes)
    calibrate_embedder = EEGConformerEmbedder(sample_shape, embedding_shape, args.n_classes, sfreq)

    calibrate_hypernet = LinearHypernet(embedding_shape, weight_shape)

    calibrate_HNBCI = HyperBCINet(
        calibrate_model, 
        calibrate_embedder,
        embedding_shape, 
        sample_shape,
        calibrate_hypernet
    )
    pretrained_params = torch.load(model_param_path)
    calibrate_HNBCI.load_state_dict(pretrained_params['HN_params_dict'])
    calibrate_HNBCI.primary_params = deepcopy(pretrained_params['primary_params'])
    # Send to GPU
    if cuda:
        calibrate_HNBCI.cuda()

    ### Calculate baseline accuracy of the uncalibrated model on the calibrate_valid set
    # create validation dataloader
    subj_valid_loader = DataLoader(subj_valid_set, batch_size=args.batch_size)

    calibrate_HNBCI.calibrating = False
    _, calibrate_baseline_acc = test_model(
        subj_valid_loader, 
        calibrate_HNBCI, 
        loss_fn, 
        regularize_tensor_distance=False,
        **(args.forward_pass_kwargs)
    )
    print(f'Before calibrating for subject {holdout_subj_id}, the baseline accuracy is {calibrate_baseline_acc}')

    ### Calibrate with varying amount of new data
    dict_subj_results = {0: [calibrate_baseline_acc,]}
    dict_subj_intermediate_outputs = {}
    calibrate_trials_num = len(subj_calibrate_set.get_metadata())
    for calibrate_data_amount in np.arange(1, (calibrate_trials_num // args.data_amount_step) + 1) * args.data_amount_step:

        test_accuracy_lst = []
        aggregated_tensor_lst = []

        ### Since we're sampling randomly, repeat for 'repetition' times
        for i in range(args.repetition):

            ## Get current calibration samples
            subj_calibrate_subset = get_subset(
                subj_calibrate_set, 
                int(calibrate_data_amount), 
                random_sample=True
            )

            # Restore to the pre-trained state
            calibrate_HNBCI.load_state_dict(pretrained_params['HN_params_dict'])
            calibrate_HNBCI.primary_params = deepcopy(pretrained_params['primary_params'])
            # Send to GPU
            if cuda:
                # calibrate_model.cuda()
                calibrate_HNBCI.cuda()
    
            ### CALIBRATE! PASS IN THE ENTIRE SUBSET
            print(
                f'Calibrating model for subject {holdout_subj_id} ' +
                f'with {len(subj_calibrate_subset)} trials (repetition {i})'
            )

            # This dataloader returns the whole subset at once.
            subj_calibrate_loader = DataLoader(
                subj_calibrate_subset, 
                batch_size=len(subj_calibrate_subset), 
                shuffle=True
            )
            calibrate_HNBCI.calibrate()
            _, _ = test_model(
                subj_calibrate_loader, 
                calibrate_HNBCI, 
                loss_fn, 
                regularize_tensor_distance=False,
                **(args.forward_pass_kwargs)
            )
            calibrate_HNBCI.calibrating = False

            # Save intermediate outputs
            aggregated_tensor_lst.append(calibrate_HNBCI.aggregated_weight_tensor)

            # Test the calibrated model
            test_loss, test_accuracy = test_model(
                subj_valid_loader, 
                calibrate_HNBCI, 
                loss_fn,
                regularize_tensor_distance=False,
                **(args.forward_pass_kwargs)
            )

            # Save test accuracy
            test_accuracy_lst.append(test_accuracy)

            print(
                f"Test Accuracy: {100 * test_accuracy:.1f}%, "
                f"Average Test Loss: {test_loss:.6f}\n"
            )
        
        dict_subj_results.update(
            {
                calibrate_data_amount: test_accuracy_lst
            }
        )

        dict_subj_intermediate_outputs.update(
            {
                calibrate_data_amount: {
                    'aggregated_tensor': aggregated_tensor_lst
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
    # Save results and intermediate outputs after done with a subject, in case server crashes
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
    ax.set_xlabel(f'Calibration data amount ({args.data_amount_unit})')
    ax.set_ylabel('Accuracy')

plt.suptitle(
    f'HYPER{args.model_name} on {args.dataset_name} Dataset \n , ' +
    'Calibrate model for each subject (cross subject calibration), ' +
    f'{args.repetition} reps each point'
)
plt.savefig(accuracy_figure_file_path)