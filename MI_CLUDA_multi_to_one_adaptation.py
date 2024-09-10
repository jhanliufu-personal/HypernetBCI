import os
import numpy as np
from copy import deepcopy
import pickle as pkl
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
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
from utils import parse_training_config, get_subset

import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
subject_ids_lst = list(range(1, 14))
# subject_ids_lst = [1, 2]
preprocessed_dir = 'data/Schirrmeister2017_preprocessed'
if os.path.exists(preprocessed_dir) and os.listdir(preprocessed_dir):
    print('Preprocessed dataset exists')
    # If a preprocessed dataset exists
    windows_dataset = load_concat_dataset(
        path = preprocessed_dir,
        preload = True,
        ids_to_load = list(range(2 * subject_ids_lst[-1])),
        target_name = None,
    )
    sfreq = windows_dataset.datasets[0].raw.info['sfreq']
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
    print('Windows dataset created')

    # Save preprocessed dataset
    windows_dataset.save(
        path=preprocessed_dir,
        overwrite=True,
    )
    print(f'Dataset saved to {preprocessed_dir}')

dir_results = 'results/'
experiment_folder_name = f'MI_MAPU_multi_to_one_adaptation_{args.experiment_version}'
temp_exp_name = 'MAPU_multi_to_one_adapt'
# Create expriment folder
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)

pretrain_file_name = f'{experiment_folder_name}_pretrain_acc'
results_file_name = f'{experiment_folder_name}_results'
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
print(f'Saving pretrain accuracy at {pretrain_file_path}')
print(f'Saving results at {results_file_path}')

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

classes = list(range(args.n_classes))
n_chans = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]
splitted_by_subj = windows_dataset.split('subject')

dict_pretrain = {}
dict_results = {}

# Load existing outputs if they exist
if os.path.exists(pretrain_file_path):
    with open(pretrain_file_path, 'rb') as f:
        dict_pretrain = pkl.load(f)

if os.path.exists(results_file_path):
    with open(results_file_path, 'rb') as f:
        dict_results = pkl.load(f)

# adapt from multiple source subjects to one target subject
for i, target_subject in enumerate(subject_ids_lst):

    dict_key = f'adapt_to_{target_subject}'

    # check if the scenario is done
    model_param_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{temp_exp_name}_{dict_key}_pretrained_model_params.pth'
    )
    figure_title = f'{temp_exp_name}_{dict_key}_pretrain_acc_curve'
    train_acc_curve_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{figure_title}.png'
    )
    model_exist = os.path.exists(model_param_path) and os.path.getsize(model_param_path) > 0
    # Also check if the pretrain accuracy has been saved
    training_done = model_exist and (dict_train.get(dict_key) is not None)

    if dict_results.get(dict_key) is not None:
        continue

    print(f'Adapt model on multi-sources to target subject {target_subject}')
    ########################################################
    ###################### TRAINING ########################
    ########################################################

    # Prepare source dataset
    src_dataset = BaseConcatDataset([
        splitted_by_subj.get(f'{i}') 
        for i in subject_ids_lst 
        if i != target_subject
    ])
    src_train_set_lst = []
    src_valid_set_lst = []
    for key, val in src_dataset.split('subject').items():
        subj_splitted_by_run = val.split('run')
        cur_train_set = subj_splitted_by_run.get('0train')
        src_train_set_lst.append(cur_train_set)
        cur_valid_set = subj_splitted_by_run.get('1test')
        src_valid_set_lst.append(cur_valid_set)
    src_train_dataset = BaseConcatDataset(src_train_set_lst)
    src_valid_dataset = BaseConcatDataset(src_valid_set_lst)
    src_train_loader = DataLoader(
        src_train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    src_valid_loader = DataLoader(
        src_valid_dataset, 
        batch_size=args.batch_size
    )

    # prepare adaptation and test dataset from target subject
    target_dataset = splitted_by_subj.get(f'{target_subject}')
    target_dataset_splitted_by_run = target_dataset.split('run')
    target_train_dataset = target_dataset_splitted_by_run.get('0train')
    target_train_loader = DataLoader(
        target_train_dataset, 
        batch_size=args.batch_size,
        shuffle=True
    )
    target_train_iterator = iter(target_train_loader)
    target_test_dataset = target_dataset_splitted_by_run.get('1test')
    target_test_loader = DataLoader(
        target_test_dataset, 
        batch_size=args.batch_size
    )

    for epoch in range(1, args.n_epochs + 1):

        for source_batch_idx, (src_x, src_y, _) in enumerate(src_train_loader):

            try:
                target_batch = next(target_train_iterator)
            except StopIteration:
                # Since the target set is much smaller, re-initialize if it's exhausted
                target_train_loader = DataLoader(
                    target_train_dataset, 
                    batch_size=args.batch_size,
                    shuffle=True
                )
                target_train_iterator = iter(target_train_loader)
                target_batch = next(target_train_iterator)



    #     pretrain_accuracy = pretrain_correct / len(src_pretrain_loader.dataset)
    #     # Save pretrain accuracy
    #     pretrain_train_acc_lst.append(pretrain_accuracy)
    #     # Save batch-averaged tov loss
    #     pretrain_tov_loss_lst.append(batch_avg_tov_loss)
    #     # Save batch-averaged classification loss
    #     pretrain_cls_loss_lst.append(batch_avg_cls_loss)

    #     # Test model on validation set
    #     network.eval()
    #     valid_correct = 0
    #     with torch.no_grad():
    #         for _, (valid_x, valid_y, _) in enumerate(src_valid_loader):
    #             valid_x, valid_y = valid_x.to(device), valid_y.to(device)
    #             _, valid_prediction = network(valid_x)
    #             valid_correct += (valid_prediction.argmax(1) == valid_y).sum().item()

    #     # Save validation accuracy
    #     valid_accuracy = valid_correct / len(src_valid_loader.dataset)
    #     pretrain_test_acc_lst.append(valid_accuracy)
    #     print(
    #         f'[Epoch : {epoch}/{args.pretrain_n_epochs}] ' 
    #         f'training accuracy = {100 * pretrain_accuracy:.1f}% ' 
    #         f'validation accuracy = {100 * valid_accuracy:.1f}% '
    #         f'tov_loss = {batch_avg_tov_loss: .3e} '
    #         f'classification_loss = {batch_avg_cls_loss: .3e} '
    #     )

    # Plot and save the pretraining accuracy curves
    plt.figure()
    plt.plot(pretrain_train_acc_lst, label='Training accuracy')
    plt.plot(pretrain_test_acc_lst, label='Validation accuracy')
    plt.legend()
    plt.xlabel('Training epochs')
    plt.ylabel('Accuracy')
    plt.title(figure_title)
    plt.tight_layout()
    plt.savefig(pretrain_acc_curve_path)
    plt.close()

    # Save pretraining accuracies
    dict_pretrain.update({
        dict_key: {
            'pretrain_test_acc': pretrain_test_acc_lst,
            'pretrain_train_acc': pretrain_train_acc_lst,
            'pretrain_tov_loss': pretrain_tov_loss_lst,
            'pretrain_cls_loss': pretrain_cls_loss_lst
        }
    })
    if os.path.exists(pretrain_file_path):
        os.remove(pretrain_file_path)
    with open(pretrain_file_path, 'wb') as f:
        pkl.dump(dict_pretrain, f)

    # Save the source pretrained model and temporal verifier
    src_only_model = deepcopy(network.state_dict())
    torch.save(src_only_model, model_param_path)
