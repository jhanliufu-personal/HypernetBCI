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

from utils import get_subset, import_model, parse_training_config, freeze_param

import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
model_object = import_model(args.model_name)
subject_ids_lst = [2,]
dataset = MOABBDataset(dataset_name=args.dataset_name, subject_ids=subject_ids_lst)
print('Data loaded')

results_file_name = f'{args.model_name}_{args.dataset_name}_finetune_lr_{args.experiment_version}'
dir_results = 'results/'
file_path = os.path.join(dir_results, f'{results_file_name}.pkl')
fig_path = os.path.join(dir_results, f'{results_file_name}.png')

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
num = 5
stop = -4
lr_arr = np.logspace(start=stop - num + 1, stop=stop, num=num) * 6.5
# for this purpose, only use one subject
for holdout_subj_id in subject_ids_lst:
    fine_tune_set = BaseConcatDataset([splitted_by_subj.get(f'{holdout_subj_id}'),])
    finetune_splitted_by_run = fine_tune_set.split('run')
    finetune_subj_train_set = finetune_splitted_by_run.get('0train')
    finetune_subj_valid_set = finetune_splitted_by_run.get('1test')

    # Get a subset of train set for fine tuning
    finetune_subj_train_set = get_subset(finetune_subj_train_set, 500, random_sample=True)

    for cur_lr in lr_arr:

        cur_model = model_object(
            n_chans,
            args.n_classes,
            input_window_samples=input_window_samples,
            **(args.model_kwargs)
        )
        
        # # Send model to GPU
        # if cuda:
        #     cur_model.cuda()

        cur_clf = EEGClassifier(
            cur_model,
            criterion = torch.nn.NLLLoss,
            optimizer = torch.optim.AdamW,
            train_split = predefined_split(finetune_subj_valid_set), 
            optimizer__lr = cur_lr,
            optimizer__weight_decay = args.weight_decay,
            batch_size = args.batch_size,
            callbacks = [
                "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=args.n_epochs - 1)),
            ],
            device = device,
            classes = classes
        )
        cur_clf.initialize()

        temp_exp_name = f'baseline_2_6_pretrain'
        cur_clf.load_params(f_params=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_model.pkl'), 
                            f_optimizer=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_opt.pkl'), 
                            f_history=os.path.join(dir_results, f'{temp_exp_name}_without_subj_{holdout_subj_id}_history.json'))
        
        ## Freeze specified layers
        if args.fine_tune_freeze_layer is not None:
            for param_name in args.fine_tune_freeze_layer:
                print(f'Freezing parameter: {param_name}')
                freeze_param(cur_clf.module, param_name)

        # Set learning rate again to make sure it's right
        cur_clf.optimizer__lr = cur_lr

        # Send model to GPU
        if cuda:
            cur_model.cuda()

        # Fine tune
        print(f'Fine tuning model for subject {holdout_subj_id} with learning rate {cur_lr} = {cur_clf.optimizer__lr}')
        _ = cur_clf.partial_fit(finetune_subj_train_set, y=None, epochs=args.n_epochs)
        df = pd.DataFrame(cur_clf.history[:, results_columns], 
                          columns=results_columns,
                          index=cur_clf.history[:, 'epoch'])
        
        dict_results.update({cur_lr: df})

        ### ----------------------------- Save results -----------------------------
        # remove existing results file if one exists
        if os.path.exists(file_path):
            os.remove(file_path)
        # save the updated one
        with open(file_path, 'wb') as f:
            pickle.dump(dict_results, f)

for lr, df in dict_results.items():
    truncated_df = df.iloc[-(args.n_epochs):]
    plt.plot(truncated_df.index, truncated_df.valid_accuracy, label=f'lr={lr:.6f}')
plt.legend()
plt.savefig(fig_path)