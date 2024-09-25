import matplotlib.pyplot as plt
import os
import pickle as pkl
import numpy as np
import torch
from copy import deepcopy
from braindecode.datasets import MOABBDataset
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
    create_windows_from_events
)
from braindecode.datautil import load_concat_dataset
from braindecode.util import set_random_seeds
from torch.utils.data import DataLoader
from utils import (
    import_model, parse_training_config, 
    train_one_epoch, test_model
)
import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
# model_object = import_model(args.model_name)
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
        Preprocessor(lambda data: np.multiply(data, factor)), 
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
experiment_folder_name = f'Ensemble_baseline_{args.experiment_version}'
# Create expriment folder
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)
training_record_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'training.pkl')
results_file_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'results.pkl')

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

seed = args.random_seed
set_random_seeds(seed=seed, cuda=cuda)

### ----------------------------- Training -----------------------------
classes = list(range(args.n_classes))
n_chans = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]
dataset_splitted_by_subject = windows_dataset.split('subject')

dict_train = {}
if os.path.exists(training_record_path):
    with open(training_record_path, 'rb') as f:
        dict_train = pkl.load(f)

for subject_id in subject_ids_lst:
    subject_model_param_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/', 
        f'subject_{subject_id}_model_params.pth'
    )
    model_param_saved = os.path.exists(subject_model_param_path)
    training_record_saved = dict_train.get(subject_id) is not None
    training_done = model_param_saved and training_record_saved

    if not training_done:
        subject_dataset = dataset_splitted_by_subject.get(f'{subject_id}')
        subject_dataset_splitted_by_run = subject_dataset.split('run')
        subject_train_loader = DataLoader(
            subject_dataset_splitted_by_run.get('0train'),
            batch_size = args.batch_size,
            shuffle = True
        )
        subject_test_loader = DataLoader(
            subject_dataset_splitted_by_run.get('1test'),
            batch_size = args.batch_size
        )

        set_random_seeds(seed=seed, cuda=cuda)
        cur_model = ShallowFBCSPNet(
            n_chans,
            args.n_classes,
            input_window_samples=input_window_samples,
            final_conv_length="auto"
        )
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

        train_acc_lst = []
        test_acc_lst = []
        for epoch in range(1, args.n_epochs + 1):
            print(f"Epoch {epoch}/{args.n_epochs}: ", end="")

            train_loss, train_accuracy = train_one_epoch(
                subject_train_loader, 
                cur_model, 
                loss_fn, 
                optimizer, 
                scheduler, 
                epoch, 
                device,
                print_batch_stats=False
            )

            test_loss, test_accuracy = test_model(
                subject_test_loader, 
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

            train_acc_lst.append(train_accuracy)
            test_acc_lst.append(test_accuracy)

        # Save accuracies
        dict_train.update({
            subject_id: {
                'test_accuracy': test_acc_lst,
                'train_accuracy': train_acc_lst
            }
        })
        if os.path.exists(training_record_path):
            os.remove(training_record_path)
        with open(training_record_path, 'wb') as f:
            pkl.dump(dict_train, f)

        # Save the trained model
        print('Save trained model')
        torch.save(deepcopy(cur_model.state_dict()), subject_model_param_path)
    else:
        print(f'Training for subject {subject_id} already done')

### ----------------------------- Testing -----------------------------
'''
For each subject, try everyone else's model
'''
dict_results = {}
for subject_id in subject_ids_lst:

    print(f'Testing with data from subject {subject_id}')

    subject_dataset = dataset_splitted_by_subject.get(f'{subject_id}')
    subject_dataset_splitted_by_run = subject_dataset.split('run')
    subject_test_loader = DataLoader(
        subject_dataset_splitted_by_run.get('1test'),
        batch_size = args.batch_size
    )

    test_accuracy_by_other_subject = []
    for other_subject_id in subject_ids_lst:

        if other_subject_id == subject_id:
            test_accuracy_by_other_subject.append(1)
            continue

        print(f'Using model from subject {other_subject_id}')

        other_subject_model = ShallowFBCSPNet(
            n_chans,
            args.n_classes,
            input_window_samples=input_window_samples,
            final_conv_length="auto"
        )
        other_subject_model_param_path = os.path.join(
            dir_results, 
            f'{experiment_folder_name}/', 
            f'subject_{other_subject_id}_model_params.pth'
        )
        other_subject_model.load_state_dict(torch.load(other_subject_model_param_path))
        if cuda:
            other_subject_model.cuda()

        loss_fn = torch.nn.NLLLoss()
        _, test_accuracy = test_model(
            subject_test_loader, 
            other_subject_model, 
            loss_fn,
            print_batch_stats=False
        )
        test_accuracy_by_other_subject.append(test_accuracy)

    dict_results.update({
        subject_id: test_accuracy_by_other_subject
    })

    if os.path.exists(results_file_path):
        os.remove(results_file_path)
    with open(results_file_path, 'wb') as f:
        pkl.dump(dict_results, f)

print('Experiment done')