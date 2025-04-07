'''
Test-time adaptation using labeled support set. Calculate prototypical support
embedding for each label class from the labeled support set, and let task embeddings
attend over these class prototypes. Use meta-learning style training, each subject
being one "task"
'''

import os
import torch
import pickle as pkl
import random
from copy import deepcopy
from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader
# from braindecode.datasets import BaseConcatDataset

from models.Embedder import ShallowFBCSPEncoder
from models.Supportnet import Supportnet
from utils import (
    freeze_all_param_but, train_one_epoch_meta_subject, 
    test_model_episodic, load_from_pickle
    )
# from loss import contrastive_loss_btw_subject

# --------------------------------------------------------------------------
# ------------------------- Define meta parameters -------------------------
# --------------------------------------------------------------------------

# subject_ids_lst = list(range(1, 14))
subject_ids_lst = [1, 2, 3]
preprocessed_dir = 'data/Schirrmeister2017_preprocessed'

# Hyperparameters
n_classes = 4
batch_size = 72
lr = 6.5e-4
weight_decay = 0
n_epochs = 30
# temperature = 0.5

gpu_number = '2'
experiment_version = 1

dir_results = 'results/'
experiment_folder_name = f'class_prototype_attention_meta_{experiment_version}'
print(experiment_folder_name)
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)
training_record_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'training.pkl')
results_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'results.pkl')

# --------------------------------------------------------------------------
# ------------------------- Load data --------------------------------------
# --------------------------------------------------------------------------

# Load dataset
windows_dataset = load_concat_dataset(
        path = preprocessed_dir,
        preload = True,
        ids_to_load = list(range(2 * subject_ids_lst[-1])),
        target_name = None,
    )
sfreq = windows_dataset.datasets[0].raw.info['sfreq']
print('Preprocessed dataset loaded')

classes = list(range(n_classes))
n_chans = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
cuda = torch.cuda.is_available()
device_count = torch.cuda.device_count()
if cuda:
    print(f'{device_count} CUDA devices available, use GPU')
    torch.backends.cudnn.benchmark = True
    device = 'cuda'
else:
    print('No CUDA available, use CPU')
    device = 'cpu'

# adapt from multiple source subjects to one target subject
dataset_splitted_by_subject = windows_dataset.split('subject')

dict_training = load_from_pickle(training_record_path)
dict_results = load_from_pickle(results_path)

# --------------------------------------------------------------------------
# ------------------------- Leave-one-out loop -----------------------------
# --------------------------------------------------------------------------

for i, target_subject in enumerate(subject_ids_lst):

    dict_key = f'adapt_to_{target_subject}'

    support_net_folder = 'class_prototype_attention_1'
    support_encoder_path = os.path.join(
        dir_results, 
        support_net_folder,
        f'{dict_key}_support_encoder.pth'
    )
    supportnet_path = os.path.join(
        dir_results, 
        support_net_folder,
        f'{dict_key}_supportnet.pth'
    )
    print(f'Adapt model from multiple sources to target subject {target_subject}')

    # Create training (subject specific) and validation dataloaders
    dict_src_subject_loader = {}
    src_train_loaders_lst = []
    src_valid_loader_lst = []
    for k, v in dataset_splitted_by_subject.items():

        if k == f'{target_subject}':
            print(f'Excluding data from target subject {target_subject}')
            continue

        subject_splitted_by_run = v.split('run')
        
        # Train set
        subject_train_set = subject_splitted_by_run.get('0train')
        src_train_loaders_lst.append(
            DataLoader(
                subject_train_set, batch_size=batch_size, 
                shuffle=True, drop_last=True
            )
        )

        # Valid set
        subject_test_set = subject_splitted_by_run.get('1test')
        src_valid_loader_lst.append(
            DataLoader(
                subject_test_set, batch_size=batch_size, 
                shuffle=True, drop_last=True
            )
        )

# --------------------------------------------------------------------------
# ------------------------- Load support encoder ---------------------------
# --------------------------------------------------------------------------

    # Load support encoder
    support_encoder = ShallowFBCSPEncoder(
        torch.Size([n_chans, input_window_samples]),
        'drop',
        n_classes
    )
    support_encoder.model.load_state_dict(torch.load(support_encoder_path))
    # Freeze support encoder
    freeze_all_param_but(support_encoder.model, [])
    if cuda:
        support_encoder.cuda()

# --------------------------------------------------------------------------
# ------------------------- Train classifier net ---------------------------
# --------------------------------------------------------------------------

    # Prepare support net model
    supportnet = Supportnet(
        support_encoder,
        # Task encoder
        ShallowFBCSPEncoder(torch.Size([n_chans, input_window_samples]), 'drop', n_classes),
        # Classification head
        classifier = torch.nn.Sequential(
            # torch.nn.Conv2d(80, 4, kernel_size=(144, 1)),
            torch.nn.Conv2d(40, 4, kernel_size=(144, 1)),
            torch.nn.LogSoftmax(dim=1)
        )
    )
    if cuda:
        supportnet.cuda()
    
    if os.path.exists(supportnet_path) and os.path.getsize(supportnet_path) > 0:
        print(f'A supportnet trained without subject {target_subject} exists')
        supportnet.load_state_dict(torch.load(supportnet_path))
    else:

        pred_loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            supportnet.parameters(),
            lr=lr, 
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs - 1
        )

        train_acc_lst, valid_acc_lst = [], []
        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}/{n_epochs}: ", end="")

            train_loss, train_acc = train_one_epoch_meta_subject(
                src_train_loaders_lst, 
                supportnet,
                pred_loss_fn,
                optimizer,
                scheduler,
                device,
                num_classes=n_classes
            )

            valid_loss, valid_acc = test_model_episodic(
                random.choice(src_valid_loader_lst),
                supportnet,
                pred_loss_fn,
                num_classes=n_classes
            )

            train_acc_lst.append(train_acc)
            valid_acc_lst.append(valid_acc)

            print(
                f"Epoch {epoch}/{n_epochs}: train accuracy = {train_acc*100:.2f}," 
                f"test accuracy = {valid_acc*100:.2f}"
            )

        # Save supportnet parameters
        torch.save(deepcopy(supportnet.state_dict()), supportnet_path)

        # Save accuracy and loss
        dict_training.update({
            dict_key: {
                'train_accuracy': train_acc_lst,
                'valid_accuracy': valid_acc_lst
            }
        })
        with open(training_record_path, 'wb') as f:
            pkl.dump(dict_training, f)

# --------------------------------------------------------------------------
# ------------------------- Test on held-out subject -----------------------
# --------------------------------------------------------------------------

    if dict_results.get(dict_key) is None:
        target_dataset = dataset_splitted_by_subject.get(f'{target_subject}')
        target_loader = DataLoader(target_dataset, batch_size=batch_size)

        pred_loss_fn = torch.nn.CrossEntropyLoss()
        target_loss, target_acc = test_model_episodic(
            target_loader, 
            supportnet, 
            pred_loss_fn
        )

        dict_results.update({
            dict_key: target_acc
        })
        with open(results_path, 'wb') as f:
            pkl.dump(dict_results, f)

        print(f'Test accuracy on target subject: {target_acc*100:.2f}')

    else:
        print(
            f'Supportnet has been tested with ' 
            f'held-out data from subject {target_subject}'
        )