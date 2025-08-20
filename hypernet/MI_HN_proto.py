'''
Experiment design: todo
'''

import os
import pickle
from numpy.random import randint
import torch
from torch.utils.data import DataLoader

from braindecode.datasets import BaseConcatDataset
from braindecode.datautil import load_concat_dataset
from braindecode.models import ShallowFBCSPNet

from .HyperProtoNet import HyperProtoNet
from class_proto_attn_meta.class_proto_attn_meta_train_test import (
    get_balanced_loader, sample_episode
)
from utils import load_from_pickle

import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# ------------------------- For saving results -----------------------------
# --------------------------------------------------------------------------
dir_results = 'results/'
experiment_version = 1
experiment_folder_name = f'MI_HN_proto_{experiment_version}'
print(experiment_folder_name)
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)
results_path = os.path.join(
    dir_results, f'{experiment_folder_name}/', 'results.pkl'
)

# --------------------------------------------------------------------------
# ------------------------- Define meta parameters -------------------------
# --------------------------------------------------------------------------
n_classes = 4
lr = 6.5e-4
weight_decay = 0
train_n_epochs = 30
max_test_episodes=30
# batch_size = 72
gpu_number = '3'

n_proto = 5
n_query = 20
n_samples_per_class = n_proto + n_query

# --------------------------------------------------------------------------
# ------------------------- Load data --------------------------------------
# --------------------------------------------------------------------------
# subject_ids_lst = list(range(1, 14))
# subject_ids_lst = [1, 2, 3]
subject_ids_lst = [1, 2,]
preprocessed_dir = 'data/Schirrmeister2017_preprocessed'

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
sample_shape = torch.Size([n_chans, input_window_samples])
# this is the input shape of the final layer of ShallowFBCSPNet
embedding_shape = torch.Size([40, 144])  

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

# --------------------------------------------------------------------------
# ------------------------- Leave-one-out loop -----------------------------
# --------------------------------------------------------------------------
dataset_splitted_by_subject = windows_dataset.split('subject')
dict_results = load_from_pickle(results_path)
loss_fn = torch.nn.NLLLoss()

for i, target_subject in enumerate(subject_ids_lst):

    if dict_results.get(target_subject) is not None:
        print(f'Experiment for subject {target_subject} already done.')
        continue

    print(f'Adapt model from multiple sources to target subject {target_subject}')

    ### ------------------------------------------------------------------------
    ### ---------------------------------------- TRAINING ----------------------
    ### ------------------------------------------------------------------------ 
    
    # Create test dataloader
    target_dataset = dataset_splitted_by_subject.get(f'{target_subject}')
    target_loader = get_balanced_loader(
        target_dataset, n_classes, n_samples_per_class
    )

    # Create train dataloaders
    src_train_loaders_lst = []
    for k, v in dataset_splitted_by_subject.items():

        if k == f'{target_subject}':
            print(f'Excluding data from target subject {target_subject}')
            continue

        src_train_loaders_lst.append(
            get_balanced_loader(v, n_classes, n_samples_per_class)
        )
    
    src_subject_cnt = len(src_train_loaders_lst)

    # Create primary network
    primary_net = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto"
    )
    HPN = HyperProtoNet(primary_net, embedding_shape, num_classes=n_classes)
    HPN.to(device)

    optimizer = torch.optim.AdamW(
        HPN.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_n_epochs - 1
    )

    train_acc_lst = []
    test_acc_lst = []
    for epoch in range(1, train_n_epochs + 1):
        print(f"Epoch {epoch}/{train_n_epochs}: ", end="")

        # ------------- Training loop -------------
        HPN.train()
        train_correct, train_loss, train_cnt = 0, 0, 0
        for batch_idx in range(src_subject_cnt):
            # Randomize access order
            cur_loader = src_train_loaders_lst[randint(0, src_subject_cnt)]
            X, y, _ = next(iter(cur_loader))
            X, y = X.to(device), y.to(device)

            try:
                proto_x, proto_y, query_x, query_y = sample_episode(
                    X, y, num_classes=n_classes, 
                    n_prototype=n_proto, n_query=n_query
                )
            except AssertionError:
                print(
                    f'train batch {batch_idx}: failed to '
                    'sample enough class examples')
                continue

            print(f'after sample_episode in batch {batch_idx}')

            optimizer.zero_grad()
            # Pass query and prototype samples forward
            query_pred = HPN(query_x, proto_x, proto_y)
            print(f'after forward pass in batch {batch_idx}')

            loss = loss_fn(query_pred, query_y)
            print(f'after loss_fn in batch {batch_idx}')

            loss.backward()
            print(f'after loss.backward in batch {batch_idx}')

            optimizer.step()
            print(f'after optimizer.step() in batch {batch_idx}')

            optimizer.zero_grad()

            train_loss += loss.item()
            train_correct += (query_pred.argmax(1) == query_y).sum().item()
            train_cnt += query_y.size(0)

        scheduler.step()
        train_acc = train_correct / train_cnt
        train_acc_lst.append(train_acc)

        # ------------- Testing loop -------------
        HPN.eval()
        test_correct, test_loss, test_cnt = 0, 0, 0
        for batch_idx, (X, y, _) in enumerate(target_loader):
            if batch_idx > max_test_episodes:
                break

            X, y = X.to(device), y.to(device)
            try:
                proto_x, proto_y, query_x, query_y = sample_episode(
                    X, y, num_classes=n_classes, 
                    n_prototype=n_proto, n_query=n_query
                )
            except AssertionError:
                print(
                    f'test batch {batch_idx}: failed to '
                    'sample enough class examples')
                continue

            with torch.no_grad():
                # Pass query and prototype samples forward
                query_pred = HPN(query_x, proto_x, proto_y)

                loss = loss_fn(query_pred, query_y)
                test_loss += loss.item()
                test_correct += (query_pred.argmax(1) == query_y).sum().item()
                test_cnt += query_y.size(0)

        test_acc = test_correct / test_cnt
        test_acc_lst.append(test_acc)

        print(
            f"Train Accuracy: {100 * train_acc:.2f}%, "
            f"Test Accuracy: {100 * test_acc:.1f}%, "
        )

    # Save the pretrain accuracy and tensor distance
    dict_results.update({
        target_subject: {
            'train_acc': train_acc,
            'test_acc': test_acc
        }
    })
    with open(results_file_path, 'wb') as f:
        pickle.dump(dict_results, f)