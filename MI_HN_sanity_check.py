from torch.utils.data import DataLoader
import numpy as np
from itertools import chain

from braindecode.datasets import MOABBDataset, BaseConcatDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
from braindecode.models import ShallowFBCSPNet #, EEGConformer
from braindecode.util import set_random_seeds
import torch
import matplotlib.pyplot as plt

from models.HypernetBCI import HyperBCINet
from models.Embedder import EEGConformerEmbedder, ShallowFBCSPEmbedder, Conv1dEmbedder
from models.Hypernet import LinearHypernet
from utils import train_one_epoch, test_model
import os

subject_id = 3
# dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[subject_id,])
# Load data from all subjects
all_subject_id_lst = list(range(1, 14))
dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=all_subject_id_lst)

### ----------------------------------- PREPROCESSING -----------------------------------
# low cut frequency for filtering
low_cut_hz = 4.0  
# high cut frequency for filtering
high_cut_hz = 38.0  
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

transforms = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(
        lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
        factor=1e6,
    ),
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Transform the data
preprocess(dataset, transforms, n_jobs=-1)

### ----------------------------------- GET TRIAL DATA -----------------------------------
trial_start_offset_seconds = -0.5
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)

### ----------------------------------- GET DATASET -----------------------------------
# splitted = windows_dataset.split('run')
# train_set = splitted['0train']  
# valid_set = splitted['1test'] 

### ----------------------------------- CREATE PRIMARY NETWORK -----------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
cuda = torch.cuda.is_available() 
device = "cuda" if cuda else "cpu"
if cuda:
    print('CUDA GPU is available. Use GPU for training')
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
n_channels = windows_dataset[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

model = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)
# Load pretrained model params
dir_results = 'results'
load_model_param_from_exp = 'ShallowFBCSPNet_Schirrmeister2017_finetune_6'
model_param_path = f'{dir_results}/{load_model_param_from_exp}/baseline_2_6_pretrain_without_subj_{subject_id}_model_params.pth'
model.load_state_dict(torch.load(model_param_path))

# model = EEGConformer(
#     n_outputs=n_classes,
#     n_chans=n_channels,
#     n_times=input_window_samples,
#     sfreq=sfreq,
#     final_fc_length=5760
# )

# # Send model to GPU
# if cuda:
#     model.cuda()
# # Parallelize training if possible
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)

### ----------------------------------- CREATE HYPERNET BCI -----------------------------------
sample_shape = torch.Size([n_channels, input_window_samples])

# For conv1d embedder
# embedding length = 729 when conv1d kernel size = 5, stide = 3, input_window_samples = 2250
# embedding_shape = torch.Size([1, 749])
# my_embedder = Conv1dEmbedder(sample_shape, embedding_shape)

# For ShallowFBCSP-based embedder
# this is the input shape of the final layer of ShallowFBCSPNet
embedding_shape = torch.Size([40, 144, 1])
my_embedder = ShallowFBCSPEmbedder(sample_shape, embedding_shape, 'drop', n_classes)

# For EEGConformerembedder
# embedding_shape = torch.Size([32])
# my_embedder = EEGConformerEmbedder(sample_shape, embedding_shape, n_classes, sfreq)

weight_shape = model.final_layer.conv_classifier.weight.shape
my_hypernet = LinearHypernet(embedding_shape, weight_shape)

myHNBCI = HyperBCINet(
    model, 
    my_embedder,
    embedding_shape, 
    sample_shape,
    my_hypernet    
)

# Send myHNBCI to GPU
if cuda:
    model.cuda()
    myHNBCI.cuda()

### ----------------------------------- MODEL TRAINING -----------------------------------
# these parameters work for the original ShallowFBSCP Net
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 30

# these are for EEGConformer
# lr = 0.0002
# weight_decay = 0
# batch_size = 72
# n_epochs = 200

# optimizer = torch.optim.AdamW(
#     myHNBCI.parameters(),
#     lr=lr, 
#     # weight_decay=weight_decay,
#     betas = (0.5, 0.999)
# )

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=lr, 
#     # weight_decay=weight_decay,
#     betas = (0.5, 0.999)
# )

# optimizer = torch.optim.AdamW(
#     myHNBCI.parameters(),
#     lr=lr, 
#     weight_decay=weight_decay
# )

optimizer = torch.optim.Adam(
    chain(
        myHNBCI.hypernet.parameters(), 
        myHNBCI.embedder.parameters()
    ), 
    lr=lr,
    # betas = (0.5, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs - 1
)
loss_fn = torch.nn.NLLLoss()

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(valid_set, batch_size=batch_size)
splitted_by_subj = windows_dataset.split('subject')
pre_train_set = BaseConcatDataset([splitted_by_subj.get(f'{i}') for i in all_subject_id_lst if i != subject_id])
pre_train_train_set_lst = []
pre_train_test_set_lst = []
for key, val in pre_train_set.split('subject').items():
    subj_splitted_by_run = val.split('run')
    cur_train_set = subj_splitted_by_run.get('0train')
    pre_train_train_set_lst.append(cur_train_set)
    cur_test_set = subj_splitted_by_run.get('1test')
    pre_train_test_set_lst.append(cur_test_set)

pre_train_train_set = BaseConcatDataset(pre_train_train_set_lst)
pre_train_test_set = BaseConcatDataset(pre_train_test_set_lst)
pre_train_train_loader = DataLoader(pre_train_train_set, batch_size=batch_size, shuffle=True)
pre_train_test_loader = DataLoader(pre_train_test_set, batch_size=batch_size)

train_acc_lst = []
test_acc_lst = []
for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}: ", end="")

    train_loss, train_accuracy = train_one_epoch(
        # train_loader, 
        pre_train_train_loader,
        myHNBCI, 
        loss_fn, 
        optimizer, 
        scheduler, 
        epoch, 
        # device,
        print_batch_stats=False
    )

    # Update weight tensor for each evaluation pass
    myHNBCI.calibrate()
    test_loss, test_accuracy = test_model(
        # test_loader,
        pre_train_test_loader, 
        myHNBCI, 
        loss_fn,
        print_batch_stats=False
    )
    myHNBCI.calibrating = False

    # train_loss, train_accuracy = train_one_epoch(
    #     train_loader, 
    #     model, 
    #     loss_fn, 
    #     optimizer, 
    #     scheduler, 
    #     epoch, 
    #     device,
    #     print_batch_stats=False
    # )

    # test_loss, test_accuracy = test_model(
    #     test_loader, 
    #     model, 
    #     loss_fn,
    #     print_batch_stats=False
    # )

    print(
        f"Train Accuracy: {100 * train_accuracy:.2f}%, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Test Accuracy: {100 * test_accuracy:.1f}%, "
        f"Average Test Loss: {test_loss:.6f}\n"
    )

    train_acc_lst.append(train_accuracy)
    test_acc_lst.append(test_accuracy)

### ----------------------------------- PLOT RESULTS -----------------------------------

plt.figure()
plt.plot(train_acc_lst, label='Training accuracy')
plt.plot(test_acc_lst, label='Test accuracy')
plt.legend()

plt.xlabel('Training epochs')
plt.ylabel('Accuracy')
plt.title('HypernetBCI sanity check 6')

plt.savefig(f'{dir_results}/HN_sanity_test_6.png')
