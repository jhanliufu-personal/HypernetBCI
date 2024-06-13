# from torch.nn import Module
# from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
# from torch import nn
# import math
import numpy
# from tqdm import tqdm
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
import torch
# from itertools import chain
import matplotlib.pyplot as plt

from models.HypernetBCI import HyperBCINet
from utils import train_one_epoch, test_model

subject_id = 3
dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[subject_id,])

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
splitted = windows_dataset.split('run')
train_set = splitted['0train']  
valid_set = splitted['1test'] 

### ----------------------------------- CREATE PRIMARY NETWORK -----------------------------------
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    print('CUDA GPU is available. Use GPU for training')
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
n_channels = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

model = ShallowFBCSPNet(
    n_channels,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

# Send model to GPU
if cuda:
    model.cuda()

### ----------------------------------- CREATE HYPERNET BCI -----------------------------------
# embedding length = 729 when conv1d kernel size = 5, stide = 3, input_window_samples = 2250
embedding_shape = torch.Size([1, 749])
sample_shape = torch.Size([n_channels, input_window_samples])
myHNBCI = HyperBCINet(model, embedding_shape, sample_shape)

### ----------------------------------- MODEL TRAINING -----------------------------------
# these parameters work for the original ShallowFBSCP Net
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 20

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(valid_set, batch_size=batch_size)

optimizer = torch.optim.AdamW(
    myHNBCI.parameters(),
    lr=lr, 
    weight_decay=weight_decay
)
# optimizer = torch.optim.Adam(
#     chain(
#         myHNBCI.hypernet.parameters(), 
#         myHNBCI.embedder.parameters()
#     ), 
#     lr=1e-3
# )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs - 1
)
loss_fn = torch.nn.NLLLoss()

train_acc_lst = []
test_acc_lst = []
for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}: ", end="")

    train_loss, train_accuracy = train_one_epoch(
        train_loader, 
        myHNBCI, 
        loss_fn, 
        optimizer, 
        scheduler, 
        epoch, 
        device,
        print_batch_stats=False
    )

    # Update weight tensor for each evaluation pass
    myHNBCI.calibrate()
    test_loss, test_accuracy = test_model(
        test_loader, 
        myHNBCI, 
        loss_fn,
        print_batch_stats=False
    )
    myHNBCI.calibrating = False

    print(
        f"Train Accuracy: {100 * train_accuracy:.2f}%, "
        f"Average Train Loss: {train_loss:.6f}, "
        f"Test Accuracy: {100 * test_accuracy:.1f}%, "
        f"Average Test Loss: {test_loss:.6f}\n"
    )

    train_acc_lst.append(train_accuracy)
    test_acc_lst.append(test_accuracy)

### ----------------------------------- PLOT RESULTS -----------------------------------
dir_results = 'results/'

plt.figure()
plt.plot(train_acc_lst, label='Training accuracy')
plt.plot(test_acc_lst, label='Test accuracy')
plt.legend()
plt.savefig(f'{dir_results}HN_sanity_test_1.png')
