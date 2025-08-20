import os
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy
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
from baseline_CLUDA.CLUDA_algorithm import CLUDA_NN
from baseline_CLUDA.CLUDA_augmentations import Augmenter
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
experiment_folder_name = f'MI_CLUDA_multi_to_one_adaptation_{args.experiment_version}'
temp_exp_name = 'CLUDA_multi_to_one_adapt'
# Create expriment folder
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)

train_file_name = f'{experiment_folder_name}_train_acc'
train_file_path = os.path.join(
    dir_results, 
    f'{experiment_folder_name}/', 
    f'{train_file_name}.pkl'
)
print(f'Saving training outcome at {train_file_path}')

embedding_file_name = f'{experiment_folder_name}_embeddings'
embedding_file_path = os.path.join(
    dir_results, 
    f'{experiment_folder_name}/', 
    f'{embedding_file_name}.pkl'
)
print(f'Saving embeddings at {embedding_file_path}')

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

# Save data
dict_train = {}
dict_embeddings = {}

if os.path.exists(train_file_path):
    with open(train_file_path, 'rb') as f:
        dict_train = pkl.load(f)

# adapt from multiple source subjects to one target subject
for i, target_subject in enumerate(subject_ids_lst):

    dict_key = f'adapt_to_{target_subject}'

    # check if a trained model exists
    model_param_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{temp_exp_name}_{dict_key}_model_params.pth'
    )
    acc_figure_title = f'{temp_exp_name}_{dict_key}_acc_curve'
    acc_curve_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{acc_figure_title}.png'
    )
    loss_figure_title = f'{temp_exp_name}_{dict_key}_loss_curve'
    loss_curve_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{loss_figure_title}.png'
    )
    model_exist = os.path.exists(model_param_path) and os.path.getsize(model_param_path) > 0
    training_done = model_exist and (dict_train.get(dict_key) is not None)

    if training_done:
        continue

    print(f'Adapt model on multi-sources to target subject {target_subject}')
    ########################################################
    ###################### TRAINING ########################
    ########################################################

    # Prepare source and target dataset
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
    src_train_loader = DataLoader(
        BaseConcatDataset(src_train_set_lst), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    src_valid_loader = DataLoader(
        BaseConcatDataset(src_valid_set_lst), 
        batch_size=args.batch_size
    )
    trg_dataset = splitted_by_subj.get(f'{target_subject}')
    trg_dataset_splitted_by_run = trg_dataset.split('run')
    trg_train_loader = DataLoader(
        trg_dataset_splitted_by_run.get('0train'), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    trg_train_iter = iter(trg_train_loader)
    trg_valid_loader = DataLoader(
        trg_dataset_splitted_by_run.get('1test'), 
        batch_size=args.batch_size
    )

    # Prepare CLUDA_NN
    embedding_dim = 40
    cluda_nn = CLUDA_NN(input_window_samples, n_chans, embedding_dim, args.n_classes, 0)

    # Prepare augmenter
    augmenter = Augmenter(cutout_length=0, cutout_prob=0, dropout_prob=0, is_cuda=cuda)
    # Tweak augmentation parameters
    pass

    # Send to GPU
    if cuda:
        set_random_seeds(seed=seed, cuda=cuda)
        cluda_nn.cuda()
        # augmenter.cuda()

    # Prepare optimizer
    optimizer = torch.optim.Adam(
        cluda_nn.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Prepare loss functions
    loss_fn = torch.nn.NLLLoss()
    criterion_CL = torch.nn.CrossEntropyLoss()

    # Mask, parameter for augmenter
    sequence_mask = torch.ones([n_chans, input_window_samples])
    sequence_mask = sequence_mask.to(device)

    train_accuracy_lst = []
    src_valid_accuracy_lst = []
    trg_valid_accuracy_lst = []
    src_contrastive_loss_lst = []
    trg_contrastive_loss_lst = []
    src_trg_contrastive_loss_lst = []
    domain_discrimination_loss_lst = []
    src_classification_loss_lst = []

    # Training loop
    for epoch in range(1, args.n_epochs + 1):

        cluda_nn.train()
        train_correct = 0
        batch_avg_loss_s = 0
        batch_avg_loss_t = 0
        batch_avg_loss_ts = 0
        batch_avg_disc_loss = 0
        batch_avg_cls_loss = 0

        # Train for one epoch: Iterate through one training batch from each dataset
        for batch_idx, (src_x, src_y, _) in enumerate(src_train_loader):

            try:
                trg_x, _, _ = next(trg_train_iter)
            except StopIteration:
                # re-initialize if done emitting data
                trg_train_loader = DataLoader(
                    trg_dataset_splitted_by_run.get('0train'), 
                    batch_size=args.batch_size, 
                    shuffle=True
                )
                trg_train_iter = iter(trg_train_loader)
                trg_x, _, _ = next(trg_train_iter)

            if len(src_x) != args.batch_size or len(trg_x) != args.batch_size:
                continue

            optimizer.zero_grad()
            src_x, src_y = src_x.to(device), src_y.to(device)
            trg_x = trg_x.to(device)

            # Augmentation
            # Queue and key sequences for source
            augmented_src_seq_q, _ = augmenter(src_x, sequence_mask)
            augmented_src_seq_k, _ = augmenter(src_x, sequence_mask)

            # Queue and key sequences for target
            augmented_trg_seq_q, _ = augmenter(trg_x, sequence_mask)
            augmented_trg_seq_k, _ = augmenter(trg_x, sequence_mask)

            # Foward pass
            # No idea what p and alpha are
            p = float(batch_idx) / 1000
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            (
                output_s, target_s, 
                output_t, target_t, 
                output_ts, target_ts, 
                output_disc, target_disc, 
                pred_s
            ) = cluda_nn(
                augmented_src_seq_q, 
                augmented_src_seq_k, 
                None, 
                augmented_trg_seq_q, 
                augmented_trg_seq_k, 
                None, 
                alpha
            )

            # Loss calculation
            loss_s = criterion_CL(output_s, target_s)
            batch_avg_loss_s = (batch_avg_loss_s * batch_idx + loss_s) / (batch_idx + 1)
            loss_t = criterion_CL(output_t, target_t)
            batch_avg_loss_t = (batch_avg_loss_t * batch_idx + loss_t) / (batch_idx + 1)
            loss_ts = criterion_CL(output_ts, target_ts)
            batch_avg_loss_ts = (batch_avg_loss_ts * batch_idx + loss_ts) / (batch_idx + 1)
            loss_disc = binary_cross_entropy(output_disc, target_disc)
            batch_avg_disc_loss = (batch_avg_disc_loss * batch_idx + loss_disc) / (batch_idx + 1)
            src_cls_loss = loss_fn(pred_s, src_y)
            batch_avg_cls_loss = (batch_avg_cls_loss * batch_idx + src_cls_loss) / (batch_idx + 1)
            train_correct += (pred_s.argmax(1) == src_y).sum().item()

            # Backprop
            total_loss = (
                0.1 * loss_s + 
                0.1 * loss_t + 
                0.2 * loss_ts + 
                1 * loss_disc + 
                1 * src_cls_loss
            )
            total_loss.backward()
            optimizer.step()

        # Calculate train accuracy
        train_accuracy = train_correct / len(src_train_loader.dataset)

        # Test model on source and target validation set
        cluda_nn.eval()

        src_valid_correct = 0
        with torch.no_grad():
            for _, (valid_x, valid_y, _) in enumerate(src_valid_loader):
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)
                valid_prediction = cluda_nn.predict(valid_x, None)
                src_valid_correct += (valid_prediction.argmax(1) == valid_y).sum().item()
        src_valid_accuracy = src_valid_correct / len(src_valid_loader.dataset)

        trg_valid_correct = 0
        with torch.no_grad():
            for _, (valid_x, valid_y, _) in enumerate(trg_valid_loader):
                valid_x, valid_y = valid_x.to(device), valid_y.to(device)
                valid_prediction = cluda_nn.predict(valid_x, None)
                trg_valid_correct += (valid_prediction.argmax(1) == valid_y).sum().item()
        trg_valid_accuracy = trg_valid_correct / len(trg_valid_loader.dataset)

        train_accuracy_lst.append(train_accuracy)
        src_valid_accuracy_lst.append(src_valid_accuracy)
        trg_valid_accuracy_lst.append(trg_valid_accuracy)
        
        src_contrastive_loss_lst.append(batch_avg_loss_s.cpu().item())
        trg_contrastive_loss_lst.append(batch_avg_loss_t.cpu().item())
        src_trg_contrastive_loss_lst.append(batch_avg_loss_ts.cpu().item())
        domain_discrimination_loss_lst.append(batch_avg_disc_loss.cpu().item())
        src_classification_loss_lst.append(batch_avg_cls_loss.cpu().item())

        print(
            f'[Epoch : {epoch}/{args.n_epochs}] ' 
            f'training accuracy = {100 * train_accuracy:.1f}% ' 
            f'source validation accuracy = {100 * src_valid_accuracy:.1f}% '
            f'target validation accuracy = {100 * trg_valid_accuracy:.1f}% '
        )

    # Plot loss (all components) and accuracy (source train, source valid, target valid) curves
    plt.figure()
    plt.plot(train_accuracy_lst, label='Train acc')
    plt.plot(src_valid_accuracy_lst, label='Valid. acc (source)')
    plt.plot(trg_valid_accuracy_lst, label='Valid. acc (target)')
    plt.legend()
    plt.xlabel('Training epochs')
    plt.ylabel('Accuracy')
    plt.title(acc_figure_title)
    plt.tight_layout()
    plt.savefig(acc_curve_path)
    plt.close()

    plt.figure()
    plt.plot(src_contrastive_loss_lst, label='Source contrastive loss')
    plt.plot(trg_contrastive_loss_lst, label='Target contrastive loss')
    plt.plot(src_trg_contrastive_loss_lst, label='Cross-domain contrastive loss')
    plt.plot(domain_discrimination_loss_lst, label='Domain disc. loss')
    plt.plot(src_classification_loss_lst, label='Source classification loss')
    plt.legend()
    plt.xlabel('Training epochs')
    plt.ylabel('Loss')
    plt.title(loss_figure_title)
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()

    # Save loss and accuracy data
    dict_train.update({
        dict_key: {
            'train_accuracy': train_accuracy_lst,
            'src_valid_accuracy': src_valid_accuracy_lst,
            'trg_valid_accuracy': trg_valid_accuracy_lst,
            'src_contrastive_loss': src_contrastive_loss_lst,
            'trg_contrastive_loss': trg_contrastive_loss_lst,
            'src_trg_contrastive_loss': src_trg_contrastive_loss_lst,
            'domain_discrimination_loss': domain_discrimination_loss_lst,
            'src_prediction_loss': src_classification_loss_lst
        }
    })
    if os.path.exists(train_file_path):
        os.remove(train_file_path)
    with open(train_file_path, 'wb') as f:
        pkl.dump(dict_train, f)

    # Save the trained model
    print('Save trained model')
    torch.save(deepcopy(cluda_nn.state_dict()), model_param_path)

    # Calculate embeddings
    print('Calculate and reduce embeddings to 2D')
    embedding_lst = []
    subject_id_lst = []
    label_lst = []
    for subject_id, subject_dataset in splitted_by_subj.items():

        subject_dataloader = DataLoader(subject_dataset, batch_size=args.batch_size)
        for _, (src_x, src_y, _) in enumerate(subject_dataloader):
            cluda_nn.eval()
            src_x = src_x.to(device)
            batch_embeddings = cluda_nn.get_encoding(src_x)
            for embedding, label in zip(
                batch_embeddings.detach().cpu().numpy(), 
                src_y.cpu().numpy()
            ):
                embedding_lst.append(embedding)
                label_lst.append(label)
                subject_id_lst.append(subject_id)

    df_embeddings = pd.DataFrame({
        'embedding': embedding_lst, 
        'subject_id': subject_id_lst, 
        'label': label_lst
    })

    # Dimensionality reduction
    tsne_model = TSNE(n_components=2, random_state=0, perplexity=10)
    reduced_embeddings = tsne_model.fit_transform(
        np.vstack(df_embeddings['embedding'].values)
    )
    df_embeddings['reduced_embedding'] = reduced_embeddings.tolist()

    print('Save embeddings')
    dict_embeddings.update({
        dict_key: df_embeddings
    })

    if os.path.exists(embedding_file_path):
        os.remove(embedding_file_path)
    with open(embedding_file_path, 'wb') as f:
        pkl.dump(dict_embeddings, f)

