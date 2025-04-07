'''
Test-time adaptation using labeled support set. Calculate prototypical support
embedding for each label class from the labeled support set, and let task embeddings
attend over these class prototypes.
'''

import os
import torch
# import pandas as pd
# import numpy as np
import pickle as pkl
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from copy import deepcopy
from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader
from braindecode.datasets import BaseConcatDataset

from models.Embedder import ShallowFBCSPEncoder
from models.Supportnet import Supportnet
from utils import (
    freeze_all_param_but, train_one_epoch_episodic, test_model, load_from_pickle
    )
from loss import contrastive_loss_btw_subject

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
temperature = 0.5

gpu_number = '1'
experiment_version = 1

dir_results = 'results/'
experiment_folder_name = f'class_prototype_attention_{experiment_version}'
print(experiment_folder_name)
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)
training_record_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'training.pkl')
# embeddings_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'embeddings.pkl')
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
# Exclude target subject
src_subject_count = len(dataset_splitted_by_subject) - 1
assert not batch_size % src_subject_count, "Get same number of samples from each person"
subject_batch_size = batch_size // src_subject_count

# dict_embeddings = load_from_pickle(embeddings_path)
dict_training = load_from_pickle(training_record_path)
dict_results = load_from_pickle(results_path)

# --------------------------------------------------------------------------
# ------------------------- Leave-one-out loop -----------------------------
# --------------------------------------------------------------------------

for i, target_subject in enumerate(subject_ids_lst):

    dict_key = f'adapt_to_{target_subject}'

    # support_net_folder = 'CL_between_subjects_6'
    support_encoder_path = os.path.join(
        dir_results, 
        experiment_folder_name,
        # f'{experiment_folder_name}/', 
        # support_net_folder,
        f'{dict_key}_support_encoder.pth'
    )
    supportnet_path = os.path.join(
        dir_results, 
        experiment_folder_name,
        # f'{experiment_folder_name}/', 
        # support_net_folder,
        f'{dict_key}_supportnet.pth'
    )
    print(f'Adapt model from multiple sources to target subject {target_subject}')

    # Create training (subject specific) and validation dataloaders
    dict_src_subject_loader = {}
    src_train_set_lst = []
    src_valid_set_lst = []
    training_sample_cnt = 0
    for k, v in dataset_splitted_by_subject.items():

        if k == f'{target_subject}':
            print(f'Excluding data from target subject {target_subject}')
            continue

        subject_splitted_by_run = v.split('run')
        subject_train_set = subject_splitted_by_run.get('0train')
        training_sample_cnt += len(subject_train_set)
        subject_test_set = subject_splitted_by_run.get('1test')
        # Subject train loader
        dict_src_subject_loader.update({
            k: iter(
                DataLoader(
                    subject_train_set, 
                    batch_size=subject_batch_size, 
                    shuffle=True
                )
            )
        })
        # Train set
        src_train_set_lst.append(subject_train_set)
        # Valid set
        src_valid_set_lst.append(subject_test_set)

    # Approximate number of batches in each epoch
    n_batches = training_sample_cnt // batch_size
    print(f'{n_batches} batches per epoch')

# --------------------------------------------------------------------------
# ------------------------- Train support encoder --------------------------
# --------------------------------------------------------------------------

    # Load support encoder
    support_encoder = ShallowFBCSPEncoder(
        torch.Size([n_chans, input_window_samples]),
        'drop',
        n_classes
    )
    # support_encoder.model.load_state_dict(torch.load(support_encoder_path))
    # # Freeze support encoder
    # freeze_all_param_but(support_encoder.model, [])
    if cuda:
        support_encoder.cuda()

    if os.path.exists(support_encoder_path) and os.path.getsize(support_encoder_path) > 0:
        print(f'A support encoder trained without subject {target_subject} exists')
        support_encoder.model.load_state_dict(torch.load(support_encoder_path))
    else:
        support_optimizer = torch.optim.AdamW(
            support_encoder.parameters(),
            lr=lr, 
            weight_decay=weight_decay)
        support_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            support_optimizer,
            T_max=n_epochs - 1
        )
        emb_loss_fn = contrastive_loss_btw_subject(
            src_subject_count, 
            subject_batch_size, 
            batch_size,
            temperature=temperature,
            device=device
        )

        #############################################################
        ################### Contrastive learning ####################
        #############################################################
        embedding_loss_lst = []
        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}/{n_epochs}: ", end="")

            support_encoder.train()
            embedding_loss, train_sample_cnt = 0, 0
            # Assemble batch data from individual subject loaders
            for batch_idx in range(n_batches):
                batch_x = []
                batch_y = []

                # Get samples from each person
                for subject_id, subject_loader in dict_src_subject_loader.items():
                    try:
                        cur_x, cur_y, _ = next(subject_loader)
                        # Check if batch size is smaller than desired
                        if cur_x.size(0) < subject_batch_size:
                            raise StopIteration
                        
                    except StopIteration:
                        # Re-initialize subject-specific loader
                        subject_loader = iter(DataLoader(
                            dataset_splitted_by_subject.get(subject_id),
                            batch_size=subject_batch_size,
                            shuffle=True
                        ))
                        # This may not be allowed at runtime
                        dict_src_subject_loader.update({subject_id: subject_loader})
                        cur_x, cur_y, _ = next(subject_loader)

                    batch_x.append(cur_x)
                    batch_y.append(cur_y)

                batch_x, batch_y = torch.cat(batch_x, dim=0).to(device), torch.cat(batch_y, dim=0).to(device)
                assert batch_x.size(0) == batch_size, "Overall batch size is incorrect"
                train_sample_cnt += batch_size

                support_optimizer.zero_grad()
                # Feed through encoder
                _ = support_encoder(batch_x)
                embeddings = support_encoder.get_embeddings()
                
                # Get embedding at the last timestamp
                embeddings = embeddings.squeeze(-1)[:,:,-1]

                # Inter-subject contrastive loss
                emb_loss = emb_loss_fn(embeddings)
                emb_loss.backward()
                support_optimizer.step()  
                support_optimizer.zero_grad()
                embedding_loss += emb_loss.item()

            embedding_loss /=  train_sample_cnt
            print(f"Average Contrastive Loss: {embedding_loss:.6f}, ")
            embedding_loss_lst.append(embedding_loss)

        # Save the trained model
        print('Save trained support encoder')
        torch.save(deepcopy(support_encoder.model.state_dict()), support_encoder_path)

    # Freeze support encoder
    freeze_all_param_but(support_encoder.model, [])

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
            torch.nn.Conv2d(80, 4, kernel_size=(144, 1)),
            # torch.nn.Conv2d(40, 4, kernel_size=(144, 1)),
            torch.nn.LogSoftmax(dim=1)
        )
    )
    if cuda:
        supportnet.cuda()
    
    if os.path.exists(supportnet_path) and os.path.getsize(supportnet_path) > 0:
        print(f'A supportnet trained without subject {target_subject} exists')
        supportnet.load_state_dict(torch.load(supportnet_path))
    else:

        # Prepare source train and valid loaders
        src_train_loader = DataLoader(
            BaseConcatDataset(src_train_set_lst), 
            batch_size=batch_size
        )
        src_valid_loader = DataLoader(
            BaseConcatDataset(src_valid_set_lst), 
            batch_size=batch_size
        )

        pred_loss_fn = torch.nn.NLLLoss()
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

            train_loss, train_acc = train_one_epoch_episodic(
                src_train_loader, 
                supportnet,
                pred_loss_fn,
                optimizer,
                scheduler,
                epoch,
                device,
                num_classes=n_classes
            )

            valid_loss, valid_acc = test_model(
                src_valid_loader,
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
        target_loss, target_acc = test_model(
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

    # ########################################################
    # ###################### EMBEDDINGS ######################
    # ########################################################
    # if dict_embeddings.get(dict_key) is not None:
    #     print(f'Embeddings by this supportnet have been reduced and saved')
    #     # support_encoder.model.load_state_dict(torch.load(support_encoder_path))
    # else:
    #     # Calculate and visualize embeddings
    #     print('Calculate and reduce embeddings to 2D')
    #     embedding_lst = []
    #     subject_id_lst = []
    #     label_lst = []
    #     for subject_id, subject_dataset in windows_dataset.split('subject').items():

    #         subject_dataloader = DataLoader(subject_dataset, batch_size=batch_size)
    #         for _, (src_x, src_y, _) in enumerate(subject_dataloader):
    #             # support_encoder.eval()
    #             # src_x = src_x.to(device)
    #             # _ = support_encoder(src_x)
    #             # batch_embeddings = support_encoder.get_embeddings()
    #             supportnet.eval()
    #             src_x = src_x.to(device)
    #             _ = supportnet(src_x)
    #             integrated_embeddings = supportnet.integrated_embeddings
    #             # print(integrated_embeddings.shape)

    #             for embedding, label in zip(
    #                 # batch_embeddings.detach().cpu().numpy(), 
    #                 integrated_embeddings.detach().cpu().numpy(),
    #                 src_y.cpu().numpy()
    #             ):
    #                 embedding_lst.append(embedding.flatten())
    #                 label_lst.append(label)
    #                 subject_id_lst.append(subject_id)

    #     df_embeddings = pd.DataFrame({
    #         'embedding': embedding_lst, 
    #         'subject_id': subject_id_lst, 
    #         'label': label_lst
    #     })

    #     # Dimensionality reduction
    #     tsne_model = TSNE(n_components=2, random_state=0, perplexity=10)
    #     reduced_embeddings = tsne_model.fit_transform(
    #         np.vstack(df_embeddings['embedding'].values)
    #     )
    #     df_embeddings['reduced_embedding'] = reduced_embeddings.tolist()

    #     print('Save embeddings')
    #     dict_embeddings.update({
    #         dict_key: df_embeddings
    #     })
    #     with open(embeddings_path, 'wb') as f:
    #         pkl.dump(dict_embeddings, f)


