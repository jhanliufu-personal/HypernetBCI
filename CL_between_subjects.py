import os
import torch
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
from copy import deepcopy
from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader
from baseline_CLUDA.CLUDA_models import ShallowFBCSPEncoder
import matplotlib.pyplot as plt
from utils import test_model
from loss import contrastive_loss_btw_subject

# subject_ids_lst = list(range(1, 14))
subject_ids_lst = [1, 2]
preprocessed_dir = 'data/Schirrmeister2017_preprocessed'

# Hyperparameters
n_classes = 4
batch_size = 32
lr = 6.5e-4
weight_decay = 0
n_epochs = 30
temperature = 0.5
emb_loss_weight = 0.5
pred_loss_weight = 0.5

gpu_number = 0
experiment_version = 1

dir_results = 'results/'
experiment_folder_name = f'CL_between_subjects_{experiment_version}'
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)
training_record_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'training.pkl')
acc_curve_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'accuracy_curve.png')
model_param_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'model_params.pth')
embeddings_path = os.path.join(dir_results, f'{experiment_folder_name}/', 'embeddings.pkl')

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

training_done = os.path.exists(acc_curve_path) and os.path.exists(model_param_path)
if not training_done:

    # Create datasets
    splitted_by_run = windows_dataset.split('run')
    test_loader = DataLoader(
        splitted_by_run.get('1test'), 
        batch_size=batch_size, 
        shuffle=True
    )

    train_dataset = splitted_by_run.get('0train')
    dataset_splitted_by_subject = train_dataset.split('subject')
    subject_count = len(dataset_splitted_by_subject)
    assert not batch_size % subject_count, "Get same number of samples from each person"
    subject_batch_size = batch_size // subject_count

    dict_subject_loader = {}
    # Prepare dataloader iterator for each subject
    for k, v in dataset_splitted_by_subject.items():
        dict_subject_loader.update({
            k: iter(DataLoader(v, batch_size=subject_batch_size, shuffle=True))
        })
    # Approximate number of batches in each epoch
    n_batches = len(train_dataset) // batch_size

    # Prepare dataloader iterator for each subject
    for k, v in dataset_splitted_by_subject.items():
        dict_subject_loader.update({
            k: iter(DataLoader(v, batch_size=subject_batch_size, shuffle=True))
        })

    # Prepare encoder
    encoder = ShallowFBCSPEncoder(
        torch.Size([n_chans, input_window_samples]),
        'drop',
        n_classes
    )
    if cuda:
        encoder.cuda()

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=lr, 
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs - 1
    )
    pred_loss_fn = torch.nn.NLLLoss()
    emb_loss_fn = contrastive_loss_btw_subject(
        subject_count, 
        subject_batch_size, 
        batch_size,
        temperature=temperature,
        device=device
    )

    ######################################################
    ################### Training loop ####################
    ######################################################
    train_acc_lst, test_acc_lst = [], []
    overall_loss_lst, prediction_loss_lst, embedding_loss_lst = [], [], []
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}: ", end="")

        encoder.train()
        overall_loss, prediction_loss, embedding_loss, correct, train_sample_cnt = 0, 0, 0, 0, 0
        # Assemble batch data from individual subject loaders
        for batch_idx in range(n_batches):
            batch_x = []
            batch_y = []

            # Get samples from each person
            for subject_id, subject_loader in dict_subject_loader.items():
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
                    dict_subject_loader.update({subject_id: subject_loader})
                    cur_x, cur_y, _ = next(subject_loader)

                batch_x.append(cur_x)
                batch_y.append(cur_y)

            batch_x, batch_y = torch.cat(batch_x, dim=0).to(device), torch.cat(batch_y, dim=0).to(device)
            assert batch_x.size(0) == batch_size, "Overall batch size is incorrect"
            train_sample_cnt += batch_size

            optimizer.zero_grad()
            # Feed through encoder
            embeddings = encoder(batch_x)
            # Get embedding at the last timestamp
            embeddings = embeddings.squeeze(-1)[:,:,-1]

            # Inter-subject contrastive loss
            emb_loss = emb_loss_fn(embeddings)
            # Classification loss
            pred_loss = pred_loss_fn(encoder.predictions, batch_y)
            # total loss
            total_loss = emb_loss_weight * emb_loss + pred_loss_weight * pred_loss

            total_loss.backward()
            optimizer.step()  
            optimizer.zero_grad()

            overall_loss += total_loss.item()
            prediction_loss += pred_loss.item()
            embedding_loss += emb_loss.item()
            correct += (encoder.predictions.argmax(1) == batch_y).sum().item()

        # Calculate train accuracy
        overall_loss /=  train_sample_cnt
        prediction_loss /=  train_sample_cnt
        embedding_loss /=  train_sample_cnt
        train_accuracy = correct / train_sample_cnt

        test_loss, test_accuracy = test_model(
            test_loader, 
            encoder, 
            pred_loss_fn,
            print_batch_stats=False
        )
        print(
            f"Train Accuracy: {100 * train_accuracy:.2f}%, "
            # f"Average Train Loss: {overall_loss:.6f}, "
            f"Average Prediction Loss: {prediction_loss:.6f}, "
            f"Average Contrastive Loss: {embedding_loss:.6f}"
            f"Test Accuracy: {100 * test_accuracy:.1f}%, "
            f"Average Test Loss: {test_loss:.6f}\n"
        )

        train_acc_lst.append(train_accuracy)
        test_acc_lst.append(test_accuracy)
        overall_loss_lst.append(overall_loss)
        prediction_loss_lst.append(prediction_loss)
        embedding_loss_lst.append(embedding_loss)

    # Plot and save the training curve
    plt.figure()
    plt.plot(train_acc_lst, label='Training accuracy')
    plt.plot(test_acc_lst, label='Test accuracy')
    plt.legend()
    plt.xlabel('Training epochs')
    plt.ylabel('Accuracy')
    # plt.title(f'Train and test accuracy with all subjects')
    plt.savefig(acc_curve_path)

    # Save accuracy and loss
    dict_training = {
        'overall_loss': overall_loss_lst,
        'prediction_loss': prediction_loss_lst,
        'contrastive_loss': embedding_loss_lst,
        'train_accuracy': train_acc_lst,
        'test_accuracy': test_acc_lst
    }
    with open(training_record_path, 'wb') as f:
        pkl.dump(dict_training, f)

    # Save the trained model
    print('Save trained model')
    torch.save(deepcopy(encoder.state_dict()), model_param_path)
else:
    print('Training done')

# Create encoder and load trained model weights
encoder = ShallowFBCSPEncoder(
    torch.Size([n_chans, input_window_samples]),
    'drop',
    n_classes
)
encoder.model.load_state_dict(torch.load(model_param_path))
if cuda:
    encoder.cuda()

# Calculate embeddings
print('Calculate and reduce embeddings to 2D')
embedding_lst = []
subject_id_lst = []
label_lst = []
splitted_by_subj = windows_dataset.split('subject')
for subject_id, subject_dataset in splitted_by_subj.items():

    subject_dataloader = DataLoader(subject_dataset, batch_size=batch_size)
    for _, (src_x, src_y, _) in enumerate(subject_dataloader):
        encoder.eval()
        src_x = src_x.to(device)
        batch_embeddings = encoder(src_x)
        for embedding, label in zip(
            batch_embeddings.detach().cpu().numpy(), 
            src_y.cpu().numpy()
        ):
            embedding_lst.append(embedding.flatten())
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
df_embeddings.to_pickle(embeddings_path)


