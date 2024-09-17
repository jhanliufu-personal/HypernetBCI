import os
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from copy import deepcopy
from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader
from braindecode.models import ShallowFBCSPNet
from baseline_CLUDA.CLUDA_models import ShallowFBCSPEncoder
import matplotlib.pyplot as plt
from utils import train_one_epoch, test_model

# subject_ids_lst = list(range(1, 14))
subject_ids_lst = [1, 2]
preprocessed_dir = 'data/Schirrmeister2017_preprocessed'

# Hyperparameters
n_classes = 4
batch_size = 72
lr = 6.5e-4
weight_decay = 0
n_epochs = 5
experiment_version = 1

dir_results = 'results/'
experiment_folder_name = f'Get_embeddings_{experiment_version}'
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)
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

training_done = os.path.exists(acc_curve_path) and os.path.exists(model_param_path)
if not training_done:

    # Create datasets
    splitted_by_run = windows_dataset.split('run')
    train_loader = DataLoader(
        splitted_by_run.get('0train'), 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        splitted_by_run.get('0train'), 
        batch_size=batch_size, 
        shuffle=True
    )

    # Prepare model
    cuda = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    if cuda:
        print(f'{device_count} CUDA devices available, use GPU for training')
        torch.backends.cudnn.benchmark = True
        device = 'cuda'
    else:
        print('No CUDA available, use CPU for training')
        device = 'cpu'

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length="auto"
    )
    if cuda:
        model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr, 
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs - 1
    )
    loss_fn = torch.nn.NLLLoss()

    # Training
    train_acc_lst = []
    test_acc_lst = []
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}: ", end="")

        train_loss, train_accuracy = train_one_epoch(
            train_loader, 
            model, 
            loss_fn, 
            optimizer, 
            scheduler, 
            epoch, 
            device,
            print_batch_stats=False
        )

        test_loss, test_accuracy = test_model(
            test_loader, 
            model, 
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

    # Plot and save the training curve
    plt.figure()
    plt.plot(train_acc_lst, label='Training accuracy')
    plt.plot(test_acc_lst, label='Test accuracy')
    plt.legend()
    plt.xlabel('Training epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Train and test accuracy with all subjects')
    plt.savefig(acc_curve_path)

    # Save the trained model
    print('Save trained model')
    torch.save(deepcopy(model.state_dict()), model_param_path)

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
        model.eval()
        src_x = src_x.to(device)
        batch_embeddings = encoder(src_x)
        for embedding, label in zip(
            batch_embeddings.detach().cpu().numpy(), 
            src_y.cpu().numpy()
        ):
            embedding_lst.append(embedding)
            label_lst.append(label)
            subject_id_lst.append(subject_id)

print(embedding_lst[0])

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


