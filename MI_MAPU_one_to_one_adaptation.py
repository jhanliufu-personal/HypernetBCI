import os
import numpy as np
from copy import deepcopy
import pickle as pkl
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
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
from baseline_MAPU.models import (
    masking, 
    myTemporal_Imputer, 
    ShallowFBCSPFeatureExtractor
)
from baseline_MAPU.loss import CrossEntropyLabelSmooth, EntropyLoss
from utils import parse_training_config, get_subset

import warnings
warnings.filterwarnings('ignore')

### ----------------------------- Experiment parameters -----------------------------
args = parse_training_config()
# subject_ids_lst = list(range(1, 14))
subject_ids_lst = [1, 2]
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
experiment_folder_name = f'MI_MAPU_one_to_one_adaptation_{args.experiment_version}'
temp_exp_name = 'MAPU_one_to_one_adapt'
# Create expriment folder
os.makedirs(os.path.join(dir_results, experiment_folder_name), exist_ok=True)

pretrain_file_name = f'{experiment_folder_name}_pretrain_acc'
results_file_name = f'{experiment_folder_name}_results'
pretrain_file_path = os.path.join(
    dir_results, 
    f'{experiment_folder_name}/', 
    f'{pretrain_file_name}.pkl'
)
results_file_path = os.path.join(
    dir_results, 
    f'{experiment_folder_name}/', 
    f'{results_file_name}.pkl'
)
print(f'Saving pretrain accuracy at {pretrain_file_path}')
print(f'Saving results at {results_file_path}')

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

dict_pretrain = {}
dict_results = {}

# Load existing outputs if they exist
if os.path.exists(pretrain_file_path):
    with open(pretrain_file_path, 'rb') as f:
        dict_pretrain = pkl.load(f)

if os.path.exists(results_file_path):
    with open(results_file_path, 'rb') as f:
        dict_results = pkl.load(f)

# adapt from one subject to another, not multi-source
for i, (source_subject, target_subject) in enumerate(args.scenarios):

    dict_key = f'from_{source_subject}to_{target_subject}'
    if dict_results.get(dict_key) is not None:
        continue

    print(f'Adapt model on source subject {source_subject} to target subject {target_subject}')
    ########################################################
    ###################### PRETRAINING #####################
    ########################################################

    # Prepare source dataset
    src_dataset = splitted_by_subj.get(f'{source_subject}')
    src_dataset_splitted_by_run = src_dataset.split('run')
    src_pretrain_dataset = src_dataset_splitted_by_run.get('0train')
    src_valid_dataset = src_dataset_splitted_by_run.get('1test')
    src_pretrain_loader = DataLoader(
        src_pretrain_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    src_valid_loader = DataLoader(
        src_valid_dataset, 
        batch_size=args.batch_size
    )

    # Prepare network, network = feature extractor + classifier
    # returns features and predictions
    network = ShallowFBCSPFeatureExtractor(
        torch.Size([n_chans, input_window_samples]), 
        'drop', 
        args.n_classes
    )

    if args.add_tov_loss:
        # Prepare the temporal imputer / verifier
        feature_dimension = 40
        temporal_verifier = myTemporal_Imputer(feature_dimension, feature_dimension)

    # Send to GPU
    if cuda:
        set_random_seeds(seed=seed, cuda=cuda)
        network.cuda()
        if args.add_tov_loss:
            temporal_verifier.cuda()

    # optimizes the network (actual feature extractor)
    pretrain_optimizer = torch.optim.Adam(
        network.parameters(),
        lr=args.pretrain_lr,
        weight_decay=args.weight_decay
    )
    if args.add_tov_loss:
        # Optimizes the temporal imputer
        tov_optimizer = torch.optim.Adam(
            temporal_verifier.parameters(),
            lr=args.imputer_lr,
            weight_decay=args.weight_decay
        )

    # Prepare pretrain loss functions
    mse_loss = torch.nn.MSELoss()
    cross_entropy = CrossEntropyLabelSmooth(args.n_classes, device, epsilon=0.1)

    # check if a source-trained model exists
    model_param_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{temp_exp_name}_{dict_key}_pretrained_model_params.pth'
    )
    if args.add_tov_loss:
        temporal_verifier_path = os.path.join(
            dir_results, 
            f'{experiment_folder_name}/',
            f'{temp_exp_name}_{dict_key}_pretrained_temporal_verifier_params.pth'
        )
    figure_title = f'{temp_exp_name}_{dict_key}_pretrain_acc_curve'
    pretrain_acc_curve_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{figure_title}.png'
    )
    model_exist = os.path.exists(model_param_path) and os.path.getsize(model_param_path) > 0
    if args.add_tov_loss:
        temporal_verifier_exist = os.path.exists(temporal_verifier_path) and os.path.getsize(temporal_verifier_path) > 0
    else:
        temporal_verifier_exist = True
    # Also check if the pretrain accuracy has been saved
    pretraining_done = model_exist and temporal_verifier_exist and (dict_pretrain.get(dict_key) is not None)

    if not pretraining_done:
        # Begin pretraining on source subject
        pretrain_train_acc_lst = []
        pretrain_test_acc_lst = []
        pretrain_tov_loss_lst = []
        pretrain_cls_loss_lst = []
        print(f'Pretraining on source subjects other than {target_subject}')
        for epoch in range(1, args.pretrain_n_epochs + 1):

            network.train()
            pretrain_correct = 0
            batch_avg_tov_loss = 0
            batch_avg_cls_loss = 0
            # Train for one epoch: Iterate through pretraining batches
            for batch_idx, (src_x, src_y, _) in enumerate(src_pretrain_loader):

                pretrain_optimizer.zero_grad()
                if args.add_tov_loss:
                    tov_optimizer.zero_grad()
                src_x, src_y = src_x.to(device), src_y.to(device)

                src_features, src_prediction = network(src_x)
                pretrain_correct += (src_prediction.argmax(1) == src_y).sum().item()
                src_features = src_features.squeeze(-1)
                src_classification_loss = cross_entropy(src_prediction, src_y)
                batch_avg_cls_loss = (batch_avg_cls_loss * batch_idx + src_classification_loss) / (batch_idx + 1)

                if args.add_tov_loss:
                    masked_x, mask = masking(src_x, num_splits=10, num_masked=2)
                    # mask the signal
                    masked_features, masked_prediction = network(masked_x)
                    # extract features from masked signal
                    masked_features = masked_features.squeeze(-1)
                    # predict full features from masked features
                    tov_predictions = temporal_verifier(masked_features.detach())
                    # calculate difference btw the full features and predicted features
                    tov_loss = mse_loss(tov_predictions, src_features)
                    batch_avg_tov_loss = (batch_avg_tov_loss * batch_idx + tov_loss) / (batch_idx + 1)
                else:
                    tov_loss = 0
                    batch_avg_tov_loss = 0

                total_loss = src_classification_loss + tov_loss
                total_loss.backward()
                pretrain_optimizer.step()
                if args.add_tov_loss:
                    tov_optimizer.step()

            pretrain_accuracy = pretrain_correct / len(src_pretrain_loader.dataset)
            # Save pretrain accuracy
            pretrain_train_acc_lst.append(pretrain_accuracy)
            # Save batch-averaged tov loss
            pretrain_tov_loss_lst.append(batch_avg_tov_loss)
            # Save batch-averaged classification loss
            pretrain_cls_loss_lst.append(batch_avg_cls_loss)

            # Test model on validation set
            network.eval()
            valid_correct = 0
            with torch.no_grad():
                for _, (valid_x, valid_y, _) in enumerate(src_valid_loader):
                    valid_x, valid_y = valid_x.to(device), valid_y.to(device)
                    _, valid_prediction = network(valid_x)
                    valid_correct += (valid_prediction.argmax(1) == valid_y).sum().item()

            # Save validation accuracy
            valid_accuracy = valid_correct / len(src_valid_loader.dataset)
            pretrain_test_acc_lst.append(valid_accuracy)
            print(
                f'[Epoch : {epoch}/{args.pretrain_n_epochs}] ' 
                f'training accuracy = {100 * pretrain_accuracy:.1f}% ' 
                f'validation accuracy = {100 * valid_accuracy:.1f}% '
                f'tov_loss = {batch_avg_tov_loss: .3e} '
                f'classification_loss = {batch_avg_cls_loss: .3e} '
            )

        # Plot and save the pretraining accuracy curves
        plt.figure()
        plt.plot(pretrain_train_acc_lst, label='Training accuracy')
        plt.plot(pretrain_test_acc_lst, label='Validation accuracy')
        plt.legend()
        plt.xlabel('Training epochs')
        plt.ylabel('Accuracy')
        plt.title(figure_title)
        plt.tight_layout()
        plt.savefig(pretrain_acc_curve_path)
        plt.close()

        # Save pretraining accuracies
        dict_pretrain.update({
            dict_key: {
                'pretrain_test_acc': pretrain_test_acc_lst,
                'pretrain_train_acc': pretrain_train_acc_lst,
                'pretrain_tov_loss': pretrain_tov_loss_lst,
                'pretrain_cls_loss': pretrain_cls_loss_lst
            }
        })
        if os.path.exists(pretrain_file_path):
            os.remove(pretrain_file_path)
        with open(pretrain_file_path, 'wb') as f:
            pkl.dump(dict_pretrain, f)

        # Save the source pretrained model and temporal verifier
        src_only_model = deepcopy(network.state_dict())
        torch.save(src_only_model, model_param_path)
        if args.add_tov_loss:
            torch.save(temporal_verifier.state_dict(), temporal_verifier_path)

    else:
        print('Pretraining done, load pretrained model and temporal verifier')
        # Load trained model
        network.load_state_dict(torch.load(model_param_path))
        if args.add_tov_loss:
            # Load temporal verifier
            temporal_verifier.load_state_dict(torch.load(temporal_verifier_path))

    if args.only_pretrain:
        continue

    ########################################################
    ###################### ADAPTATION ######################
    ########################################################

    # prepare adaptation and test dataset from target subject
    target_dataset = splitted_by_subj.get(f'{target_subject}')
    target_dataset_splitted_by_run = target_dataset.split('run')
    target_adaptation_dataset = target_dataset_splitted_by_run.get('0train')
    target_test_dataset = target_dataset_splitted_by_run.get('1test')
    target_test_loader = DataLoader(
        target_test_dataset, 
        batch_size=args.batch_size
    )

    # Measure baseline accuracy before adaptation
    network.eval()
    baseline_test_correct = 0
    with torch.no_grad():
        for _, (test_x, test_y, _) in enumerate(target_test_loader):
            test_x, test_y = test_x.to(device), test_y.to(device)
            _, test_prediction = network(test_x)
            baseline_test_correct += (test_prediction.argmax(1) == test_y).sum().item()

    # Save baseline accuracy
    baseline_test_accuracy = baseline_test_correct / len(target_test_loader.dataset)
    dict_subj_results = {0: [baseline_test_accuracy,]}
    print(f'Before adaptation, the baseline accuracy is {baseline_test_accuracy*100:.1f}')

    '''
    recording best model; overall best accuracy is the best accuracy ever 
    achieved on a target subject across all data amount and runs
    '''
    overall_best_test_accuracy = 0
    overall_best_model = deepcopy(network.state_dict())

    adaptation_trials_num = len(target_adaptation_dataset)
    for adaptation_data_amount in np.arange(
        1, 
        (adaptation_trials_num // args.data_amount_step) + 1
    ) * args.data_amount_step:

        adaptation_test_acc_lst = []
        ### Since we're sampling randomly, repeat for 'repetition' times
        for i in range(args.repetition):

            ## Get current calibration samples
            cur_adaptation_subset = get_subset(
                target_adaptation_dataset, 
                int(adaptation_data_amount), 
                random_sample=True
            )
            cur_batch_size = args.batch_size if args.batch_size <= adaptation_data_amount else int(adaptation_data_amount // 2)
            cur_adaptation_subset_loader = DataLoader(
                target_test_dataset, 
                batch_size=cur_batch_size,
                shuffle=True
            )

            # Reload trained model
            network.load_state_dict(torch.load(model_param_path))
            # Reload temporal verifier
            temporal_verifier.load_state_dict(torch.load(temporal_verifier_path))

            # Freeze the classifier and temporal verifier
            # only feature extractor can be changed from this point on
            for k, v in network.named_parameters():
                if 'final_layer' in k:
                    v.requires_grad = False
            for _, v in temporal_verifier.named_parameters():
                v.requires_grad = False

            # Reset adaptation optimizer and lr scheduler
            adaptation_optimizer = torch.optim.Adam(
                network.parameters(),
                lr=args.adaptation_lr,
                weight_decay=args.weight_decay
            )
            lr_scheduler = StepLR(
                adaptation_optimizer, 
                step_size=args.adaptation_lr_step_size, 
                gamma=args.adaptation_lr_decay
            )

            print(
                f'Adapting to target subject {target_subject} ' 
                f'with {adaptation_data_amount} adaptation samples '
                f'(repetition {i+1})'
            )
            cur_run_best_accuracy = 0
            for epoch in range(1, args.adaptation_n_epochs + 1):
                # Adapt for one epoch
                network.train()
                # batch_avg_tov_loss = 0
                for batch_idx, (trg_x, _, _) in enumerate(cur_adaptation_subset_loader):

                    adaptation_optimizer.zero_grad()
                    trg_x = trg_x.to(device)

                    trg_features, trg_prediction = network(trg_x)
                    trg_features = trg_features.squeeze(-1)
                    # select evidential vs softmax probabilities
                    trg_prob = torch.nn.Softmax(dim=1)(network.logits)

                    # Entropy loss
                    trg_ent = args.ent_loss_wt * torch.mean(EntropyLoss(trg_prob))
                    # IM loss
                    trg_ent -= args.im * torch.sum(-trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                    # Calculate temporal consistency loss
                    masked_x, mask = masking(trg_x, num_splits=10, num_masked=2)
                    # mask the signal
                    masked_features, masked_prediction = network(masked_x)
                    # extract features from masked signal
                    masked_features = masked_features.squeeze(-1)
                    # predict full features from masked features
                    tov_predictions = temporal_verifier(masked_features.detach())
                    # calculate difference btw the full features and predicted features
                    tov_loss = mse_loss(tov_predictions, trg_features)

                    # Overall loss
                    loss = trg_ent + args.TOV_wt * tov_loss
                    loss.backward()
                    adaptation_optimizer.step()
                    lr_scheduler.step()

                # Test adapted model
                network.eval()
                test_correct = 0
                with torch.no_grad():
                    for _, (test_x, test_y, _) in enumerate(target_test_loader):
                        test_x, test_y = test_x.to(device), test_y.to(device)
                        _, test_prediction = network(test_x)
                        test_correct += (test_prediction.argmax(1) == test_y).sum().item()

                # Save test accuracy
                test_accuracy = test_correct / len(target_test_loader.dataset)
                print(
                    f'[Epoch : {epoch}/{args.adaptation_n_epochs}] ' 
                    f'test accuracy after adaptation = {100 * test_accuracy:.1f} '
                )

                if test_accuracy > cur_run_best_accuracy:
                    cur_run_best_accuracy = test_accuracy

                if test_accuracy > overall_best_test_accuracy:
                    best_model = deepcopy(network.state_dict())
                    overall_best_test_accuracy = test_accuracy
                    print(f'New overall best accuracy achieved: {overall_best_test_accuracy*100:.1f}')

            adaptation_test_acc_lst.append(cur_run_best_accuracy)

        dict_subj_results.update({
            adaptation_data_amount: adaptation_test_acc_lst
        })

    # Save best model for the target subject across the board
    best_model_path = os.path.join(
        dir_results, 
        f'{experiment_folder_name}/',
        f'{temp_exp_name}_{dict_key}_best_adapted_model_params.pth'
    )
    torch.save(best_model, best_model_path)

    # Save results
    dict_results.update({
        dict_key: dict_subj_results
    })

    # Save results to pickle file
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
    with open(results_file_path, 'wb') as f:
        pkl.dump(dict_results, f)
