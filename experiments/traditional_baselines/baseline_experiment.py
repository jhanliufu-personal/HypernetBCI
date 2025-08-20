"""
Traditional baseline experiments using standard training approaches.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from braindecode import EEGClassifier
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from core.base_experiment import BaseExperiment
from core.utils import get_subset, test_model


class BaselineExperiment(BaseExperiment):
    """
    Base class for traditional baseline experiments.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        super().__init__(config_path, config_dict)
        self.classifier = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for baseline experiments."""
        config = super()._get_default_config()
        config.update({
            "data_amount_step": 20,
            "data_amount_unit": "trial",  # or "min", "sec"
            "trial_len_sec": 4,
            "max_data_amount": 200,
            "use_braindecode_classifier": True,
        })
        return config
    
    def create_braindecode_classifier(self, **kwargs):
        """Create EEGClassifier using braindecode."""
        model_name = self.config.get('model_name', 'ShallowFBCSPNet')
        model_object = import_model(model_name)
        
        n_chans = self.windows_dataset[0][0].shape[0] if self.windows_dataset else 22
        n_classes = self.config.get('n_classes', 4)
        input_window_samples = self.windows_dataset[0][0].shape[1] if self.windows_dataset else int(4 * 250)
        
        lr = self.config.get('lr', 0.0625 * 0.01)
        weight_decay = self.config.get('weight_decay', 0)
        batch_size = self.config.get('batch_size', 64)
        n_epochs = self.config.get('n_epochs', 40)
        
        model_kwargs = self.config.get('model_kwargs', {})
        model_kwargs.update(kwargs)
        
        self.classifier = EEGClassifier(
            model_object,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=n_epochs,
            train_split=predefined_split(None),  # Use all data for training, validation set provided separately
            callbacks=[
                ('lr_scheduler', LRScheduler(
                    'CosineAnnealingLR',
                    T_max=n_epochs - 1,
                ))
            ],
            device='cuda' if self.device == 'cuda' else 'cpu',
            module__n_chans=n_chans,
            module__n_classes=n_classes,
            module__input_window_samples=input_window_samples,
            **{f'module__{k}': v for k, v in model_kwargs.items()}
        )
        
        self.logger.info(f'Created braindecode EEGClassifier with {model_name}')
        return self.classifier
    
    def train_on_subject_data(self, subject_dataset, validation_dataset=None):
        """Train the classifier on subject data."""
        if self.config.get('use_braindecode_classifier', True):
            # Use braindecode training
            if validation_dataset is not None:
                self.classifier.fit(subject_dataset, y=None, valid_ds=validation_dataset)
            else:
                self.classifier.fit(subject_dataset, y=None)
        else:
            # Use custom training loop
            self._custom_training_loop(subject_dataset, validation_dataset)
    
    def _custom_training_loop(self, train_dataset, valid_dataset=None):
        """Custom training loop for more control."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 64),
            shuffle=True,
            num_workers=0
        )
        
        if valid_dataset:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.config.get('batch_size', 64),
                shuffle=False,
                num_workers=0
            )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('lr', 0.0625 * 0.01),
            weight_decay=self.config.get('weight_decay', 0)
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        n_epochs = self.config.get('n_epochs', 40)
        
        self.model.train()
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0
            
            for batch_x, batch_y in train_loader:
                if self.device == 'cuda':
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
                self.logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
                
                # Validation
                if valid_dataset:
                    val_acc = self.evaluate_on_dataset(valid_dataset)
                    self.logger.info(f"Validation Accuracy: {val_acc:.4f}")
    
    def evaluate_on_dataset(self, dataset):
        """Evaluate classifier on a dataset."""
        if self.config.get('use_braindecode_classifier', True):
            # Use braindecode evaluation
            return self.classifier.score(dataset)
        else:
            # Use custom evaluation
            test_loader = DataLoader(
                dataset,
                batch_size=self.config.get('batch_size', 64),
                shuffle=False,
                num_workers=0
            )
            return test_model(self.model, test_loader, device=self.device)
    
    def _prepare_subject_dataset(self, subject_ids: List[int]):
        """Prepare dataset for specific subjects."""
        if not self.windows_dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")
            
        # For Schirrmeister2017, each subject has 2 sessions
        dataset_indices = []
        for subject_id in subject_ids:
            dataset_indices.extend([2*(subject_id-1), 2*(subject_id-1)+1])
        
        subject_datasets = [self.windows_dataset.datasets[i] for i in dataset_indices]
        
        from braindecode.datasets import BaseConcatDataset
        return BaseConcatDataset(subject_datasets)
    
    def plot_results(self, results: Dict[str, Any]):
        """Plot baseline experiment results."""
        if 'all_subjects_results' not in results:
            self.logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        all_results = results['all_subjects_results']
        data_amounts = results.get('data_amounts', [])
        
        if not data_amounts:
            return
        
        # Calculate mean and std across subjects
        mean_accuracies = []
        std_accuracies = []
        
        for i, amount in enumerate(data_amounts):
            accuracies_for_amount = []
            for subject_result in all_results:
                if i < len(subject_result['accuracies']):
                    accuracies_for_amount.extend(subject_result['accuracies'][i])
            
            if accuracies_for_amount:
                mean_accuracies.append(np.mean(accuracies_for_amount))
                std_accuracies.append(np.std(accuracies_for_amount))
            else:
                mean_accuracies.append(0)
                std_accuracies.append(0)
        
        # Convert data amounts to appropriate units
        data_amount_unit = self.config.get('data_amount_unit', 'trial')
        trial_len_sec = self.config.get('trial_len_sec', 4)
        
        if data_amount_unit == 'min':
            unit_multiplier = trial_len_sec / 60
        elif data_amount_unit == 'sec':
            unit_multiplier = trial_len_sec
        else:
            unit_multiplier = 1
            
        x_values = [amount * unit_multiplier for amount in data_amounts]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(x_values, mean_accuracies, yerr=std_accuracies, 
                    marker='o', capsize=5, capthick=2, linewidth=2)
        plt.xlabel(f'Training Data Amount ({data_amount_unit})')
        plt.ylabel('Classification Accuracy')
        plt.title(f'{self._get_experiment_name()} - Results')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.results_dir / 'results_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Results plot saved to {plot_path}")


class FromScratchExperiment(BaselineExperiment):
    """
    Baseline 1: Train from scratch for each subject.
    Model is trained from scratch for each subject with varying amounts of data.
    """
    
    def _get_experiment_name(self) -> str:
        model_name = self.config.get('model_name', 'ShallowFBCSP')
        dataset_name = self.config.get('dataset_name', 'Schirrmeister2017')
        version = self.config.get('experiment_version', 1)
        return f"FromScratch_{model_name}_{dataset_name}_{version}"
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the from-scratch training experiment."""
        # Load data
        subject_ids = list(range(1, 14))
        self.load_data(subject_ids)
        
        # Setup data amounts to test
        data_amount_step = self.config.get('data_amount_step', 20)
        max_data_amount = self.config.get('max_data_amount', 200)
        data_amounts = list(range(data_amount_step, max_data_amount + data_amount_step, data_amount_step))
        
        all_results = []
        repetition = self.config.get('repetition', 10)
        
        # Train from scratch for each subject
        for subject_id in subject_ids:
            self.logger.info(f"\n--- Processing subject {subject_id} ---")
            
            # Prepare subject dataset
            subject_dataset = self._prepare_subject_dataset([subject_id])
            
            # Split into train and validation (80/20)
            total_trials = len(subject_dataset)
            train_size = int(0.8 * total_trials)
            
            # Create indices
            indices = np.random.permutation(total_trials)
            train_indices = indices[:train_size]
            valid_indices = indices[train_size:]
            
            # Create validation set
            from braindecode.datasets.base import EEGWindowsDataset
            validation_dataset = EEGWindowsDataset(
                subject_dataset.raw if hasattr(subject_dataset, 'raw') else subject_dataset.datasets[0].raw,
                subject_dataset.get_metadata().iloc[valid_indices],
                description=f"Validation data for subject {subject_id}"
            )
            
            subject_results = {
                'subject': subject_id,
                'data_amounts': data_amounts,
                'accuracies': []
            }
            
            for data_amount in data_amounts:
                self.logger.info(f"Training with {data_amount} trials")
                
                accuracies_for_amount = []
                
                for rep in range(repetition):
                    # Create fresh classifier for each repetition
                    if self.config.get('use_braindecode_classifier', True):
                        self.create_braindecode_classifier()
                    else:
                        self.create_model()
                    
                    # Get training subset
                    available_train_indices = train_indices[:min(data_amount, len(train_indices))]
                    
                    train_subset = EEGWindowsDataset(
                        subject_dataset.raw if hasattr(subject_dataset, 'raw') else subject_dataset.datasets[0].raw,
                        subject_dataset.get_metadata().iloc[available_train_indices],
                        description=f"Training data for subject {subject_id}"
                    )
                    
                    # Train the model
                    self.train_on_subject_data(train_subset, validation_dataset)
                    
                    # Evaluate on validation set
                    accuracy = self.evaluate_on_dataset(validation_dataset)
                    accuracies_for_amount.append(accuracy)
                    
                    self.logger.info(f"Rep {rep+1}/{repetition}, Accuracy: {accuracy:.4f}")
                
                subject_results['accuracies'].append(accuracies_for_amount)
            
            all_results.append(subject_results)
        
        # Aggregate final results
        final_results = {
            'all_subjects_results': all_results,
            'data_amounts': data_amounts,
            'config': self.config
        }
        
        return final_results


class TransferLearningExperiment(BaselineExperiment):
    """
    Baseline 2: Transfer learning approach.
    Take each person as the hold out / new person and pretrain the BCI model on everyone 
    else together as a big pretrain pool. No distinction between people in the pretrain model.
    """
    
    def _get_experiment_name(self) -> str:
        model_name = self.config.get('model_name', 'ShallowFBCSP')
        dataset_name = self.config.get('dataset_name', 'Schirrmeister2017')
        version = self.config.get('experiment_version', 1)
        return f"TransferLearning_{model_name}_{dataset_name}_{version}"
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the transfer learning experiment."""
        # Load data
        subject_ids = list(range(1, 14))
        self.load_data(subject_ids)
        
        # Setup data amounts to test
        data_amount_step = self.config.get('data_amount_step', 20)
        max_data_amount = self.config.get('max_data_amount', 200)
        data_amounts = list(range(data_amount_step, max_data_amount + data_amount_step, data_amount_step))
        
        all_results = []
        repetition = self.config.get('repetition', 3)
        
        # Hold out each subject for testing
        for held_out_subject in subject_ids:
            self.logger.info(f"\n--- Processing held-out subject {held_out_subject} ---")
            
            # Prepare pretrain dataset (all subjects except held-out)
            pretrain_subjects = [s for s in subject_ids if s != held_out_subject]
            pretrain_dataset = self._prepare_subject_dataset(pretrain_subjects)
            
            # Prepare target subject dataset
            target_dataset = self._prepare_subject_dataset([held_out_subject])
            
            # Split target dataset into fine-tuning and validation
            total_trials = len(target_dataset)
            finetune_size = int(0.8 * total_trials)
            
            indices = np.random.permutation(total_trials)
            finetune_indices = indices[:finetune_size]
            valid_indices = indices[finetune_size:]
            
            from braindecode.datasets.base import EEGWindowsDataset
            validation_dataset = EEGWindowsDataset(
                target_dataset.raw if hasattr(target_dataset, 'raw') else target_dataset.datasets[0].raw,
                target_dataset.get_metadata().iloc[valid_indices],
                description=f"Validation data for subject {held_out_subject}"
            )
            
            subject_results = {
                'subject': held_out_subject,
                'data_amounts': data_amounts,
                'accuracies': []
            }
            
            for data_amount in data_amounts:
                self.logger.info(f"Fine-tuning with {data_amount} trials")
                
                accuracies_for_amount = []
                
                for rep in range(repetition):
                    # Create and pretrain model
                    if self.config.get('use_braindecode_classifier', True):
                        self.create_braindecode_classifier()
                    else:
                        self.create_model()
                    
                    # Pretrain on other subjects
                    self.logger.info("Pretraining on other subjects...")
                    self.train_on_subject_data(pretrain_dataset)
                    
                    # Fine-tune on target subject data
                    available_finetune_indices = finetune_indices[:min(data_amount, len(finetune_indices))]
                    
                    finetune_subset = EEGWindowsDataset(
                        target_dataset.raw if hasattr(target_dataset, 'raw') else target_dataset.datasets[0].raw,
                        target_dataset.get_metadata().iloc[available_finetune_indices],
                        description=f"Fine-tuning data for subject {held_out_subject}"
                    )
                    
                    # Reduce learning rate for fine-tuning
                    if hasattr(self, 'classifier') and self.classifier is not None:
                        # For braindecode classifier, adjust learning rate
                        original_lr = self.classifier.optimizer__lr
                        self.classifier.set_params(optimizer__lr=original_lr * 0.1)
                    
                    self.logger.info("Fine-tuning on target subject...")
                    self.train_on_subject_data(finetune_subset, validation_dataset)
                    
                    # Evaluate on validation set
                    accuracy = self.evaluate_on_dataset(validation_dataset)
                    accuracies_for_amount.append(accuracy)
                    
                    self.logger.info(f"Rep {rep+1}/{repetition}, Accuracy: {accuracy:.4f}")
                
                subject_results['accuracies'].append(accuracies_for_amount)
            
            all_results.append(subject_results)
        
        # Aggregate final results
        final_results = {
            'all_subjects_results': all_results,
            'data_amounts': data_amounts,
            'config': self.config
        }
        
        return final_results