"""
Hypernetwork-based BCI experiments.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from itertools import chain
from pytorch_warmup import UntunedLinearWarmup

from core.base_experiment import BaseExperiment
from core.utils import get_subset, train_one_epoch, test_model
from models.HypernetBCI import HyperBCINet
from models.Embedder import Conv1dEmbedder, ShallowFBCSPEmbedder, EEGConformerEmbedder
from models.Hypernet import LinearHypernet


class HypernetworkExperiment(BaseExperiment):
    """
    Base class for hypernetwork-based BCI experiments.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        super().__init__(config_path, config_dict)
        self.hypernet_model = None
        self.embedder = None
        self.hypernet = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for hypernetwork experiments."""
        config = super()._get_default_config()
        config.update({
            "data_amount_step": 40,
            "data_amount_unit": "min",
            "trial_len_sec": 4,
            "embedding_dim": 32,
            "embedding_length": 128,
            "embedder_type": "EEGConformer",  # or "ShallowFBCSP", "Conv1d"
        })
        return config
    
    def create_hypernet_model(self, sample_shape: Tuple[int, ...], **kwargs):
        """Create hypernetwork-based BCI model."""
        # Create primary network
        self.create_model(**kwargs)
        
        # Setup embedding configuration
        embedding_dim = self.config.get('embedding_dim', 32)
        embedding_length = self.config.get('embedding_length', 128)
        embedding_shape = (embedding_dim, embedding_length)
        n_classes = self.config.get('n_classes', 4)
        
        # Create embedder
        embedder_type = self.config.get('embedder_type', 'EEGConformer')
        if embedder_type == 'EEGConformer':
            self.embedder = EEGConformerEmbedder(sample_shape, embedding_shape, n_classes, self.sfreq)
        elif embedder_type == 'ShallowFBCSP':
            self.embedder = ShallowFBCSPEmbedder(sample_shape, embedding_shape, 'drop', n_classes)
        elif embedder_type == 'Conv1d':
            self.embedder = Conv1dEmbedder(sample_shape, embedding_shape)
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
            
        # Create hypernet
        weight_shape = self.model.final_layer.conv_classifier.weight.shape
        self.hypernet = LinearHypernet(embedding_shape, weight_shape)
        
        # Create HypernetBCI model
        self.hypernet_model = HyperBCINet(
            primary_net=self.model,
            embedder=self.embedder,
            embedding_shape=embedding_shape,
            sample_shape=sample_shape,
            hypernet=self.hypernet
        )
        
        if self.device == 'cuda':
            self.hypernet_model = self.hypernet_model.cuda()
            
        self.logger.info(f'Created HypernetBCI model with {embedder_type} embedder')
        return self.hypernet_model
    
    def pretrain_stage(self, pretrain_subjects: List[int], **kwargs) -> Dict[str, Any]:
        """
        Pretrain the hypernetwork model on a set of subjects.
        
        Args:
            pretrain_subjects: List of subject IDs for pretraining
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing pretraining results
        """
        self.logger.info(f"Starting pretraining on subjects: {pretrain_subjects}")
        
        # Prepare pretraining dataset
        pretrain_dataset = self._prepare_subject_dataset(pretrain_subjects)
        
        # Setup training parameters
        lr = self.config.get('lr', 0.0625 * 0.01)
        n_epochs = self.config.get('n_epochs', 40)
        batch_size = self.config.get('batch_size', 64)
        weight_decay = self.config.get('weight_decay', 0)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.hypernet_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup data loader
        train_loader = DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Training loop
        self.hypernet_model.train()
        pretrain_losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = train_one_epoch(
                self.hypernet_model,
                train_loader,
                optimizer,
                device=self.device
            )
            pretrain_losses.append(epoch_loss)
            
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"Pretrain Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        
        # Save the pretrained model
        pretrain_subjects_str = "_".join(map(str, pretrain_subjects))
        model_filename = f"hypernet_pretrained_subjects_{pretrain_subjects_str}.pth"
        self.save_model(self.hypernet_model, model_filename)
        
        self.logger.info("Pretraining completed")
        return {
            'pretrain_losses': pretrain_losses,
            'pretrain_subjects': pretrain_subjects,
            'model_path': str(self.results_dir / model_filename)
        }
    
    def calibration_stage(self, calibration_subject: int, data_amounts: List[int], **kwargs) -> Dict[str, Any]:
        """
        Calibrate the pretrained model for a specific subject with varying amounts of data.
        
        Args:
            calibration_subject: Subject ID for calibration
            data_amounts: List of data amounts to test
            **kwargs: Additional calibration parameters
            
        Returns:
            Dictionary containing calibration results
        """
        self.logger.info(f"Starting calibration for subject {calibration_subject}")
        
        # Prepare calibration and validation datasets
        calibration_dataset, validation_dataset = self._prepare_calibration_datasets(calibration_subject)
        
        results = {
            'subject': calibration_subject,
            'data_amounts': data_amounts,
            'accuracies': [],
            'std_devs': []
        }
        
        repetition = self.config.get('repetition', 3)
        
        for data_amount in data_amounts:
            self.logger.info(f"Calibrating with {data_amount} data amount")
            
            accuracies_for_amount = []
            
            for rep in range(repetition):
                # Get subset of calibration data
                calibration_subset = get_subset(
                    calibration_dataset,
                    data_amount,
                    random_sample=True
                )
                
                # Create fresh copy of pretrained model for calibration
                calibration_model = deepcopy(self.hypernet_model)
                calibration_model.calibrate()  # Set to calibration mode
                
                # Calibration data loader
                calibration_loader = DataLoader(
                    calibration_subset,
                    batch_size=self.config.get('batch_size', 64),
                    shuffle=False,
                    num_workers=0
                )
                
                # Perform calibration (forward pass through calibration data)
                calibration_model.eval()
                with torch.no_grad():
                    for batch_x, _ in calibration_loader:
                        if self.device == 'cuda':
                            batch_x = batch_x.cuda()
                        _ = calibration_model(batch_x)  # This generates and aggregates weights
                
                # Evaluate on validation set
                calibration_model.eval()
                validation_loader = DataLoader(
                    validation_dataset,
                    batch_size=self.config.get('batch_size', 64),
                    shuffle=False,
                    num_workers=0
                )
                
                accuracy = test_model(calibration_model, validation_loader, device=self.device)
                accuracies_for_amount.append(accuracy)
                
                self.logger.info(f"Rep {rep+1}/{repetition}, Accuracy: {accuracy:.4f}")
            
            results['accuracies'].append(accuracies_for_amount)
            results['std_devs'].append(np.std(accuracies_for_amount))
            
        self.logger.info(f"Calibration completed for subject {calibration_subject}")
        return results
    
    def _prepare_subject_dataset(self, subject_ids: List[int]):
        """Prepare dataset for specific subjects."""
        if not self.windows_dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")
            
        # For Schirrmeister2017, each subject has 2 sessions (datasets)
        # Subject i corresponds to datasets at indices [2*(i-1), 2*(i-1)+1]
        dataset_indices = []
        for subject_id in subject_ids:
            dataset_indices.extend([2*(subject_id-1), 2*(subject_id-1)+1])
        
        # Extract datasets for specified subjects
        subject_datasets = [self.windows_dataset.datasets[i] for i in dataset_indices]
        
        # Create combined dataset
        from braindecode.datasets import BaseConcatDataset
        return BaseConcatDataset(subject_datasets)
    
    def _prepare_calibration_datasets(self, subject_id: int):
        """Prepare calibration and validation datasets for a subject."""
        subject_dataset = self._prepare_subject_dataset([subject_id])
        
        # Split into calibration (80%) and validation (20%)
        total_trials = len(subject_dataset)
        calibration_size = int(0.8 * total_trials)
        
        # Create splits
        calibration_indices = list(range(calibration_size))
        validation_indices = list(range(calibration_size, total_trials))
        
        # Create subset datasets
        from braindecode.datasets.base import EEGWindowsDataset
        calibration_dataset = EEGWindowsDataset(
            subject_dataset.raw if hasattr(subject_dataset, 'raw') else subject_dataset.datasets[0].raw,
            subject_dataset.get_metadata().iloc[calibration_indices],
            description=f"Calibration data for subject {subject_id}"
        )
        
        validation_dataset = EEGWindowsDataset(
            subject_dataset.raw if hasattr(subject_dataset, 'raw') else subject_dataset.datasets[0].raw,
            subject_dataset.get_metadata().iloc[validation_indices],
            description=f"Validation data for subject {subject_id}"
        )
        
        return calibration_dataset, validation_dataset
    
    def plot_results(self, results: Dict[str, Any]):
        """Plot hypernetwork experiment results."""
        if 'all_subjects_results' not in results:
            self.logger.warning("No results to plot")
            return
            
        # Extract data for plotting
        all_results = results['all_subjects_results']
        data_amounts = all_results[0]['data_amounts'] if all_results else []
        
        # Calculate mean and std across subjects
        mean_accuracies = []
        std_accuracies = []
        
        for i, amount in enumerate(data_amounts):
            accuracies_for_amount = []
            for subject_result in all_results:
                accuracies_for_amount.extend(subject_result['accuracies'][i])
            
            mean_accuracies.append(np.mean(accuracies_for_amount))
            std_accuracies.append(np.std(accuracies_for_amount))
        
        # Convert data amounts to appropriate units
        data_amount_unit = self.config.get('data_amount_unit', 'min')
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
        plt.title(f'{self._get_experiment_name()} - Cross-Subject Results')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.results_dir / 'results_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Results plot saved to {plot_path}")


class CrossSubjectCalibrationExperiment(HypernetworkExperiment):
    """
    Cross-subject calibration experiment using hypernetworks.
    Hold out each subject and pretrain on others, then calibrate on the held-out subject.
    """
    
    def _get_experiment_name(self) -> str:
        model_name = self.config.get('model_name', 'ShallowFBCSP')
        dataset_name = self.config.get('dataset_name', 'Schirrmeister2017')
        version = self.config.get('experiment_version', 1)
        return f"HN_CrossSubject_{model_name}_{dataset_name}_{version}"
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the cross-subject calibration experiment."""
        # Load data
        subject_ids = list(range(1, 14))  # Schirrmeister2017 subjects
        self.load_data(subject_ids)
        
        # Get data dimensions
        sample_shape = self.windows_dataset[0][0].shape
        
        # Setup data amounts to test
        data_amount_step = self.config.get('data_amount_step', 40)
        max_data_amount = 200  # trials
        data_amounts = list(range(data_amount_step, max_data_amount + data_amount_step, data_amount_step))
        
        all_results = []
        
        # Cross-subject validation: hold out each subject
        for held_out_subject in subject_ids:
            self.logger.info(f"\n--- Processing held-out subject {held_out_subject} ---")
            
            # Create fresh model for this subject
            self.create_hypernet_model(sample_shape)
            
            # Pretrain on all other subjects
            pretrain_subjects = [s for s in subject_ids if s != held_out_subject]
            pretrain_results = self.pretrain_stage(pretrain_subjects)
            
            # Calibrate on held-out subject
            calibration_results = self.calibration_stage(held_out_subject, data_amounts)
            
            # Combine results
            subject_results = {**pretrain_results, **calibration_results}
            all_results.append(subject_results)
        
        # Aggregate final results
        final_results = {
            'all_subjects_results': all_results,
            'data_amounts': data_amounts,
            'config': self.config
        }
        
        return final_results


class SanityCheckExperiment(HypernetworkExperiment):
    """
    Sanity check experiment: train hypernetwork on single subject to verify it works.
    """
    
    def _get_experiment_name(self) -> str:
        model_name = self.config.get('model_name', 'ShallowFBCSP')
        version = self.config.get('experiment_version', 1)
        return f"HN_SanityCheck_{model_name}_{version}"
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the sanity check experiment."""
        # Load data for single subject
        subject_id = self.config.get('subject_id', 3)
        self.load_data([subject_id])
        
        # Get data dimensions
        sample_shape = self.windows_dataset[0][0].shape
        
        # Create models
        self.create_hypernet_model(sample_shape)
        
        # Train on the subject's data
        pretrain_results = self.pretrain_stage([subject_id])
        
        # Test different data amounts
        data_amounts = [20, 40, 80, 160]
        calibration_results = self.calibration_stage(subject_id, data_amounts)
        
        # Combine results
        final_results = {
            **pretrain_results,
            **calibration_results,
            'config': self.config
        }
        
        return final_results


class BaselineExperiment(HypernetworkExperiment):
    """
    Baseline experiment: train hypernetwork from scratch for each subject.
    """
    
    def _get_experiment_name(self) -> str:
        model_name = self.config.get('model_name', 'ShallowFBCSP')
        dataset_name = self.config.get('dataset_name', 'Schirrmeister2017')
        version = self.config.get('experiment_version', 1)
        return f"HN_Baseline_{model_name}_{dataset_name}_{version}"
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the baseline experiment."""
        # Load data
        subject_ids = list(range(1, 14))
        self.load_data(subject_ids)
        
        # Get data dimensions
        sample_shape = self.windows_dataset[0][0].shape
        
        # Setup data amounts to test
        data_amount_step = self.config.get('data_amount_step', 40)
        max_data_amount = 200
        data_amounts = list(range(data_amount_step, max_data_amount + data_amount_step, data_amount_step))
        
        all_results = []
        
        # Train from scratch for each subject
        for subject_id in subject_ids:
            self.logger.info(f"\n--- Processing subject {subject_id} ---")
            
            # Create fresh model for this subject
            self.create_hypernet_model(sample_shape)
            
            # "Pretrain" on the subject's own data (train from scratch)
            pretrain_results = self.pretrain_stage([subject_id])
            
            # Test different data amounts (using same data for consistency)
            calibration_results = self.calibration_stage(subject_id, data_amounts)
            
            # Combine results
            subject_results = {**pretrain_results, **calibration_results}
            all_results.append(subject_results)
        
        # Aggregate final results
        final_results = {
            'all_subjects_results': all_results,
            'data_amounts': data_amounts,
            'config': self.config
        }
        
        return final_results