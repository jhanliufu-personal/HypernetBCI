"""
Contrastive learning and class prototype attention experiments.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from core.base_experiment import BaseExperiment
from core.utils import freeze_all_param_but, train_one_epoch_episodic, test_model_episodic, load_from_pickle
from core.loss import contrastive_loss_btw_subject
from models.Embedder import ShallowFBCSPEncoder
from models.Supportnet import Supportnet


class ContrastiveLearningExperiment(BaseExperiment):
    """
    Base class for contrastive learning experiments.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        super().__init__(config_path, config_dict)
        self.encoder = None
        self.supportnet = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for contrastive learning experiments."""
        config = super()._get_default_config()
        config.update({
            "temperature": 0.5,
            "n_support_shots": 5,
            "n_query_shots": 15,
            "embedding_dim": 40,
            "use_attention": True,
            "freeze_encoder": True,
            "contrastive_weight": 1.0,
        })
        return config
    
    def create_encoder(self, **kwargs):
        """Create encoder model."""
        n_chans = self.windows_dataset[0][0].shape[0] if self.windows_dataset else 22
        n_classes = self.config.get('n_classes', 4)
        input_window_samples = self.windows_dataset[0][0].shape[1] if self.windows_dataset else int(4 * 250)
        
        self.encoder = ShallowFBCSPEncoder(
            n_chans,
            n_classes,
            input_window_samples=input_window_samples,
            **kwargs
        )
        
        if self.device == 'cuda':
            self.encoder = self.encoder.cuda()
            
        self.logger.info('Created ShallowFBCSP encoder')
        return self.encoder
    
    def create_supportnet(self, encoder_output_dim: int, **kwargs):
        """Create support network for prototype attention."""
        n_classes = self.config.get('n_classes', 4)
        embedding_dim = self.config.get('embedding_dim', 40)
        use_attention = self.config.get('use_attention', True)
        
        self.supportnet = Supportnet(
            encoder_output_dim=encoder_output_dim,
            n_classes=n_classes,
            embedding_dim=embedding_dim,
            use_attention=use_attention,
            **kwargs
        )
        
        if self.device == 'cuda':
            self.supportnet = self.supportnet.cuda()
            
        self.logger.info(f'Created Supportnet with {"attention" if use_attention else "no attention"}')
        return self.supportnet
    
    def contrastive_pretraining_stage(self, subjects: List[int], **kwargs) -> Dict[str, Any]:
        """
        Pretrain encoder using contrastive learning between subjects.
        
        Args:
            subjects: List of subject IDs for pretraining
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing pretraining results
        """
        self.logger.info(f"Starting contrastive pretraining on subjects: {subjects}")
        
        # Prepare datasets for each subject
        subject_datasets = []
        for subject_id in subjects:
            subject_dataset = self._prepare_subject_dataset([subject_id])
            subject_datasets.append(subject_dataset)
        
        # Setup training parameters
        lr = self.config.get('lr', 6.5e-4)
        n_epochs = self.config.get('n_epochs', 30)
        batch_size = self.config.get('batch_size', 72)
        weight_decay = self.config.get('weight_decay', 0)
        temperature = self.config.get('temperature', 0.5)
        
        # Setup optimizer (only train encoder during contrastive pretraining)
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Training loop
        self.encoder.train()
        contrastive_losses = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            # Sample batches from different subjects for contrastive learning
            for batch_idx in range(len(subject_datasets[0]) // batch_size):
                optimizer.zero_grad()
                
                batch_loss = 0
                batch_embeddings = []
                batch_labels = []
                
                # Get batch from each subject
                for subject_idx, dataset in enumerate(subject_datasets):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(dataset))
                    
                    if start_idx >= len(dataset):
                        continue
                        
                    # Get batch data
                    batch_data = []
                    batch_targets = []
                    for i in range(start_idx, end_idx):
                        x, y = dataset[i]
                        batch_data.append(x)
                        batch_targets.append(y)
                    
                    if not batch_data:
                        continue
                        
                    batch_x = torch.stack(batch_data)
                    batch_y = torch.tensor(batch_targets)
                    
                    if self.device == 'cuda':
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()
                    
                    # Get embeddings
                    embeddings = self.encoder(batch_x)
                    batch_embeddings.append(embeddings)
                    batch_labels.append(torch.full((embeddings.size(0),), subject_idx, device=self.device))
                
                if len(batch_embeddings) < 2:
                    continue
                
                # Compute contrastive loss between subjects
                combined_embeddings = torch.cat(batch_embeddings, dim=0)
                combined_labels = torch.cat(batch_labels, dim=0)
                
                contrastive_loss = contrastive_loss_btw_subject(
                    combined_embeddings, combined_labels, temperature
                )
                
                contrastive_loss.backward()
                optimizer.step()
                
                epoch_losses.append(contrastive_loss.item())
            
            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                contrastive_losses.append(avg_epoch_loss)
                
                if (epoch + 1) % 5 == 0:
                    self.logger.info(f"Contrastive Epoch {epoch + 1}/{n_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        self.logger.info("Contrastive pretraining completed")
        return {
            'contrastive_losses': contrastive_losses,
            'pretrain_subjects': subjects
        }
    
    def episodic_training_stage(self, subjects: List[int], **kwargs) -> Dict[str, Any]:
        """
        Train support network using episodic learning.
        
        Args:
            subjects: List of subject IDs for episodic training
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        self.logger.info(f"Starting episodic training on subjects: {subjects}")
        
        # Freeze encoder if specified
        if self.config.get('freeze_encoder', True):
            freeze_all_param_but(self.encoder, [])  # Freeze all encoder parameters
        
        # Setup training parameters
        lr = self.config.get('lr', 6.5e-4)
        n_epochs = self.config.get('n_epochs', 30)
        n_support_shots = self.config.get('n_support_shots', 5)
        n_query_shots = self.config.get('n_query_shots', 15)
        
        # Setup optimizer (train supportnet, optionally encoder)
        parameters = list(self.supportnet.parameters())
        if not self.config.get('freeze_encoder', True):
            parameters += list(self.encoder.parameters())
            
        optimizer = torch.optim.AdamW(parameters, lr=lr)
        
        # Prepare episodic datasets
        subject_datasets = []
        for subject_id in subjects:
            subject_dataset = self._prepare_subject_dataset([subject_id])
            subject_datasets.append(subject_dataset)
        
        # Training loop
        self.encoder.eval() if self.config.get('freeze_encoder', True) else self.encoder.train()
        self.supportnet.train()
        
        episodic_losses = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            # Episodic training
            for episode in range(len(subject_datasets[0]) // (n_support_shots + n_query_shots)):
                optimizer.zero_grad()
                
                # Sample episode from random subject
                subject_idx = np.random.choice(len(subjects))
                dataset = subject_datasets[subject_idx]
                
                # Create support and query sets
                support_data, support_labels, query_data, query_labels = self._create_episode(
                    dataset, n_support_shots, n_query_shots
                )
                
                if support_data is None:
                    continue
                
                # Forward pass
                loss = train_one_epoch_episodic(
                    encoder=self.encoder,
                    supportnet=self.supportnet,
                    support_data=support_data,
                    support_labels=support_labels,
                    query_data=query_data,
                    query_labels=query_labels,
                    optimizer=optimizer,
                    device=self.device
                )
                
                epoch_losses.append(loss)
            
            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                episodic_losses.append(avg_epoch_loss)
                
                if (epoch + 1) % 5 == 0:
                    self.logger.info(f"Episodic Epoch {epoch + 1}/{n_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        self.logger.info("Episodic training completed")
        return {
            'episodic_losses': episodic_losses,
            'training_subjects': subjects
        }
    
    def evaluation_stage(self, test_subjects: List[int], **kwargs) -> Dict[str, Any]:
        """
        Evaluate model on test subjects using episodic evaluation.
        
        Args:
            test_subjects: List of subject IDs for evaluation
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Starting evaluation on subjects: {test_subjects}")
        
        self.encoder.eval()
        self.supportnet.eval()
        
        results = {
            'subject_accuracies': {},
            'mean_accuracy': 0,
            'std_accuracy': 0
        }
        
        n_support_shots = self.config.get('n_support_shots', 5)
        n_query_shots = self.config.get('n_query_shots', 15)
        n_episodes = self.config.get('n_test_episodes', 100)
        
        all_accuracies = []
        
        for subject_id in test_subjects:
            self.logger.info(f"Evaluating subject {subject_id}")
            
            subject_dataset = self._prepare_subject_dataset([subject_id])
            subject_accuracies = []
            
            for episode in range(n_episodes):
                # Create test episode
                support_data, support_labels, query_data, query_labels = self._create_episode(
                    subject_dataset, n_support_shots, n_query_shots
                )
                
                if support_data is None:
                    continue
                
                # Evaluate episode
                accuracy = test_model_episodic(
                    encoder=self.encoder,
                    supportnet=self.supportnet,
                    support_data=support_data,
                    support_labels=support_labels,
                    query_data=query_data,
                    query_labels=query_labels,
                    device=self.device
                )
                
                subject_accuracies.append(accuracy)
            
            if subject_accuracies:
                subject_mean_acc = np.mean(subject_accuracies)
                results['subject_accuracies'][subject_id] = {
                    'mean': subject_mean_acc,
                    'std': np.std(subject_accuracies),
                    'accuracies': subject_accuracies
                }
                all_accuracies.extend(subject_accuracies)
                
                self.logger.info(f"Subject {subject_id} accuracy: {subject_mean_acc:.4f} ± {np.std(subject_accuracies):.4f}")
        
        # Overall statistics
        if all_accuracies:
            results['mean_accuracy'] = np.mean(all_accuracies)
            results['std_accuracy'] = np.std(all_accuracies)
            
        self.logger.info(f"Overall accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        return results
    
    def _create_episode(self, dataset, n_support: int, n_query: int):
        """Create an episode with support and query sets."""
        n_classes = self.config.get('n_classes', 4)
        
        # Group data by class
        class_data = {i: [] for i in range(n_classes)}
        for idx, (x, y) in enumerate(dataset):
            if y < n_classes:  # Ensure valid class
                class_data[y].append((x, y))
        
        # Check if we have enough data for each class
        min_samples_per_class = n_support + n_query
        for class_id, samples in class_data.items():
            if len(samples) < min_samples_per_class:
                return None, None, None, None  # Not enough data
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        # Sample from each class
        for class_id in range(n_classes):
            samples = class_data[class_id]
            np.random.shuffle(samples)
            
            # Support set
            for i in range(n_support):
                x, y = samples[i]
                support_data.append(x)
                support_labels.append(y)
            
            # Query set
            for i in range(n_support, n_support + n_query):
                x, y = samples[i]
                query_data.append(x)
                query_labels.append(y)
        
        # Convert to tensors
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels)
        
        if self.device == 'cuda':
            support_data = support_data.cuda()
            support_labels = support_labels.cuda()
            query_data = query_data.cuda()
            query_labels = query_labels.cuda()
        
        return support_data, support_labels, query_data, query_labels
    
    def _prepare_subject_dataset(self, subject_ids: List[int]):
        """Prepare dataset for specific subjects."""
        if not self.windows_dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")
            
        # For Schirrmeister2017, each subject has 2 sessions (datasets)
        dataset_indices = []
        for subject_id in subject_ids:
            dataset_indices.extend([2*(subject_id-1), 2*(subject_id-1)+1])
        
        # Extract datasets for specified subjects
        subject_datasets = [self.windows_dataset.datasets[i] for i in dataset_indices]
        
        # Create combined dataset
        from braindecode.datasets import BaseConcatDataset
        return BaseConcatDataset(subject_datasets)
    
    def plot_results(self, results: Dict[str, Any]):
        """Plot contrastive learning experiment results."""
        if 'subject_accuracies' not in results:
            self.logger.warning("No results to plot")
            return
        
        # Extract accuracies
        subject_ids = list(results['subject_accuracies'].keys())
        mean_accuracies = [results['subject_accuracies'][sid]['mean'] for sid in subject_ids]
        std_accuracies = [results['subject_accuracies'][sid]['std'] for sid in subject_ids]
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(subject_ids)), mean_accuracies, yerr=std_accuracies, 
                      capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        
        plt.xlabel('Subject ID')
        plt.ylabel('Classification Accuracy')
        plt.title(f'{self._get_experiment_name()} - Subject-wise Results')
        plt.xticks(range(len(subject_ids)), subject_ids)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add overall mean line
        overall_mean = results.get('mean_accuracy', np.mean(mean_accuracies))
        plt.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, 
                   label=f'Overall Mean: {overall_mean:.3f}')
        plt.legend()
        
        # Save plot
        plot_path = self.results_dir / 'results_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Results plot saved to {plot_path}")


class ClassPrototypeAttentionExperiment(ContrastiveLearningExperiment):
    """
    Class prototype attention experiment using labeled support set.
    Calculate prototypical support embedding for each label class from the labeled support set,
    and let task embeddings attend over these class prototypes.
    """
    
    def _get_experiment_name(self) -> str:
        version = self.config.get('experiment_version', 1)
        return f"ClassPrototypeAttention_{version}"
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the class prototype attention experiment."""
        # Load data
        subject_ids = list(range(1, 14))
        self.load_data(subject_ids)
        
        # Create models
        self.create_encoder()
        encoder_output_dim = self.encoder.get_output_dim() if hasattr(self.encoder, 'get_output_dim') else 40
        self.create_supportnet(encoder_output_dim)
        
        # Split subjects for training and testing
        train_subjects = subject_ids[:-3]  # Use most subjects for training
        test_subjects = subject_ids[-3:]   # Hold out last 3 for testing
        
        # Contrastive pretraining stage
        contrastive_results = self.contrastive_pretraining_stage(train_subjects)
        
        # Episodic training stage
        episodic_results = self.episodic_training_stage(train_subjects)
        
        # Evaluation stage
        evaluation_results = self.evaluation_stage(test_subjects)
        
        # Combine all results
        final_results = {
            **contrastive_results,
            **episodic_results,
            **evaluation_results,
            'config': self.config,
            'train_subjects': train_subjects,
            'test_subjects': test_subjects
        }
        
        return final_results


class ContrastiveBetweenSubjectsExperiment(ContrastiveLearningExperiment):
    """
    Contrastive learning between subjects experiment.
    This script only does the contrastive learning between people.
    """
    
    def _get_experiment_name(self) -> str:
        version = self.config.get('experiment_version', 1)
        return f"ContrastiveBetweenSubjects_{version}"
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the contrastive learning between subjects experiment."""
        # Load data
        subject_ids = list(range(1, 14))
        self.load_data(subject_ids)
        
        # Create encoder
        self.create_encoder()
        
        # Contrastive pretraining on all subjects
        contrastive_results = self.contrastive_pretraining_stage(subject_ids)
        
        # Simple evaluation: test encoder representations
        evaluation_results = self._evaluate_learned_representations(subject_ids)
        
        # Combine results
        final_results = {
            **contrastive_results,
            **evaluation_results,
            'config': self.config,
            'subjects': subject_ids
        }
        
        return final_results
    
    def _evaluate_learned_representations(self, subjects: List[int]) -> Dict[str, Any]:
        """Evaluate the quality of learned representations."""
        self.logger.info("Evaluating learned representations")
        
        self.encoder.eval()
        
        # Extract embeddings for each subject
        subject_embeddings = {}
        
        with torch.no_grad():
            for subject_id in subjects:
                subject_dataset = self._prepare_subject_dataset([subject_id])
                embeddings = []
                labels = []
                
                # Sample some data points
                n_samples = min(100, len(subject_dataset))
                indices = np.random.choice(len(subject_dataset), n_samples, replace=False)
                
                for idx in indices:
                    x, y = subject_dataset[idx]
                    x = x.unsqueeze(0)  # Add batch dimension
                    if self.device == 'cuda':
                        x = x.cuda()
                    
                    embedding = self.encoder(x)
                    embeddings.append(embedding.cpu().numpy())
                    labels.append(y)
                
                subject_embeddings[subject_id] = {
                    'embeddings': np.vstack(embeddings),
                    'labels': np.array(labels)
                }
        
        self.logger.info("Representation evaluation completed")
        return {
            'subject_embeddings': subject_embeddings,
            'evaluation_completed': True
        }