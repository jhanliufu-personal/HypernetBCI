"""
Base experiment class that provides common functionality for all HypernetBCI experiments.
All experiments should inherit from this class to maintain consistency and reduce code duplication.
"""

import os
import json
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import DataLoader

from braindecode.datasets import MOABBDataset, BaseConcatDataset
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import (
    Preprocessor, exponential_moving_standardize, preprocess, create_windows_from_events
)
from braindecode.util import set_random_seeds
from numpy import multiply

from core.utils import get_subset, import_model, parse_training_config


class BaseExperiment(ABC):
    """
    Base class for all HypernetBCI experiments.
    
    This class provides common functionality including:
    - Configuration loading and parsing
    - Device setup (CUDA/CPU)
    - Data loading and preprocessing
    - Results saving and plotting
    - Logging setup
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize the experiment.
        
        Args:
            config_path: Path to JSON configuration file
            config_dict: Dictionary containing configuration (alternative to config_path)
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = self._get_default_config()
            
        # Set up logging
        self._setup_logging()
        
        # Initialize experiment attributes
        self.device = None
        self.results = {}
        self.windows_dataset = None
        self.sfreq = None
        self.model = None
        
        # Setup experiment
        self._setup_experiment()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration. Should be overridden by subclasses."""
        return {
            "gpu_number": "0",
            "model_name": "ShallowFBCSPNet",
            "dataset_name": "Schirrmeister2017",
            "n_classes": 4,
            "lr": 0.0625 * 0.01,
            "batch_size": 64,
            "n_epochs": 40,
            "weight_decay": 0,
            "repetition": 3,
            "significance_level": 0.95,
            "experiment_version": 1
        }
    
    def _setup_logging(self):
        """Set up logging for the experiment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        warnings.filterwarnings('ignore')
    
    def _setup_experiment(self):
        """Setup experiment directory and CUDA device."""
        # Setup CUDA
        self._setup_device()
        
        # Create results directory
        self._setup_results_directory()
    
    def _setup_device(self):
        """Setup CUDA device based on configuration."""
        gpu_number = str(self.config.get('gpu_number', '0'))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
        
        cuda = torch.cuda.is_available()
        if cuda:
            device_count = torch.cuda.device_count()
            self.logger.info(f'{device_count} CUDA devices available, using GPU {gpu_number}')
            torch.backends.cudnn.benchmark = True
            self.device = 'cuda'
        else:
            self.logger.info('No CUDA available, using CPU')
            self.device = 'cpu'
    
    def _setup_results_directory(self):
        """Create results directory."""
        experiment_name = self._get_experiment_name()
        self.results_dir = Path('results') / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Results will be saved to: {self.results_dir}')
    
    @abstractmethod
    def _get_experiment_name(self) -> str:
        """Get the experiment name for results directory. Must be implemented by subclasses."""
        pass
    
    def load_data(self, subject_ids: Optional[List[int]] = None) -> BaseConcatDataset:
        """
        Load and preprocess data.
        
        Args:
            subject_ids: List of subject IDs to load. If None, loads default range.
            
        Returns:
            Loaded and preprocessed dataset
        """
        if subject_ids is None:
            subject_ids = list(range(1, 14))  # Default for Schirrmeister2017
            
        preprocessed_dir = 'data/Schirrmeister2017_preprocessed'
        
        # Try to load preprocessed data first
        if os.path.exists(preprocessed_dir) and os.listdir(preprocessed_dir):
            self.logger.info('Loading preprocessed dataset')
            self.windows_dataset = load_concat_dataset(
                path=preprocessed_dir,
                preload=True,
                ids_to_load=list(range(2 * subject_ids[-1])),
                target_name=None,
            )
            self.sfreq = self.windows_dataset.datasets[0].raw.info['sfreq']
            self.logger.info('Preprocessed dataset loaded')
        else:
            self.logger.info('Loading raw dataset for preprocessing')
            self.windows_dataset = self._load_and_preprocess_raw_data(subject_ids)
            
        return self.windows_dataset
    
    def _load_and_preprocess_raw_data(self, subject_ids: List[int]) -> BaseConcatDataset:
        """Load and preprocess raw data from MOABB."""
        dataset_name = self.config.get('dataset_name', 'Schirrmeister2017')
        dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_ids)
        self.logger.info('Raw dataset loaded')
        
        # Preprocessing parameters
        low_cut_hz = 4.0
        high_cut_hz = 38.0
        factor_new = 1e-3
        init_block_size = 1000
        factor = 1e6  # Convert from V to uV
        
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),
            Preprocessor(lambda data: multiply(data, factor)),
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
            Preprocessor(exponential_moving_standardize,
                        factor_new=factor_new, init_block_size=init_block_size)
        ]
        
        preprocess(dataset, preprocessors, n_jobs=-1)
        self.logger.info('Dataset preprocessed')
        
        # Extract trial windows
        trial_start_offset_seconds = -0.5
        self.sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == self.sfreq for ds in dataset.datasets])
        
        trial_start_offset_samples = int(trial_start_offset_seconds * self.sfreq)
        
        trial_len_sec = self.config.get('trial_len_sec', 4)
        window_size_samples = int(trial_len_sec * self.sfreq)
        
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples,
            preload=True,
        )
        
        self.logger.info('Trial windows extracted')
        return windows_dataset
    
    def create_model(self, **kwargs):
        """Create model based on configuration. Should be overridden by subclasses."""
        model_name = self.config.get('model_name', 'ShallowFBCSPNet')
        model_object = import_model(model_name)
        
        n_chans = self.windows_dataset[0][0].shape[0] if self.windows_dataset else 22
        n_classes = self.config.get('n_classes', 4)
        input_window_samples = self.windows_dataset[0][0].shape[1] if self.windows_dataset else int(4 * 250)
        
        model_kwargs = self.config.get('model_kwargs', {})
        model_kwargs.update(kwargs)
        
        self.model = model_object(
            n_chans,
            n_classes,
            input_window_samples=input_window_samples,
            **model_kwargs
        )
        
        if self.device == 'cuda':
            self.model = self.model.cuda()
            
        self.logger.info(f'Created {model_name} model')
        return self.model
    
    @abstractmethod
    def run_experiment(self):
        """Run the main experiment. Must be implemented by subclasses."""
        pass
    
    def save_results(self, results: Dict[str, Any], filename: str = 'results.pkl'):
        """Save experiment results."""
        results_path = self.results_dir / filename
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        self.logger.info(f'Results saved to {results_path}')
    
    def load_results(self, filename: str = 'results.pkl') -> Dict[str, Any]:
        """Load experiment results."""
        results_path = self.results_dir / filename
        if results_path.exists():
            with open(results_path, 'rb') as f:
                return pickle.load(f)
        else:
            self.logger.warning(f'Results file not found: {results_path}')
            return {}
    
    def save_model(self, model: torch.nn.Module, filename: str = 'model.pth', 
                   save_config: bool = True):
        """
        Save trained model weights and configuration.
        
        Args:
            model: The trained PyTorch model
            filename: Model filename (default: 'model.pth')
            save_config: Whether to save model configuration alongside weights
        """
        from copy import deepcopy
        
        # Save model state dict
        model_path = self.results_dir / filename
        torch.save(deepcopy(model.state_dict()), model_path)
        self.logger.info(f'Model weights saved to {model_path}')
        
        # Save model configuration for deployment
        if save_config:
            config_filename = filename.replace('.pth', '_config.json')
            config_path = self.results_dir / config_filename
            
            # Create model configuration for deployment
            model_config = {
                'model_type': self.__class__.__name__.lower().replace('experiment', ''),
                'model_class': model.__class__.__name__,
                'experiment_config': self.config,
                'input_shape': getattr(model, 'input_shape', None),
                'output_shape': getattr(model, 'output_shape', None),
                'created_at': str(datetime.now())
            }
            
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            self.logger.info(f'Model configuration saved to {config_path}')
    
    def load_model(self, filename: str = 'model.pth') -> Optional[torch.nn.Module]:
        """
        Load saved model weights.
        
        Args:
            filename: Model filename to load
            
        Returns:
            Loaded model with weights, or None if not found
        """
        model_path = self.results_dir / filename
        if not model_path.exists():
            self.logger.warning(f'Model file not found: {model_path}')
            return None
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Try to create model and load weights
        if hasattr(self, 'model') and self.model is not None:
            self.model.load_state_dict(state_dict)
            self.logger.info(f'Model weights loaded from {model_path}')
            return self.model
        else:
            self.logger.warning('No model instance available. Create model first before loading weights.')
            return None
    
    def plot_results(self, results: Dict[str, Any]):
        """Plot experiment results. Should be overridden by subclasses."""
        self.logger.info("Base plotting method - override in subclass for custom plots")
        
    def _calculate_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data."""
        alpha = 1.0 - confidence
        return stats.t.interval(alpha, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
    
    def run(self):
        """Main entry point to run the experiment."""
        try:
            self.logger.info(f"Starting experiment: {self._get_experiment_name()}")
            results = self.run_experiment()
            self.save_results(results)
            self.plot_results(results)
            self.logger.info("Experiment completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Experiment failed with error: {str(e)}")
            raise