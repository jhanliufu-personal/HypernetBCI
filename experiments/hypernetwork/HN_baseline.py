#!/usr/bin/env python3
"""
HN baseline 1 experiment: Train hypernetwork from scratch for each subject.
This is similar to baseline 1 but with hypernet. HyperXYZ is trained from scratch for each 
person using varying amount of training data.

Refactored version using class-based approach.
"""

import argparse
from core.utils import parse_training_config
from .hypernetwork_experiment import BaselineExperiment


def main():
    """Main entry point for the hypernetwork baseline experiment."""
    # Parse arguments
    args = parse_training_config()
    
    # Convert args to config dictionary
    config = {
        'gpu_number': args.gpu_number,
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'experiment_version': args.experiment_version,
        'model_kwargs': args.model_kwargs,
        'data_amount_step': args.data_amount_step,
        'repetition': args.repetition,
        'n_classes': args.n_classes,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'weight_decay': args.weight_decay,
        'data_amount_unit': args.data_amount_unit,
        'trial_len_sec': args.trial_len_sec,
        'significance_level': args.significance_level,
    }
    
    # Create and run experiment
    experiment = BaselineExperiment(config_dict=config)
    results = experiment.run()
    
    print("Hypernetwork baseline experiment completed successfully!")
    return results


if __name__ == "__main__":
    main()