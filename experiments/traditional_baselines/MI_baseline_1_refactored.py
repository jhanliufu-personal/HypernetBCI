#!/usr/bin/env python3
"""
Baseline 1: Train from scratch approach.
Model is trained from scratch for each subject with varying amounts of training data.

Refactored version using class-based approach.
"""

import argparse
from .baseline_experiment import FromScratchExperiment


def main():
    """Main entry point for the from-scratch baseline experiment."""
    parser = argparse.ArgumentParser(description='From-Scratch Baseline Experiment')
    parser.add_argument('--model_name', type=str, default='ShallowFBCSPNet', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='Schirrmeister2017', help='Dataset name')
    parser.add_argument('--experiment_version', type=int, default=3, help='Experiment version')
    parser.add_argument('--data_amount_step', type=int, default=20, help='Data amount step size')
    parser.add_argument('--repetition', type=int, default=10, help='Number of repetitions per data amount')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--lr', type=float, default=0.0625*0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--data_amount_unit', type=str, default='min', help='Data amount unit (min, sec, trial)')
    parser.add_argument('--trial_len_sec', type=int, default=4, help='Trial length in seconds')
    parser.add_argument('--significance_level', type=float, default=0.95, help='Significance level for statistics')
    parser.add_argument('--max_data_amount', type=int, default=200, help='Maximum data amount to test')
    
    args = parser.parse_args()
    
    # Convert args to config dictionary
    config = {
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'experiment_version': args.experiment_version,
        'data_amount_step': args.data_amount_step,
        'repetition': args.repetition,
        'n_classes': args.n_classes,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'data_amount_unit': args.data_amount_unit,
        'trial_len_sec': args.trial_len_sec,
        'significance_level': args.significance_level,
        'max_data_amount': args.max_data_amount,
        'use_braindecode_classifier': True,
    }
    
    # Create and run experiment
    experiment = FromScratchExperiment(config_dict=config)
    results = experiment.run()
    
    print("From-scratch baseline experiment completed successfully!")
    return results


if __name__ == "__main__":
    main()