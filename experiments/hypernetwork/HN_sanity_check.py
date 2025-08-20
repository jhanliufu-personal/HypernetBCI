#!/usr/bin/env python3
"""
HN sanity check experiment: Train hypernetwork on single subject to verify functionality.
This takes arbitrary primary network XYZ and builds a hypernetwork over it. The entire 
architecture is referred to as HyperXYZ. HyperXYZ is trained with all training data from 
an individual subject to make sure that it at least performs similarly as the original network.

Refactored version using class-based approach.
"""

import argparse
from .hypernetwork_experiment import SanityCheckExperiment


def main():
    """Main entry point for the sanity check experiment."""
    parser = argparse.ArgumentParser(description='HypernetBCI Sanity Check Experiment')
    parser.add_argument('--subject_id', type=int, default=3, help='Subject ID to use for sanity check')
    parser.add_argument('--model_name', type=str, default='ShallowFBCSPNet', help='Model name')
    parser.add_argument('--gpu_number', type=str, default='0', help='GPU number')
    parser.add_argument('--experiment_version', type=int, default=1, help='Experiment version')
    parser.add_argument('--n_epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0625*0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')
    
    args = parser.parse_args()
    
    # Convert args to config dictionary
    config = {
        'subject_id': args.subject_id,
        'gpu_number': args.gpu_number,
        'model_name': args.model_name,
        'dataset_name': 'Schirrmeister2017',
        'experiment_version': args.experiment_version,
        'n_classes': args.n_classes,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'weight_decay': 0,
        'repetition': 3,
    }
    
    # Create and run experiment
    experiment = SanityCheckExperiment(config_dict=config)
    results = experiment.run()
    
    print("Sanity check experiment completed successfully!")
    return results


if __name__ == "__main__":
    main()