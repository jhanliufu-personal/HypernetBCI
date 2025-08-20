#!/usr/bin/env python3
"""
This script only does the contrastive learning between people. 
Use later versions to do both that and classification.

Refactored version using class-based approach.
"""

import argparse
from .contrastive_learning_experiment import ContrastiveBetweenSubjectsExperiment


def main():
    """Main entry point for the contrastive learning between subjects experiment."""
    parser = argparse.ArgumentParser(description='Contrastive Learning Between Subjects Experiment')
    parser.add_argument('--gpu_number', type=str, default='0', help='GPU number')
    parser.add_argument('--experiment_version', type=int, default=0, help='Experiment version')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=6.5e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=78, help='Batch size')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for contrastive loss')
    
    args = parser.parse_args()
    
    # Convert args to config dictionary
    config = {
        'gpu_number': args.gpu_number,
        'experiment_version': args.experiment_version,
        'n_classes': args.n_classes,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'weight_decay': 0,
        'temperature': args.temperature,
    }
    
    # Create and run experiment
    experiment = ContrastiveBetweenSubjectsExperiment(config_dict=config)
    results = experiment.run()
    
    print("Contrastive learning between subjects experiment completed successfully!")
    return results


if __name__ == "__main__":
    main()