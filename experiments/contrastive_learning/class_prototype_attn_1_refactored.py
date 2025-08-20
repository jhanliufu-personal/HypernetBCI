#!/usr/bin/env python3
"""
Test-time adaptation using labeled support set. Calculate prototypical support
embedding for each label class from the labeled support set, and let task embeddings
attend over these class prototypes.

Refactored version using class-based approach.
"""

import argparse
from .contrastive_learning_experiment import ClassPrototypeAttentionExperiment


def main():
    """Main entry point for the class prototype attention experiment."""
    parser = argparse.ArgumentParser(description='Class Prototype Attention Experiment')
    parser.add_argument('--gpu_number', type=str, default='1', help='GPU number')
    parser.add_argument('--experiment_version', type=int, default=1, help='Experiment version')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=6.5e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=72, help='Batch size')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for contrastive loss')
    parser.add_argument('--n_support_shots', type=int, default=5, help='Number of support shots per class')
    parser.add_argument('--n_query_shots', type=int, default=15, help='Number of query shots per class')
    parser.add_argument('--embedding_dim', type=int, default=40, help='Embedding dimension')
    
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
        'n_support_shots': args.n_support_shots,
        'n_query_shots': args.n_query_shots,
        'embedding_dim': args.embedding_dim,
        'use_attention': True,
        'freeze_encoder': True,
        'n_test_episodes': 100,
    }
    
    # Create and run experiment
    experiment = ClassPrototypeAttentionExperiment(config_dict=config)
    results = experiment.run()
    
    print("Class prototype attention experiment completed successfully!")
    return results


if __name__ == "__main__":
    main()