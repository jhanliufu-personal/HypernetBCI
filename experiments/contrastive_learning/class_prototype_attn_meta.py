#!/usr/bin/env python3
"""
Test-time adaptation using labeled support set with meta-learning approach.
Calculate prototypical support embedding for each label class from the labeled support set,
and let task embeddings attend over these class prototypes. Use meta-learning style training, 
each subject being one "task".

Refactored version using class-based approach.
"""

import argparse
from .contrastive_learning_experiment import ClassPrototypeAttentionMetaExperiment


def main():
    """Main entry point for the meta-learning class prototype attention experiment."""
    parser = argparse.ArgumentParser(description='Meta-Learning Class Prototype Attention Experiment')
    parser.add_argument('--gpu_number', type=str, default='2', help='GPU number')
    parser.add_argument('--experiment_version', type=int, default=1, help='Experiment version')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=6.5e-4, help='Learning rate')
    parser.add_argument('--meta_batch_size', type=int, default=72, help='Meta-learning batch size')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--support_encoder_experiment', type=str, default='class_prototype_attention_1', 
                       help='Name of experiment folder containing pre-trained support encoder')
    parser.add_argument('--use_cosine_scheduler', action='store_true', default=True,
                       help='Use cosine annealing learning rate scheduler')
    parser.add_argument('--subject_ids', type=int, nargs='+', default=list(range(1, 14)),
                       help='List of subject IDs to use (default: all subjects 1-13)')
    
    args = parser.parse_args()
    
    # Convert args to config dictionary
    config = {
        'gpu_number': args.gpu_number,
        'experiment_version': args.experiment_version,
        'n_classes': args.n_classes,
        'lr': args.lr,
        'meta_batch_size': args.meta_batch_size,
        'n_epochs': args.n_epochs,
        'weight_decay': args.weight_decay,
        'support_encoder_experiment': args.support_encoder_experiment,
        'use_cosine_scheduler': args.use_cosine_scheduler,
        'subject_ids': args.subject_ids,
    }
    
    # Create and run experiment
    experiment = ClassPrototypeAttentionMetaExperiment(config_dict=config)
    results = experiment.run()
    
    print("Meta-learning class prototype attention experiment completed successfully!")
    
    # Print summary results
    if 'subject_results' in results:
        valid_accuracies = [acc for acc in results['subject_results'].values() if acc is not None]
        if valid_accuracies:
            mean_acc = results.get('mean_accuracy', 0) * 100
            std_acc = results.get('std_accuracy', 0) * 100
            print(f"Overall Results: {mean_acc:.2f}% ± {std_acc:.2f}%")
            
            # Print per-subject results
            print("\nPer-subject results:")
            for key, acc in results['subject_results'].items():
                if acc is not None:
                    subject_id = key.split('_')[-1]
                    print(f"  Subject {subject_id}: {acc*100:.2f}%")
    
    return results


if __name__ == "__main__":
    main()