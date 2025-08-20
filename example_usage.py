#!/usr/bin/env python3
"""
Example script showing how to use the refactored class-based experiment system.
"""

def example_hypernetwork_experiment():
    """Example of running a hypernetwork experiment."""
    from experiments.hypernetwork.hypernetwork_experiment import CrossSubjectCalibrationExperiment
    
    # Define configuration
    config = {
        'gpu_number': '0',
        'model_name': 'ShallowFBCSPNet',
        'dataset_name': 'Schirrmeister2017',
        'experiment_version': 1,
        'n_classes': 4,
        'lr': 0.0625 * 0.01,
        'batch_size': 64,
        'n_epochs': 5,  # Reduced for example
        'weight_decay': 0,
        'data_amount_step': 40,
        'repetition': 2,  # Reduced for example
        'embedder_type': 'EEGConformer',
        'embedding_dim': 32,
        'embedding_length': 128,
    }
    
    # Create and run experiment
    experiment = CrossSubjectCalibrationExperiment(config_dict=config)
    results = experiment.run()
    
    print("Hypernetwork experiment completed!")
    return results


def example_contrastive_learning_experiment():
    """Example of running a contrastive learning experiment."""
    from experiments.contrastive_learning.contrastive_learning_experiment import ClassPrototypeAttentionExperiment
    
    # Define configuration
    config = {
        'gpu_number': '0',
        'experiment_version': 1,
        'n_classes': 4,
        'lr': 6.5e-4,
        'batch_size': 32,  # Reduced for example
        'n_epochs': 5,     # Reduced for example
        'temperature': 0.5,
        'n_support_shots': 5,
        'n_query_shots': 15,
        'embedding_dim': 40,
        'use_attention': True,
        'freeze_encoder': True,
        'n_test_episodes': 50,  # Reduced for example
    }
    
    # Create and run experiment
    experiment = ClassPrototypeAttentionExperiment(config_dict=config)
    results = experiment.run()
    
    print("Contrastive learning experiment completed!")
    return results


def example_baseline_experiment():
    """Example of running a baseline experiment."""
    from experiments.traditional_baselines.baseline_experiment import FromScratchExperiment
    
    # Define configuration
    config = {
        'model_name': 'ShallowFBCSPNet',
        'dataset_name': 'Schirrmeister2017',
        'experiment_version': 1,
        'data_amount_step': 20,
        'repetition': 2,  # Reduced for example
        'n_classes': 4,
        'lr': 0.0625 * 0.01,
        'weight_decay': 0,
        'batch_size': 64,
        'n_epochs': 5,  # Reduced for example
        'max_data_amount': 60,  # Reduced for example
        'use_braindecode_classifier': True,
    }
    
    # Create and run experiment
    experiment = FromScratchExperiment(config_dict=config)
    results = experiment.run()
    
    print("Baseline experiment completed!")
    return results


def main():
    """Run example experiments."""
    print("HypernetBCI Refactored Experiments - Examples")
    print("=" * 50)
    
    # You can uncomment any of these to run the examples:
    
    # print("\n1. Running Hypernetwork Cross-Subject Calibration Example...")
    # example_hypernetwork_experiment()
    
    # print("\n2. Running Contrastive Learning Example...")
    # example_contrastive_learning_experiment()
    
    # print("\n3. Running Baseline From-Scratch Example...")
    # example_baseline_experiment()
    
    print("\nTo run these examples, uncomment the desired lines in main() function.")
    print("\nAlternatively, use the experiment runner:")
    print("python run_experiment.py --experiment-class CrossSubjectCalibrationExperiment --config config/example_config.json")


if __name__ == "__main__":
    main()