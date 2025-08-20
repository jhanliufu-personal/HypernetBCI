#!/usr/bin/env python3
"""
HN cross-subject calibration experiment: hold out each person as the new arrival, and pre-train
the HyperBCI on everyone else put together as the pre-train pool. For the new arrival person, 
their data is split into calibration set and validation set. Varying amount of data is drawn
from the calibration set for calibration, then the calibrated model is evaluated using the test set.

The calibration process is unsupervised; the HN is expected to pick up relevant info from the
calibration set.

Refactored version using class-based approach.
"""

import argparse
from core.utils import parse_training_config
from .hypernetwork_experiment import CrossSubjectCalibrationExperiment


def main():
    """Main entry point for the cross-subject calibration experiment."""
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
    experiment = CrossSubjectCalibrationExperiment(config_dict=config)
    results = experiment.run()
    
    print("Cross-subject calibration experiment completed successfully!")
    return results


if __name__ == "__main__":
    main()