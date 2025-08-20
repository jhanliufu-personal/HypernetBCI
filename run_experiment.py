#!/usr/bin/env python3
"""
Main experiment runner for HypernetBCI experiments.

Usage:
    # Using original scripts
    python run_experiment.py --experiment hypernetwork --script MI_HN_cross_subject_calibration.py --config config/MI_HN_xsubj_calib_1.json
    
    # Using refactored class-based scripts  
    python run_experiment.py --experiment hypernetwork --script MI_HN_cross_subject_calibration_refactored.py --config config/MI_HN_xsubj_calib_1.json
    python run_experiment.py --experiment contrastive_learning --script class_prototype_attn_1_refactored.py
    python run_experiment.py --experiment traditional_baselines --script MI_baseline_1_refactored.py
    
    # Using class-based approach directly
    python run_experiment.py --experiment-class CrossSubjectCalibrationExperiment --config config/MI_HN_xsubj_calib_1.json
    python run_experiment.py --experiment-class ClassPrototypeAttentionExperiment
"""

import os
import sys
import argparse
import subprocess
import importlib
from pathlib import Path
from typing import Optional, Dict, Any


def get_experiment_path(experiment_type: str, script_name: str) -> Path:
    """Get the full path to an experiment script."""
    base_dir = Path(__file__).parent
    
    experiment_dirs = {
        'hypernetwork': 'experiments/hypernetwork',
        'contrastive_learning': 'experiments/contrastive_learning', 
        'traditional_baselines': 'experiments/traditional_baselines',
        'mapu': 'baselines/MAPU',
        'cluda': 'baselines/CLUDA'
    }
    
    if experiment_type not in experiment_dirs:
        raise ValueError(f"Unknown experiment type: {experiment_type}. "
                        f"Available types: {list(experiment_dirs.keys())}")
    
    script_path = base_dir / experiment_dirs[experiment_type] / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    return script_path


def run_experiment_class(experiment_class_name: str, config_path: Optional[str] = None, 
                        config_dict: Optional[Dict[str, Any]] = None):
    """Run experiment using class-based approach."""
    
    # Map experiment class names to their modules and classes
    experiment_classes = {
        # Hypernetwork experiments
        'CrossSubjectCalibrationExperiment': ('experiments.hypernetwork.hypernetwork_experiment', 'CrossSubjectCalibrationExperiment'),
        'SanityCheckExperiment': ('experiments.hypernetwork.hypernetwork_experiment', 'SanityCheckExperiment'),
        'HypernetBaselineExperiment': ('experiments.hypernetwork.hypernetwork_experiment', 'BaselineExperiment'),
        
        # Contrastive learning experiments
        'ClassPrototypeAttentionExperiment': ('experiments.contrastive_learning.contrastive_learning_experiment', 'ClassPrototypeAttentionExperiment'),
        'ContrastiveBetweenSubjectsExperiment': ('experiments.contrastive_learning.contrastive_learning_experiment', 'ContrastiveBetweenSubjectsExperiment'),
        
        # Traditional baselines
        'FromScratchExperiment': ('experiments.traditional_baselines.baseline_experiment', 'FromScratchExperiment'),
        'TransferLearningExperiment': ('experiments.traditional_baselines.baseline_experiment', 'TransferLearningExperiment'),
    }
    
    if experiment_class_name not in experiment_classes:
        raise ValueError(f"Unknown experiment class: {experiment_class_name}. "
                        f"Available classes: {list(experiment_classes.keys())}")
    
    module_name, class_name = experiment_classes[experiment_class_name]
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Get the class
        experiment_class = getattr(module, class_name)
        
        # Create and run experiment
        if config_path:
            experiment = experiment_class(config_path=config_path)
        elif config_dict:
            experiment = experiment_class(config_dict=config_dict)
        else:
            experiment = experiment_class()
        
        print(f"Running experiment: {experiment_class_name}")
        results = experiment.run()
        
        print(f"Experiment {experiment_class_name} completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error running experiment class {experiment_class_name}: {e}", file=sys.stderr)
        raise


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    import json
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run HypernetBCI experiments")
    
    # Two modes: script-based or class-based
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment', 
                      choices=['hypernetwork', 'contrastive_learning', 'traditional_baselines', 'mapu', 'cluda'],
                      help='Type of experiment to run (script-based mode)')
    group.add_argument('--experiment-class',
                      help='Name of experiment class to run (class-based mode)')
    
    parser.add_argument('--script',
                       help='Name of the experiment script to run (required for script-based mode)')
    parser.add_argument('--config', 
                       help='Path to configuration file (JSON format)')
    parser.add_argument('--gpu', default='0',
                       help='GPU number to use (default: 0)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print the command that would be run without executing')
    
    args = parser.parse_args()
    
    # Set GPU environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    try:
        if args.experiment_class:
            # Class-based mode
            print(f"Running experiment class: {args.experiment_class}")
            if args.config:
                print(f"Config: {args.config}")
            print(f"GPU: {args.gpu}")
            print("-" * 50)
            
            if args.dry_run:
                print("Dry run - experiment not executed")
                return
            
            # Load config if provided
            config_dict = None
            if args.config:
                config_dict = load_config_file(args.config)
                # Add GPU setting to config
                config_dict['gpu_number'] = args.gpu
            else:
                config_dict = {'gpu_number': args.gpu}
            
            # Run experiment
            results = run_experiment_class(args.experiment_class, config_dict=config_dict)
            
        else:
            # Script-based mode
            if not args.script:
                print("Error: --script is required when using --experiment", file=sys.stderr)
                sys.exit(1)
                
            script_path = get_experiment_path(args.experiment, args.script)
            
            # Construct the command
            cmd = [sys.executable, str(script_path)]
            
            # Add config if provided
            if args.config:
                cmd.extend(['--config', args.config])
            
            # Print command info
            print(f"Running {args.experiment} experiment:")
            print(f"Script: {script_path}")
            if args.config:
                print(f"Config: {args.config}")
            print(f"GPU: {args.gpu}")
            print(f"Command: {' '.join(cmd)}")
            print("-" * 50)
            
            if args.dry_run:
                print("Dry run - command not executed")
                return
            
            # Run the experiment
            result = subprocess.run(cmd)
            sys.exit(result.returncode)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()