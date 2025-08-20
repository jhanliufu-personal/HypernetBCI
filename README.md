# HypernetBCI

Quick calibration of deep learning-based brain-computer interface (BCI) models using hypernetworks and contrastive learning.

## Table of Contents

- [Quickstart](#quickstart)
- [Codebase Structure](#codebase-structure)
- [HyperNet Experiments](#hypernet-experiments)
- [SupportNet Experiments](#supportnet-experiments)

## Quickstart

### Setup Environment

```bash
# Clone the repository
git clone <repository_url>
cd HypernetBCI/METHODS/HypernetBCI

# Create conda environment
conda env create -f environment.yml
conda activate hypernetbci
```

### Run an Experiment

```bash
# Run cross-subject hypernetwork calibration experiment
python run_experiment.py --experiment-class CrossSubjectCalibrationExperiment --gpu 0

# Or run a traditional baseline for comparison
python run_experiment.py --experiment-class FromScratchExperiment --gpu 0

# Run with custom configuration
python run_experiment.py --experiment-class ClassPrototypeAttentionExperiment --config config/example_config.json --gpu 0
```

## Codebase Structure

```
HypernetBCI/
├── core/                           # Core utilities and base classes
│   ├── base_experiment.py          # Base experiment class with common functionality
│   ├── utils.py                    # Training, evaluation, and data utilities
│   └── loss.py                     # Custom loss functions (contrastive loss)
├── data_utils/                     # Data processing and dataset utilities
│   ├── dataset_download.py         # Download EEG datasets (MOABB)
│   ├── preprocess_dataset.py       # EEG preprocessing pipeline
│   └── get_all_embeddings.py       # Extract and analyze embeddings
├── models/                         # Neural network architectures
│   ├── HypernetBCI.py             # Main HyperBCI model combining primary net + hypernet
│   ├── Embedder.py                # Embedding networks (EEGConformer, ShallowFBCSP)
│   ├── Hypernet.py                # Hypernetworks that generate weights
│   └── Supportnet.py              # Support networks for prototype attention
├── experiments/                    # Organized experiment implementations
│   ├── hypernetwork/              # Hypernetwork-based BCI experiments
│   ├── contrastive_learning/      # Contrastive learning and prototype attention
│   └── traditional_baselines/     # Standard BCI training approaches
├── baselines/                     # External baseline implementations
│   ├── MAPU/                      # Multi-source Adversarial Domain Adaptation
│   └── CLUDA/                     # Cross-participant Learning using Domain Adaptation
├── config/                        # Experiment configuration files (JSON)
├── results/                       # Experiment outputs and saved models
└── run_experiment.py              # Unified experiment runner
```

### Folder Details

- **`core/`**: Base experiment classes, training utilities, and loss functions shared across all experiments
- **`data_utils/`**: Scripts for downloading, preprocessing, and analyzing EEG datasets
- **`models/`**: Neural network architectures including hypernetworks, embedders, and support networks
- **`experiments/`**: Your research contributions organized by methodology (hypernetworks, contrastive learning, baselines)
- **`baselines/`**: External baseline methods (MAPU, CLUDA) for comparison
- **`config/`**: JSON configuration files specifying hyperparameters for different experiments
- **`results/`**: Generated experiment results, plots, and trained model checkpoints

## HyperNet Experiments

### Overview

HyperNetworks generate subject-specific weights for BCI models, enabling rapid calibration with minimal data. Instead of fine-tuning entire models, hypernetworks generate only the final classification layer weights based on subject-specific embeddings. This approach dramatically reduces calibration time and data requirements.

### Experiment Types

| Experiment | Description | Key Features |
|------------|-------------|--------------|
| **CrossSubjectCalibrationExperiment** | Main experiment: pretrain hypernet on multiple subjects, then calibrate on target subject | • Leave-one-subject-out validation<br>• Unsupervised calibration<br>• Varying data amounts |
| **SanityCheckExperiment** | Verify hypernet performs similarly to original network on single subject | • Single subject training<br>• Baseline comparison<br>• Architecture validation |
| **BaselineExperiment** | Train hypernet from scratch per subject (no cross-subject transfer) | • Subject-specific training<br>• No transfer learning<br>• Comparison baseline |

### Key Components

- **Embedders**: Extract subject-specific features (EEGConformer, ShallowFBCSP, Conv1D)
- **Hypernetworks**: Generate classification weights from embeddings
- **Primary Networks**: Base BCI models (ShallowFBCSPNet, EEGConformer)
- **Calibration**: Aggregate embeddings from calibration data to generate final weights

## SupportNet Experiments

### Overview

SupportNet uses **contrastive learning** and **class prototype attention** for few-shot BCI adaptation. The approach learns generalizable representations through contrastive learning between subjects, then uses prototype attention mechanisms to adapt to new subjects with minimal labeled data. Meta-learning treats each subject as a separate "task" for improved generalization.

### Experiment Types

| Experiment | Description | Key Features |
|------------|-------------|--------------|
| **ClassPrototypeAttentionExperiment** | Few-shot adaptation using labeled support sets and class prototypes | • **Prototype attention**<br>• Episodic training<br>• Support/query sets |
| **ClassPrototypeAttentionMetaExperiment** | Meta-learning version where each subject is treated as a separate task | • **Meta-learning** approach<br>• Subject-as-task paradigm<br>• Pre-trained support encoder |
| **ContrastiveBetweenSubjectsExperiment** | Pure contrastive learning between subjects to learn generalizable representations | • **Contrastive learning**<br>• Cross-subject representation<br>• Temperature-based loss |

### Key Components

- **Support Encoder**: Extracts prototypical embeddings from support sets (frozen after pre-training)
- **Task Encoder**: Extracts embeddings from query/target data
- **Attention Mechanism**: Allows task embeddings to attend over class prototypes
- **Episodic Training**: Few-shot learning with support/query episodes
- **Contrastive Loss**: Learns to distinguish between different subjects while maintaining class structure

### Meta-Learning Features

- **Subject-as-Task**: Each subject treated as a separate meta-learning task
- **Support Sets**: Small labeled datasets used to compute class prototypes
- **Query Sets**: Test data for few-shot evaluation
- **Attention Over Prototypes**: Dynamic weighting of class prototypes based on query similarity