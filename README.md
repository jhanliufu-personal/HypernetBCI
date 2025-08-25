# HyperUDA and SupportNet: zero-shot and few-shot DNN adaptation using hypernetwork and contrastive learning 

Applied to create DNN-based self-calibrating brain computer interface (BCI). This is Jhan's Bachelor's [thesis]() work. Publication is underway.

## Table of Contents

- [Quickstart](#quickstart)
- [Evaluation](#evaluation) 
- [Codebase Structure](#codebase-structure)
- [HyperNet Experiments](#hypernet-experiments)
- [SupportNet Experiments](#supportnet-experiments)
- [MSP Deployment](#msp-deployment)

## Quickstart

### Setup Environment

```bash
# Clone the repository
git clone git@github.com:jhanliufu-personal/HypernetBCI.git
cd HypernetBCI/METHODS/HypernetBCI

# Create conda environment
conda env create -f environment.yml
conda activate hypernetbci

# Clone and setup Lupe for MSP deployment (optional)
git clone https://github.com/mingyuan-xiang/lupe.git
cd lupe
pip install -e .
cd ..
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

## Evaluation

We focus on domain adaptation (DA). The specific task we take on is quickly calibrating a pretrained DL-based
BCI model to an unseen individual, whose brain and neural signal features (hopefully) differ from the pretrain set. We selected two BCI tasks (datasets) for evaluation, a [motor](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Schirrmeister2017.html) imagery decoding task and a [sleep](https://braindecode.org/stable/generated/braindecode.datasets.SleepPhysionet.html) staging task. The BCI models we selected are [ShallowFBCSPNet](https://braindecode.org/0.7/generated/braindecode.models.ShallowFBCSPNet.html#braindecode.models.ShallowFBCSPNet), [SleepStagerChambon2018](https://braindecode.org/0.7/generated/braindecode.models.SleepStagerChambon2018.html#braindecode.models.SleepStagerChambon2018), [SleepStagerEldele2021](https://braindecode.org/0.7/generated/braindecode.models.SleepStagerEldele2021.html#braindecode.models.SleepStagerEldele2021) and [TCN](https://braindecode.org/0.7/generated/braindecode.models.TCN.html#braindecode.models.TCN). These models involve well-known architectures such as temporal convolution and transformers. Datasets and model implementations are from the [braindecode](https://braindecode.org/stable/index.html) library.

We compare our method against two baselines. [CLUDA](https://arxiv.org/pdf/2206.06243) and [MAPU](https://arxiv.org/html/2406.02635v2) achieve zero-shot adaptation through contrastive learning and temporal imputation at test time. Although unsupervised, they require iterative optimization and source data for adaptation, which exclude them from time-sensitive and resource-constrained applications. We also compare our method to supervised fine tuning.

Since we care about resource-constrained applications, we profile our adaptation method on a microcontroller (TI MSP430). We measure its computational cost in energy and FLOP unit. MSP430 requires C code; we use [Lupe](https://github.com/mingyuan-xiang/lupe/tree/main) to convert trained PyTorch model to its optimized C representation, ready to be deployed on MSP. See the [MSP Deployment](#msp-deployment) section for details. 

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
├── run_experiment.py              # Unified experiment runner
└── deploy_to_msp.py               # Generate optimized C code for deploying trained model on MSP          
```

### Folder Details

- **`core/`**: Base experiment classes, training utilities, and loss functions shared across all experiments
- **`data_utils/`**: Scripts for downloading, preprocessing, and analyzing EEG datasets
- **`models/`**: Neural network architectures including hypernetworks, embedders, and support networks
- **`experiments/`**: Your research contributions organized by methodology (hypernetworks, contrastive learning, baselines)
- **`baselines/`**: External baseline methods ([MAPU](https://arxiv.org/html/2406.02635v2), [CLUDA](https://arxiv.org/pdf/2206.06243)) for comparison
- **`config/`**: JSON configuration files specifying hyperparameters for different experiments

## HyperNet Experiments

### Overview

[HyperNetworks](https://arxiv.org/abs/1609.09106) are small-scale neural networks that generate the weights of a larger
network (main network). We explore using hypernetworks for unsupervised adaptation. We jointly pretrain the hypernetwork and
the main network on multiple source domains. At test time, we use the hypernetwork to generate weight corrections ([LoRA](https://arxiv.org/abs/2106.09685)-style) for the main network to adapt it for the target domain. This adaptation approach
is fully unsupervised and free of iterative optimization. Adaption is done through one forward pass.

### Experiment Types

| Experiment | Description | Key Features |
|------------|-------------|--------------|
| **CrossSubjectCalibrationExperiment** | Main experiment: pretrain hypernet on multiple subjects, then calibrate on target subject | • Leave-one-subject-out validation<br>• Unsupervised calibration<br>• Varying data amounts |
| **SanityCheckExperiment** | Verify hypernet performs similarly to original network on single subject | • Single subject training<br>• Baseline comparison<br>• Architecture validation |
| **BaselineExperiment** | Train hypernet from scratch per subject (no cross-subject transfer) | • Subject-specific training<br>• No transfer learning<br>• Comparison baseline |

### Key Components

- **Embedders**: Extract subject-specific features (```EEGConformer```, ```ShallowFBCSP```, ```Conv1D```)
- **Hypernetworks**: Generate weight tensors from embeddings, currently implemented with ```torch.nn.Linear```
- **Main networks**: Base BCI models (```ShallowFBCSPNet```, ```EEGConformer```)

## SupportNet Experiments

### Overview

**SupportNet** is a few-shot adaptation framework using **contrastive learning** and **class prototype attention**. It learns generalizable subject-identity representations through contrastive learning between subjects. It then uses prototype attention mechanisms to adapt to new subject by attending over class prototypes from source subjects. We use meta learning techniques to enable better generalization.

### Experiment Types

| Experiment | Description | Key Features |
|------------|-------------|--------------|
| **ClassPrototypeAttentionExperiment** | Few-shot adaptation using labeled support sets and class prototypes | • **Class prototype attention**<br>• Episodic training<br>• Support/query sets |
| **ClassPrototypeAttentionMetaExperiment** | Meta-learning version where each subject is treated as a separate task | • **Meta-learning** approach<br>• Subject-as-task paradigm<br>• Pre-trained support encoder |
| **ContrastiveBetweenSubjectsExperiment** | Contrastive learning between subjects to learn generalizable representations | • **Contrastive learning**<br>• Cross-subject representation<br>• tSNE visualization |

### Key Components

- **Support Encoder**: Contrastively trained, extracts prototypical embeddings from support sets (frozen after pre-training)
- **Task Encoder**: Extracts embeddings from query/target data
- **Attention Mechanism**: Allows task embeddings to attend over class prototypes
- **Episodic Training**: Few-shot learning with support/query episodes
- **Contrastive Loss**: Learns to distinguish between different subjects while maintaining class structure

### Meta-Learning Features

- **Subject-as-Task**: Each subject treated as a separate meta-learning task
- **Support Sets**: Small labeled datasets used to compute class prototypes
- **Query Sets**: Test data for few-shot evaluation
- **Attention Over Prototypes**: Dynamic weighting of class prototypes based on query similarity

## MSP Deployment

Deploy trained HypernetBCI models to MSP (Texas Instruments microcontrollers) for edge BCI applications using the [Lupe](https://github.com/mingyuan-xiang/lupe) framework.

### Setup

Lupe converts PyTorch models to optimized C code for MSP deployment:

```bash
# Clone and install Lupe (if not done during initial setup)
git clone https://github.com/mingyuan-xiang/lupe.git
cd lupe
pip install -e .
cd ..
```

### Convert Model for MSP

Use the MSP deployment script to convert trained models:

```bash
# Convert a trained HyperNet model
python deploy_to_msp.py --model-type hypernet --model-path results/hypernet_model.pth --output-dir msp_deployment/

# Convert a trained SupportNet model
python deploy_to_msp.py --model-type supportnet --model-path results/supportnet_model.pth --output-dir msp_deployment/

# Convert with custom configuration
python deploy_to_msp.py --model-type hypernet --model-path results/hypernet_model.pth --config config/msp_deployment.json --output-dir msp_deployment/
```

### MSP Deployment Features

- **ONNX Export**: Converts PyTorch models to ONNX format
- **C Code Generation**: Uses Lupe to generate optimized C code for MSP
- **Memory Optimization**: Quantization and pruning for resource-constrained deployment
- **Model Validation**: Compares outputs between PyTorch and C implementations

The deployment script handles:
- Model loading and preprocessing
- ONNX export with proper input/output shapes
- Lupe integration for C code generation
- Validation and testing of converted models