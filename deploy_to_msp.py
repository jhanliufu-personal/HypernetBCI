#!/usr/bin/env python3
"""
MSP Deployment Script for HypernetBCI Models

Converts trained HyperNet or SupportNet models to MSP-compatible C code using Lupe framework.
Handles ONNX export, model optimization, and C code generation for edge deployment.

Usage:
    python deploy_to_msp.py --model-type hypernet --model-path results/hypernet_model.pth --output-dir msp_deployment/
    python deploy_to_msp.py --model-type supportnet --model-path results/supportnet_model.pth --config config/msp_config.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

import torch
import torch.onnx
import onnx
from onnx import shape_inference

# Import HypernetBCI models
from models.HypernetBCI import HyperBCINet
from models.Supportnet import Supportnet
from models.Embedder import Conv1dEmbedder, ShallowFBCSPEmbedder, EEGConformerEmbedder
from models.Hypernet import LinearHypernet

try:
    import lupe
    from lupe import export_to_c, optimize_for_msp
except ImportError:
    print("Warning: Lupe not installed. Please install with: pip install -e lupe/")
    lupe = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MSPDeployer:
    """Handles conversion of HypernetBCI models to MSP-compatible C code."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config['model_type']
        self.model_path = Path(config['model_path'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default input shapes for different models
        self.default_shapes = {
            'hypernet': (1, 22, 1000),  # (batch, channels, time_samples)
            'supportnet': (1, 22, 1000),  # (batch, channels, time_samples)
        }
        
        self.input_shape = config.get('input_shape', self.default_shapes[self.model_type])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> torch.nn.Module:
        """Load the trained model based on model type."""
        logger.info(f"Loading {self.model_type} model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        if self.model_type == 'hypernet':
            model = self._load_hypernet_model(state_dict)
        elif self.model_type == 'supportnet':
            model = self._load_supportnet_model(state_dict)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        model.eval()
        logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters())}")
        return model
    
    def _load_hypernet_model(self, state_dict: Dict) -> HyperBCINet:
        """Load HypernetBCI model from state dict."""
        # Note: This requires the original model configuration to reconstruct the architecture
        # In practice, you would save this configuration alongside the model weights
        
        # For demonstration, using default configuration
        # You should load the actual configuration used during training
        from braindecode.models import ShallowFBCSPNet
        
        # Create primary network (this should match training configuration)
        primary_net = ShallowFBCSPNet(
            n_chans=22,
            n_classes=4,
            input_window_samples=1000,
            final_conv_length='auto'
        )
        
        # Create embedder
        embedder = EEGConformerEmbedder(
            n_chans=22,
            n_classes=4,
            input_window_samples=1000,
            embedding_dim=32,
            embedding_length=128
        )
        
        # Create hypernet
        embedding_shape = torch.Size([32, 128])
        sample_shape = torch.Size([22, 1000])
        weight_shape = primary_net.final_layer.conv_classifier.weight.shape
        hypernet = LinearHypernet(embedding_shape, weight_shape)
        
        # Create HyperBCINet
        model = HyperBCINet(
            primary_net=primary_net,
            embedder=embedder,
            embedding_shape=embedding_shape,
            sample_shape=sample_shape,
            hypernet=hypernet
        )
        
        model.load_state_dict(state_dict)
        return model
    
    def _load_supportnet_model(self, state_dict: Dict) -> Supportnet:
        """Load Supportnet model from state dict."""
        # Note: Similar to hypernet, this requires configuration to reconstruct
        from braindecode.models import ShallowFBCSPNet

        # Create encoders (this should match training configuration)
        support_encoder = EEGConformerEmbedder(
            n_chans=22,
            n_classes=4,
            input_window_samples=1000,
            embedding_dim=40,
            embedding_length=144
        )
        
        encoder = EEGConformerEmbedder(
            n_chans=22,
            n_classes=4,
            input_window_samples=1000,
            embedding_dim=40,
            embedding_length=144
        )
        
        classifier = ShallowFBCSPNet(
            n_chans=80,  # concatenated embeddings
            n_classes=4,
            input_window_samples=144,
            final_conv_length='auto'
        )
        
        model = Supportnet(support_encoder, encoder, classifier)
        model.load_state_dict(state_dict)
        return model
    
    def export_to_onnx(self, model: torch.nn.Module) -> str:
        """Export PyTorch model to ONNX format."""
        logger.info("Exporting model to ONNX format...")
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Set output path
        onnx_path = self.output_dir / f"{self.model_type}_model.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # Infer shapes
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, str(onnx_path))
        
        logger.info(f"ONNX model exported to: {onnx_path}")
        return str(onnx_path)
    
    def optimize_model(self, onnx_path: str) -> str:
        """Optimize ONNX model for MSP deployment."""
        logger.info("Optimizing model for MSP deployment...")
        
        if lupe is None:
            logger.warning("Lupe not available, skipping optimization")
            return onnx_path
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Apply MSP-specific optimizations
        optimized_model = optimize_for_msp(
            onnx_model,
            target_memory_kb=self.config.get('target_memory_kb', 512),
            quantize=self.config.get('quantize', True),
            prune_threshold=self.config.get('prune_threshold', 0.01)
        )
        
        # Save optimized model
        optimized_path = self.output_dir / f"{self.model_type}_model_optimized.onnx"
        onnx.save(optimized_model, str(optimized_path))
        
        logger.info(f"Optimized model saved to: {optimized_path}")
        return str(optimized_path)
    
    def generate_c_code(self, onnx_path: str) -> str:
        """Generate C code from ONNX model using Lupe."""
        logger.info("Generating C code using Lupe...")
        
        if lupe is None:
            raise RuntimeError("Lupe is required for C code generation")
        
        # Set output directory for C code
        c_output_dir = self.output_dir / "c_code"
        c_output_dir.mkdir(exist_ok=True)
        
        # Generate C code
        c_files = export_to_c(
            onnx_path,
            output_dir=str(c_output_dir),
            model_name=f"{self.model_type}_model",
            target_platform="msp432",
            optimize_memory=True,
            use_fixed_point=self.config.get('use_fixed_point', True)
        )
        
        logger.info(f"C code generated in: {c_output_dir}")
        return str(c_output_dir)
    
    def validate_conversion(self, model: torch.nn.Module, onnx_path: str, c_output_dir: str):
        """Validate the model conversion by comparing outputs."""
        logger.info("Validating model conversion...")
        
        # Generate test input
        test_input = torch.randn(self.input_shape).to(self.device)
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = model(test_input).cpu().numpy()
        
        # Get ONNX output
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path)
        onnx_output = ort_session.run(None, {'input': test_input.cpu().numpy()})[0]
        
        # Compare outputs
        mse = np.mean((pytorch_output - onnx_output) ** 2)
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        
        logger.info(f"Validation results:")
        logger.info(f"  MSE between PyTorch and ONNX: {mse:.8f}")
        logger.info(f"  Max absolute difference: {max_diff:.8f}")
        
        if mse < 1e-5:
            logger.info("Conversion validation passed")
        else:
            logger.warning("Conversion validation failed - large differences detected")
        
        # Save validation results
        validation_results = {
            'mse': float(mse),
            'max_diff': float(max_diff),
            'pytorch_output_shape': list(pytorch_output.shape),
            'onnx_output_shape': list(onnx_output.shape)
        }
        
        with open(self.output_dir / 'validation_results.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
    
    def generate_deployment_guide(self, c_output_dir: str):
        """Generate deployment guide and Makefile for MSP."""
        logger.info("Generating deployment guide...")
        
        guide_content = f"""
# MSP Deployment Guide for {self.model_type.upper()} Model

## Generated Files
- C source code: {c_output_dir}/
- Model weights: {c_output_dir}/model_weights.c
- Model header: {c_output_dir}/model.h
- Main inference function: {c_output_dir}/inference.c

## MSP Integration Steps

1. **Copy files to MSP project:**
   ```bash
   cp {c_output_dir}/* /path/to/msp/project/src/
   ```

2. **Include in your MSP main.c:**
   ```c
   #include "model.h"
   
   int main(void) {{
       // Initialize model
       model_init();
       
       // Prepare input data (shape: {self.input_shape})
       float input_data[{np.prod(self.input_shape)}];
       // ... fill input_data with EEG samples ...
       
       // Run inference
       float output[4];  // Assuming 4 classes
       model_inference(input_data, output);
       
       // Process output
       int predicted_class = argmax(output, 4);
       
       return 0;
   }}
   ```

3. **Compilation:**
   - Add generated C files to your MSP project
   - Ensure sufficient RAM allocation ({self.config.get('target_memory_kb', 512)}KB recommended)
   - Enable floating-point support if not using fixed-point

## Performance Considerations
- Model memory usage: ~{self.config.get('target_memory_kb', 512)}KB
- Input processing: Real-time EEG buffer management required
- Power consumption: Optimize inference frequency based on battery constraints

## Troubleshooting
- Memory issues: Reduce model size or enable quantization
- Accuracy loss: Verify input data preprocessing matches training pipeline
- Performance: Consider reducing model complexity for faster inference
"""
        
        with open(self.output_dir / 'DEPLOYMENT_GUIDE.md', 'w') as f:
            f.write(guide_content)
        
        # Generate Makefile template
        makefile_content = f"""
# MSP432 Makefile for {self.model_type.upper()} Model
CC = arm-none-eabi-gcc
MCU = msp432p401r

CFLAGS = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
CFLAGS += -std=c99 -Wall -g -Os
CFLAGS += -I./src -I./c_code

SOURCES = main.c c_code/inference.c c_code/model_weights.c
TARGET = {self.model_type}_bci

$(TARGET).elf: $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^

flash: $(TARGET).elf
	mspdebug tilib "prog $(TARGET).elf"

clean:
	rm -f $(TARGET).elf

.PHONY: flash clean
"""
        
        with open(self.output_dir / 'Makefile', 'w') as f:
            f.write(makefile_content)
        
        logger.info("Deployment guide and Makefile generated")
    
    def deploy(self):
        """Main deployment pipeline."""
        logger.info(f"Starting MSP deployment for {self.model_type} model")
        
        try:
            # Load model
            model = self.load_model()
            
            # Export to ONNX
            onnx_path = self.export_to_onnx(model)
            
            # Optimize for MSP
            optimized_onnx_path = self.optimize_model(onnx_path)
            
            # Generate C code
            c_output_dir = self.generate_c_code(optimized_onnx_path)
            
            # Validate conversion
            self.validate_conversion(model, optimized_onnx_path, c_output_dir)
            
            # Generate deployment guide
            self.generate_deployment_guide(c_output_dir)
            
            logger.info(f"MSP deployment completed successfully!")
            logger.info(f"Output directory: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Return default configuration
    return {
        'target_memory_kb': 512,
        'quantize': True,
        'prune_threshold': 0.01,
        'use_fixed_point': True
    }


def main():
    parser = argparse.ArgumentParser(description='Deploy HypernetBCI models to MSP')
    parser.add_argument('--model-type', required=True, choices=['hypernet', 'supportnet'],
                        help='Type of model to deploy')
    parser.add_argument('--model-path', required=True, help='Path to trained model file (.pth)')
    parser.add_argument('--output-dir', default='msp_deployment', help='Output directory')
    parser.add_argument('--config', help='Configuration JSON file')
    parser.add_argument('--input-shape', nargs=3, type=int, help='Input shape (batch, channels, samples)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.update({
        'model_type': args.model_type,
        'model_path': args.model_path,
        'output_dir': args.output_dir
    })
    
    if args.input_shape:
        config['input_shape'] = tuple(args.input_shape)
    
    # Create deployer and run deployment
    deployer = MSPDeployer(config)
    deployer.deploy()


if __name__ == '__main__':
    main()