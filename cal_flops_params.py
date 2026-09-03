"""
Deep Learning Model Complexity & Inference Speed Profiler
=======================================================

Description:
    This script evaluates the computational complexity and inference speed of 
    PyTorch models. It calculates:
    1. FLOPs (Floating Point Operations): Theoretical computational cost.
    2. Parameters: Model size (number of learnable weights).
    3. FPS (Frames Per Second): Real-world inference throughput on the current hardware.

    It utilizes the 'thop' library for counting operations and standard PyTorch 
    timing utilities for FPS measurement.

Dependencies:
    pip install thop torch
"""

import torch
import time
import copy
import logging
import argparse
from typing import Tuple, Any

# ==============================================================================
# Model Import (Modify this line to import your specific model architecture)
# ==============================================================================
try:
    from Your_model import Your_model
except ImportError:
    # Fallback for demonstration if the specific file isn't present during testing
    import torchvision.models as models
    print("Warning: Custom model not found. Using ResNet18 for demonstration.")
    Your_model = models.resnet18

# External Dependency Check
try:
    from thop import profile, clever_format
except ImportError:
    raise ImportError("Library 'thop' is required. Install via: pip install thop")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_flops_and_params(model: torch.nn.Module, 
                             input_tensor: torch.Tensor) -> Tuple[str, str]:
    """
    Calculates the FLOPs and Parameters of a PyTorch model.

    Args:
        model (torch.nn.Module): The neural network model.
        input_tensor (torch.Tensor): A dummy input tensor defining the input shape.

    Returns:
        Tuple[str, str]: Formatted strings for FLOPs (e.g., "1.2G") and Params (e.g., "5.0M").
    """
    # Deepcopy the model to avoid 'thop' hooks interfering with the original model instance.
    # 'thop' adds hooks to layers to count operations, which can sometimes persist 
    # or cause issues if the model is used immediately after for inference timing.
    model_clone = copy.deepcopy(model)
    
    # Move clone to same device as input
    model_clone.to(input_tensor.device)
    model_clone.eval()

    logger.info("Profiling model complexity (FLOPs & Params)...")
    
    try:
        # 'verbose=False' suppresses the layer-by-layer printout
        flops, params = profile(model_clone, inputs=(input_tensor, ), verbose=False)
    except Exception as e:
        logger.error(f"Error during profiling: {e}")
        return "N/A", "N/A"
    finally:
        # Explicit cleanup to free GPU memory immediately
        del model_clone
        torch.cuda.empty_cache()

    # Format large numbers into readable units (K, M, G)
    flops_str, params_str = clever_format([flops, params], "%.3f")
    
    return flops_str, params_str


def measure_inference_speed(model: torch.nn.Module, 
                            input_tensor: torch.Tensor, 
                            warmup_iters: int = 10, 
                            measure_iters: int = 100) -> float:
    """
    Measures the inference speed (FPS) of the model.

    Args:
        model (torch.nn.Module): The neural network model.
        input_tensor (torch.Tensor): Dummy input tensor.
        warmup_iters (int): Number of iterations to run before timing (to warm up GPU caches).
        measure_iters (int): Number of iterations to average the timing over.

    Returns:
        float: Frames Per Second (FPS).
    """
    device = input_tensor.device
    model.eval()

    logger.info(f"Measuring inference speed (Warmup={warmup_iters}, Iters={measure_iters})...")

    # 1. Warmup Phase
    # GPU kernels need time to initialize. Running a few passes ensures 
    # we measure the steady-state performance, not initialization overhead.
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_tensor)
    
    # Synchronize needed to ensure all CUDA kernels are finished before starting timer
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 2. Timing Phase
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(input_tensor)
            
    # Synchronize again to ensure the last batch actually finished
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()

    # 3. Calculate FPS
    avg_time_per_batch = (end_time - start_time) / measure_iters
    # Note: If batch_size > 1, multiply fps by batch_size for strict 'throughput'
    fps = 1.0 / avg_time_per_batch
    
    return fps


def main():
    # ===============================
    # Configuration
    # ===============================
    # Define input resolution (Batch, Channels, Height, Width)
    INPUT_SHAPE = (1, 3, 256, 256) 
    NUM_CLASSES = 1
    WARMUP_ROUNDS = 20
    TEST_ROUNDS = 100

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    # ===============================
    # Model Initialization
    # ===============================
    # Replace arguments below with those required by your specific model
    try:
        model = Your_model(n_class=NUM_CLASSES)
    except TypeError:
        # Fallback if using the ResNet demo
        model = Your_model()
        
    model.to(device)
    model.eval()

    # Create dummy input
    input_tensor = torch.randn(INPUT_SHAPE).to(device)

    # ===============================
    # Execution
    # ===============================
    
    # 1. Calculate Complexity
    flops, params = compute_flops_and_params(model, input_tensor)

    # 2. Calculate Speed
    fps = measure_inference_speed(
        model, 
        input_tensor, 
        warmup_iters=WARMUP_ROUNDS, 
        measure_iters=TEST_ROUNDS
    )

    # ===============================
    # Report Generation
    # ===============================
    print("\n" + "="*40)
    print(f" Model Architecture Evaluation")
    print("="*40)
    print(f" Input Shape : {INPUT_SHAPE}")
    print(f" Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("-" * 40)
    print(f" Parameters  : {params}")
    print(f" FLOPs       : {flops}")
    print(f" FPS         : {fps:.2f} frames/sec")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()