#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Example calibration data loader for INT8 quantization in TensorRT.

This script demonstrates various ways to load calibration data for INT8 quantization.
Modify the load_data() function to match your specific use case.
"""

import numpy as np
import glob
import os


def load_data():
    """
    Generator function that yields calibration data.
    
    This function should be modified to load your actual calibration dataset.
    It should yield dictionaries mapping input names to numpy arrays.
    
    Yields:
        dict: Dictionary mapping input tensor names to numpy arrays
    """
    # Example 1: Generate random data (for testing only)
    # Replace this with actual data loading logic
    
    num_calibration_samples = 100  # Use 500-1000 in practice
    
    for i in range(num_calibration_samples):
        # Generate random data matching your model's input shape
        # Replace with actual data loading
        data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        # Yield as dictionary mapping input names to arrays
        # Replace "input" with your actual input tensor name
        yield {"input": data}


def load_data_from_images():
    """
    Example: Load calibration data from image files.
    
    This demonstrates how to load and preprocess images for calibration.
    """
    # Path to calibration images
    image_dir = "/path/to/calibration/images"
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    # Limit to desired number of calibration samples
    image_files = image_files[:1000]
    
    for img_path in image_files:
        try:
            # Load image using PIL or OpenCV
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            
            # Resize to model input size
            img = img.resize((224, 224))
            
            # Convert to numpy array
            img_array = np.array(img).astype(np.float32)
            
            # Normalize (adjust based on your model's requirements)
            # Common normalizations:
            # 1. Scale to [0, 1]: img_array = img_array / 255.0
            # 2. ImageNet normalization:
            mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
            std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)
            img_array = (img_array - mean) / std
            
            # Convert from HWC to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Yield the preprocessed image
            yield {"input": img_array}
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue


def load_data_from_numpy():
    """
    Example: Load calibration data from saved numpy files.
    
    This is useful if you've preprocessed and saved your calibration data.
    """
    # Path to saved numpy files
    data_dir = "/path/to/calibration/numpy_files"
    data_files = glob.glob(os.path.join(data_dir, "*.npy"))
    
    for data_path in data_files[:1000]:
        try:
            # Load numpy array
            data = np.load(data_path)
            
            # Ensure correct shape (add batch dimension if needed)
            if len(data.shape) == 3:  # CHW format
                data = np.expand_dims(data, axis=0)  # Add batch dimension
            
            # Ensure correct dtype
            data = data.astype(np.float32)
            
            yield {"input": data}
            
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            continue


def load_data_from_dataset():
    """
    Example: Load calibration data from a dataset using a data loader.
    
    This demonstrates integration with PyTorch or TensorFlow datasets.
    """
    try:
        # Example using PyTorch
        import torch
        from torchvision import datasets, transforms
        
        # Define preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # Load dataset
        dataset = datasets.ImageFolder(
            root="/path/to/calibration/dataset",
            transform=transform
        )
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4
        )
        
        # Yield calibration samples
        for i, (images, _) in enumerate(data_loader):
            if i >= 1000:  # Limit to 1000 samples
                break
            
            # Convert PyTorch tensor to numpy
            img_array = images.numpy()
            
            yield {"input": img_array}
            
    except ImportError:
        print("PyTorch not available. Install with: pip install torch torchvision")
        return


def load_data_multi_input():
    """
    Example: Load calibration data for models with multiple inputs.
    
    This demonstrates how to handle models with multiple input tensors.
    """
    num_calibration_samples = 100
    
    for i in range(num_calibration_samples):
        # Generate or load data for each input
        input1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
        input2 = np.random.rand(1, 1, 224, 224).astype(np.float32)
        
        # Yield dictionary with all inputs
        yield {
            "input1": input1,
            "input2": input2,
        }


def load_data_segmentation():
    """
    Example: Load calibration data for segmentation models.
    
    This demonstrates preprocessing for semantic segmentation models.
    """
    image_dir = "/path/to/calibration/images"
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))[:1000]
    
    for img_path in image_files:
        try:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            
            # Resize to model input size
            img = img.resize((512, 512))  # Common size for segmentation
            
            # Convert to numpy and normalize
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Convert to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            yield {"input": img_array}
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue


# Default function used by Polygraphy
# Modify this to use one of the above functions or implement your own
def load_data_default():
    """
    Default calibration data loader.
    
    Modify this function to load your actual calibration data.
    """
    # For demonstration, use random data
    # In practice, replace this with load_data_from_images() or similar
    return load_data()


# Polygraphy will call this function by default
# You can also specify a different function using:
# --data-loader-script calibration_data_loader.py:load_data_from_images
load_data = load_data_default


if __name__ == "__main__":
    # Test the data loader
    print("Testing calibration data loader...")
    
    count = 0
    for data in load_data():
        count += 1
        print(f"Sample {count}:")
        for name, array in data.items():
            print(f"  {name}: shape={array.shape}, dtype={array.dtype}, "
                  f"min={array.min():.4f}, max={array.max():.4f}, mean={array.mean():.4f}")
        
        if count >= 5:  # Show first 5 samples
            break
    
    print(f"\nSuccessfully loaded {count} calibration samples.")
