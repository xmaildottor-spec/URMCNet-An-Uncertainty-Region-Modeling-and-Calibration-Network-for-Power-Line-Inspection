"""
CSV to Image Converter for Thermal/Sensor Datasets
==================================================

Description:
    This script converts 2D numerical data stored in CSV files (typically raw thermal 
    sensor data or single-channel matrices) into normalized 8-bit JPG images.
    
    The script performs Min-Max normalization per file to map the floating-point 
    or large integer range of the CSV data to the [0, 255] range required for 
    standard image formats.

Usage:
    python csv2img_converter.py --input_dir "./dataset/val/ir" --output_dir "./dataset/val/ir_jpg"

Author: [Your Name/Lab Name]
Date: 2026-01-13
License: MIT License
"""

import os
import argparse
import logging
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm  # Recommended for progress visualization (pip install tqdm)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_csv_data(file_path: str) -> np.ndarray:
    """
    Loads numerical data from a CSV file into a NumPy array.

    Args:
        file_path (str): The absolute or relative path to the CSV file.

    Returns:
        np.ndarray: A 2D NumPy array containing the raw data.
    
    Raises:
        IOError: If the file cannot be read.
    """
    try:
        # Load data assuming comma delimiter
        data = np.loadtxt(file_path, delimiter=",")
        return data
    except Exception as e:
        logger.error(f"Failed to load CSV file: {file_path}. Error: {e}")
        return None

def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """
    Normalizes a 2D array to the [0, 255] range and converts to uint8.
    
    Algorithm:
        pixel_new = (pixel - min) / (max - min) * 255
    
    Args:
        data (np.ndarray): The raw 2D numerical data.

    Returns:
        np.ndarray: The normalized image data with type uint8.
    """
    min_val = data.min()
    max_val = data.max()

    # Avoid division by zero if the image is flat (all pixels have same value)
    if max_val > min_val:
        norm_data = (data - min_val) / (max_val - min_val) * 255.0
    else:
        norm_data = np.zeros_like(data)
        logger.warning(f"Data range is zero (min={min_val}, max={max_val}). Outputting black image.")

    return norm_data.astype(np.uint8)

def process_dataset(input_dir: str, output_dir: str) -> None:
    """
    Iterates through the input directory, processes all CSV files, and saves them as images.

    Args:
        input_dir (str): Directory containing source .csv files.
        output_dir (str): Directory where .jpg files will be saved.
    """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Gather all CSV files (case insensitive search is preferred, but glob is case sensitive on Linux)
    # Using simple list comprehension for better cross-platform compatibility
    all_files = os.listdir(input_dir)
    csv_files = [f for f in all_files if f.lower().endswith('.csv')]

    if not csv_files:
        logger.error(f"No CSV files found in directory: {input_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files. Starting conversion...")

    # Use tqdm for a professional progress bar
    # If tqdm is not installed, this loop works but without the visual bar
    for file_name in tqdm(csv_files, desc="Converting"):
        src_path = os.path.join(input_dir, file_name)
        
        # 1. Load Data
        raw_data = load_csv_data(src_path)
        if raw_data is None:
            continue

        # 2. Normalize Data
        img_uint8 = normalize_to_uint8(raw_data)

        # 3. Generate Output Path
        base_name = os.path.splitext(file_name)[0]
        dst_path = os.path.join(output_dir, base_name + ".jpg")

        # 4. Save Image
        success = cv2.imwrite(dst_path, img_uint8)
        if not success:
            logger.error(f"Failed to write image to: {dst_path}")

    logger.info("Conversion task completed successfully.")

def main():
    """
    Main entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Convert CSV matrices to JPG images for Computer Vision datasets."
    )
    
    # Arguments
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True, 
        help='Path to the directory containing input CSV files.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help='Path to the directory where output JPG images will be saved.'
    )

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    process_dataset(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()