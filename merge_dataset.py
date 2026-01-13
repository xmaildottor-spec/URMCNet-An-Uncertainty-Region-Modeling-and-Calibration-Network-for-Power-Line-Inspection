"""
Image Mosaic Stitcher (2x2 Grid)
================================

Description:
    This utility reconstructs larger images from smaller image patches. 
    Specifically, it takes a sequence of images and merges them in groups of 4 
    into a 2x2 grid.
    
    Logic:
        Input:  [Patch 1, Patch 2, Patch 3, Patch 4] (Each 256x256)
        Output: 
                +---------+---------+
                | Patch 1 | Patch 2 |
                +---------+---------+
                | Patch 3 | Patch 4 |
                +---------+---------+
                Result: 512x512 Combined Image

Usage:
    python image_mosaic_stitcher.py --input_dir "./data/patches" --output_dir "./data/merged"
"""

import os
import argparse
import logging
import cv2
import numpy as np
from typing import List, Optional
from tqdm import tqdm  # pip install tqdm

# Configure logging for professional output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_sorted_files(directory: str, extension: str) -> List[str]:
    """
    Retrieves and numerically sorts files with a specific extension.
    
    Args:
        directory (str): Path to search.
        extension (str): File suffix (e.g., '.jpg').
        
    Returns:
        List[str]: Sorted list of filenames.
    """
    files = [f for f in os.listdir(directory) if f.lower().endswith(extension.lower())]
    
    # Sort by the numeric value of the filename (e.g., '1.jpg' before '10.jpg')
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        logger.warning("Filenames are not purely numeric. Falling back to string sorting.")
        files.sort()
        
    return files

def stitch_2x2_grid(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Stitches a list of 4 images into a single 2x2 matrix.

    Args:
        images (List[np.ndarray]): A list containing exactly 4 image arrays.

    Returns:
        np.ndarray: The stitched image, or None if dimensions mismatch.
    """
    if len(images) != 4:
        logger.error(f"Expected 4 images, got {len(images)}")
        return None

    img1, img2, img3, img4 = images

    # Check for dimension consistency
    if not (img1.shape == img2.shape == img3.shape == img4.shape):
        logger.error("Dimension mismatch among the batch of 4 images.")
        return None

    # Construct the grid
    # Row 1: Left-Top + Right-Top
    top_row = np.hstack([img1, img2])
    # Row 2: Left-Bottom + Right-Bottom
    bottom_row = np.hstack([img3, img4])
    # Combine Rows: Top + Bottom
    merged_image = np.vstack([top_row, bottom_row])

    return merged_image

def process_stitching(input_dir: str, output_dir: str, extension: str = ".jpg") -> None:
    """
    Main processing loop to read, stitch, and save images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Gather files
    file_list = get_sorted_files(input_dir, extension)
    total_files = len(file_list)
    
    if total_files == 0:
        logger.error(f"No images found with extension {extension} in {input_dir}")
        return

    # 2. Validate count
    if total_files % 4 != 0:
        logger.error(f"Total images ({total_files}) is not divisible by 4. Stitching requires groups of 4.")
        raise ValueError("Image count mismatch.")

    num_batches = total_files // 4
    logger.info(f"Found {total_files} images. Generating {num_batches} merged images.")

    # 3. Process in batches of 4
    for i in tqdm(range(0, total_files, 4), desc="Stitching"):
        batch_files = file_list[i : i+4]
        batch_images = []
        read_success = True

        # Load images
        for fname in batch_files:
            path = os.path.join(input_dir, fname)
            img = cv2.imread(path)
            
            if img is None:
                logger.error(f"Failed to read image: {path}")
                read_success = False
                break
                
            # Optional: Warning for unexpected sizes (standard is often 256x256)
            h, w = img.shape[:2]
            if h != 256 or w != 256:
                logger.warning(f"Non-standard size detected in {fname}: {w}x{h}")
                
            batch_images.append(img)

        if not read_success:
            continue

        # Stitch images
        merged = stitch_2x2_grid(batch_images)
        
        if merged is not None:
            # Generate output name (e.g., 00000_merge.jpg)
            # Using i//4 ensures sequential indexing: 0, 1, 2...
            output_filename = f"{i // 4:05d}_merge{extension}"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, merged)
        else:
            logger.error(f"Failed to merge batch starting with {batch_files[0]}")

    logger.info("Stitching task completed.")

def main():
    parser = argparse.ArgumentParser(description="Merge patches into a 2x2 grid image.")
    
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing the image patches.')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save stitched images.')
    parser.add_argument('--ext', type=str, default='.jpg', 
                        help='Image extension to look for (default: .jpg).')

    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    process_stitching(args.input_dir, args.output_dir, args.ext)

if __name__ == "__main__":

    main()
