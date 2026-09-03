# coding: utf-8
"""
Dataset Preparation Utility: Image Patch Stitching
This script merges batches of four adjacent 256x256 image patches into single 512x512 high-resolution images.
Designed for preparing spatially consistent multi-modal (e.g., Thermal Infrared) dataset inputs for deep learning frameworks.
"""

import os
import cv2
import numpy as np
import argparse

def stitch_image_patches(input_dir: str, output_dir: str) -> None:
    """
    Reads supported image patches from the input directory, sorts them numerically,
    and stitches every four consecutive patches into a 2x2 grid.
    
    Args:
        input_dir (str): Directory containing the source image patches.
        output_dir (str): Directory where the stitched high-resolution images will be saved.
    """
    # Supported image formats
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    
    os.makedirs(output_dir, exist_ok=True)

    # Retrieve and sort files numerically (e.g., "0000.png" -> 0) to ensure spatial continuity
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    total = len(files)

    if total == 0:
        print(f"No supported image files found in {input_dir}. Please check the path.")
        return

    # Verify that the total number of images forms complete 4-patch groups
    if total % 4 != 0:
        raise ValueError(f"The total number of images is not a multiple of 4. Current count: {total}")

    print(f"Detected {total} images. Generating {total // 4} stitched high-resolution images...")

    # Process images in batches of 4
    for i in range(0, total, 4):
        batch_files = files[i:i+4]
        img_paths = [os.path.join(input_dir, f) for f in batch_files]

        images = [cv2.imread(path) for path in img_paths]

        if any(img is None for img in images):
            raise RuntimeError(f"Failed to read one or more images. Please check for file corruption: {batch_files}")

        # Verify spatial dimension consistency across the batch
        base_h, base_w, _ = images[0].shape
        for img in images[1:]:
            if img.shape != images[0].shape:
                raise ValueError(f"Inconsistent dimensions detected within batch: {batch_files}")

        if base_h != 256 or base_w != 256:
            print(f"Warning: Non-standard 256x256 dimensions detected in {batch_files[0]}: {base_w}x{base_h}")

        # Stitching: Assemble the 2x2 grid
        top_row = np.hstack([images[0], images[1]])     # Top-Left + Top-Right
        bottom_row = np.hstack([images[2], images[3]])  # Bottom-Left + Bottom-Right
        merged_img = np.vstack([top_row, bottom_row])   # Vertical stacking -> 512x512

        # Output filename generation and saving
        out_name = f"{i // 4:05d}_merge.jpg"
        out_path = os.path.join(output_dir, out_name)

        cv2.imwrite(out_path, merged_img)
        print(f"Generated: {out_name} (Source format: {os.path.splitext(batch_files[0])[1]})")

    print("All stitching operations completed successfully.")

if __name__ == "__main__":
    # Command-line argument parsing replaces hardcoded local paths for open-source flexibility
    parser = argparse.ArgumentParser(description="Stitch adjacent 256x256 image patches into 512x512 images for multi-modal segmentation datasets.")
    parser.add_argument('--input_dir', type=str, default='./dataset/train/ir_img_gray', 
                        help='Path to the directory containing source image patches.')
    parser.add_argument('--output_dir', type=str, default='./dataset/train/ir_img_merge', 
                        help='Path to the directory for saving the stitched images.')
    
    args = parser.parse_args()
    
    stitch_image_patches(args.input_dir, args.output_dir)