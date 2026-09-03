import os
import glob
import numpy as np
import pandas as pd
import cv2

def convert_csv_to_images_without_seams(csv_dir, output_dir, group_size=4, apply_colormap=True):
    """
    Reads infrared CSV data in groups and applies global min-max normalization 
    to eliminate stitching seams.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Retrieve and strictly sort CSV files to ensure that 4 consecutive images belong to the same frame
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}.")
        return

    total_files = len(csv_files)
    if total_files % group_size != 0:
        print(f"Warning: The total number of files ({total_files}) is not a multiple of {group_size}. The remaining files will be skipped.")

    print(f"Found {total_files} files. Starting global normalization conversion in groups of {group_size}...")

    # Iterate through files with a step size equal to group_size (4)
    for i in range(0, total_files - total_files % group_size, group_size):
        batch_files = csv_files[i : i + group_size]
        batch_data = []
        
        # 1. Read all CSV data for the current group
        for file_path in batch_files:
            df = pd.read_csv(file_path, header=None)
            df = df.dropna(axis=1, how='all')
            batch_data.append(df.values)
            
        # 2. Calculate the [Global Minimum] and [Global Maximum] across the 4 images in the current group
        global_min = min([np.min(data) for data in batch_data])
        global_max = max([np.max(data) for data in batch_data])
        
        # 3. Normalize uniformly using the global extrema and output the images
        for j, data in enumerate(batch_data):
            file_name = os.path.basename(batch_files[j])
            img_name = file_name.replace('.csv', '.png')
            output_path = os.path.join(output_dir, img_name)

            if global_max == global_min:
                normalized_data = np.zeros_like(data, dtype=np.uint8)
            else:
                # Core modification: Use global_min and global_max instead of local data_min and data_max
                normalized_data = ((data - global_min) / (global_max - global_min) * 255.0).astype(np.uint8)

            if apply_colormap:
                visual_img = cv2.applyColorMap(normalized_data, cv2.COLORMAP_INFERNO)
            else:
                visual_img = normalized_data

            cv2.imwrite(output_path, visual_img)
            
        print(f"Completed image conversion for group {i//group_size + 1}.")

    print("Data conversion completed for all full groups.")

# Example Usage
if __name__ == "__main__":
    # TODO: Replace with your actual dataset paths before deployment
    CSV_FOLDER = r"./data/val/ir_csv" 
    OUTPUT_FOLDER = r"./data/val/ir_img_gray"
    
    # Execute the conversion. This ensures the 4 output images can be fed into existing 
    # stitching code without producing color/intensity discontinuities.
    convert_csv_to_images_without_seams(CSV_FOLDER, OUTPUT_FOLDER, group_size=4, apply_colormap=False)