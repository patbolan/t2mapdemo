#!/usr/bin/env python3
"""
File Copy Script for T2Map Demo

I have the full set of infered data files in a separate folder (prostate_t2map_modified/predictions/).
This demo won't have access to that folder, so I will copy over files as needed. 
This script is a helper to check the files, and then copy them  over to the demo folder structure.

"""

import os
import shutil
import nibabel as nib
import matplotlib.pyplot as plt 
import numpy as np

def main():


    # Copy files from source to target areas
    source_root = "/home/pbolan/prj/prostate_t2map_modified/predictions/"
    target_root = "/home/pbolan/prj/t2mapdemo/static/predictions/"

    # A list of dataset Folders, [INVIVO2D_SET3, INVIVO2D_SET3_NOISE_1, INVIVO2D_SET3_NOISE_2, ..., INVIVO2D_SET3_NOISE_9]
    dataset_folders = ["INVIVO2D_SET3"] + [f"INVIVO2D_SET3_NOISE_{i}" for i in range(1, 10)]    

    # LIst of methods
    methods = ["FIT_NLLS", "CNN_IMAGENET"]  

    case_id = '000118'
    case_id = '000277'
    #case_id = '000400'

    # Preview a single file
    if False:
        test_file = os.path.join(source_root, "INVIVO2D_SET3", "FIT_NLLS", f"preds_{case_id}.nii.gz")
        if os.path.exists(test_file):
            # load this nifti file and plot an image

            nifti_img = nib.load(test_file)
            img_data = nifti_img.get_fdata()
            img = np.rot90(img_data[:,:,0,1])
            plt.imshow(img, cmap='viridis', vmax=2, interpolation='nearest')
            plt.show()

    # Copy everything
    if False:
        for dataset in dataset_folders:
            for method in methods:
                source_file = os.path.join(source_root, dataset, method, f"preds_{case_id}.nii.gz")
                destination_dir = os.path.join(target_root, dataset, method)

                # First, just check that the source file exits
                if not os.path.exists(source_file):
                    print(f"Source file does not exist: {source_file}")
                    # Abort further processing for this file    
                    raise FileNotFoundError(f"Source file does not exist: {source_file}")   
                
                # Check if the destination directory exists
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir, exist_ok=True)
                    print(f"Created destination directory: {destination_dir}")
                else:
                    print(f"Destination directory already exists: {destination_dir}")

                # Check if the file exists at the destination
                dest_file = os.path.join(destination_dir, f"preds_{case_id}.nii.gz")
                if os.path.exists(dest_file):
                    print(f"Destination file already exists, skipping copy: {dest_file}")
                    continue  # Skip copying if the file already exists

                # Copy the file using shutil
                shutil.copy2(source_file, dest_file)
                print(f"Copied {source_file} to {dest_file}")

if __name__ == "__main__":
    main()