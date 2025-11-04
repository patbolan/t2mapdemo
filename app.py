from flask import Flask, render_template, abort, request
import platform
import time
import shutil
import os
import socket
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import glob
import psutil

app = Flask(__name__)
start_time = time.time()




def load_images(dataset, case_id, type, method=None):
    """
    More general image loading function that will eventually replace load_images().
    This version returns the images as base64 encoded PNG strings for direct embedding in HTML.
    This code handles the rotations for the prostate data
    
    Args:
        dataset (str): Dataset name (e.g., "IMAGENET_TEST_1k", "INVIVO2D_SET3")
        case_id (str): Case identifier (e.g., "000001")
        type (str): Type of data to load:
            - "image_series": Returns the series of input images (e.g., multi-echo data)
            - "label": Returns the ground truth parametric maps from /labels folder
            - "prediction": Returns prediction results (requires method parameter)
        method (str, optional): Prediction method name (e.g., "FIT_NLLS"). Required when type="prediction"
    
    Returns:
        dict: Dictionary containing the loaded images as ?????
    """

    # File prefixes vary by dataset type
    if dataset.lower().startswith("synth") or dataset.lower().startswith("imagenet") or dataset.lower().startswith("urand"):
        is_synth = True
        prefix = "synth"    
        rotations = 0
    else:
        is_synth = False
        prefix = "invivo"
        rotations = 1

    result = {}
    if type == "image_series":
        images_dir = f"static/datasets/{dataset}/images"
        image_file = os.path.join(images_dir, f"{prefix}_{case_id}.nii.gz")
        
        if not os.path.exists(image_file):
            print(f"Image series file not found: {image_file}")
            return None
        
        # Load the NIfTI file
        img = nib.load(image_file)
        img3d = img.get_fdata()

        img3d = img3d[:,:,0,:].squeeze()

        if not is_synth:
            # Normalize invivo data to [0,1]
            img3d = img3d / np.max(img3d) 

        num_volumes = img3d.shape[2]
        for i in range(num_volumes):
            result[f'image_{i}'] = np.rot90(img3d[:, :, i], k=rotations)

    elif type == "label":
        # Load ground truth parametric maps from labels folder
        labels_dir = f"static/datasets/{dataset}/labels"
        label_file = os.path.join(labels_dir, f"{prefix}_{case_id}.nii.gz")
        
        if not os.path.exists(label_file):
            print(f"Label file not found: {label_file}")
            return None
        
        print(f'Loading label file: {label_file}')

        # Load the NIfTI file
        img = nib.load(label_file)
        data = img.get_fdata()
        
        result['label_S0'] = np.rot90(data[:, :, 0, 0], k=rotations) # S0 parameter
        result['label_T'] = np.rot90(data[:, :, 0, 1], k=rotations) # T parameter

 
    elif type == "prediction":
        # Load prediction results
        if method is None:
            raise ValueError("Method parameter is required when type='prediction'")
        
        preds_base_dir = f"static/predictions/{dataset}"        
        pred_file = os.path.join(preds_base_dir, method, f"preds_{case_id}.nii.gz")
        
        if not os.path.exists(pred_file):
            print(f"Prediction file not found: {pred_file}")
            return None
        
        # Load the NIfTI file
        img = nib.load(pred_file)
        data = img.get_fdata()
        
        result['pred_S0'] = np.rot90(data[:, :, 0, 0], k=rotations)  # S0 parameter
        result['pred_T'] = np.rot90(data[:, :, 0, 1], k=rotations) 
        
    else:
        raise ValueError(f"Unknown type: {type}. Must be 'image_series', 'label', or 'prediction'")

    # Convert numpy arrays to JSON-serializable lists
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()

    return result
    



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare_methods')
def compare_methods():

    # Hardwire the case_id for now
    case_id = '000001'
    dataset_name = 'synth_imagenet_1k_test'
    labels = load_images('synth_imagenet_1k_test', case_id, "label")
    method1_preds = load_images('IMAGENET_TEST_1k', case_id, "prediction", method="FIT_NLLS")
    method2_preds = load_images('IMAGENET_TEST_1k', case_id, "prediction", method="CNN_IMAGENET")

    # Check if any of the loads failed
    if labels is None or method1_preds is None or method2_preds is None:
        abort(404, description="One or more required datasets not found.")

    return render_template('compare_methods.html', labels=labels, method1_preds=method1_preds, method2_preds=method2_preds)



@app.route('/explore_datasets')
def explore_datasets():
    # Load all three sample datasets
    
    ################################################
    # 1. ImageNet dataset
    case_id = '000066'
    dataset_name = 'synth_imagenet_1k_train'
    
    # Load image series and labels for ImageNet
    imagenet_series = load_images(dataset_name, case_id, "image_series")
    imagenet_labels = load_images(dataset_name, case_id, "label")
    
    # Combine ImageNet data
    if imagenet_series is not None and imagenet_labels is not None:
        imagenet_imageset = {**imagenet_series, **imagenet_labels}
        print(f'ImageNet data keys: {list(imagenet_imageset.keys())}')
    else:
        print("ImageNet dataset images not found")
        imagenet_imageset = None
    
    ################################################
    # 2. URAND dataset
    dataset_name = 'synth_urand_1k_train'
    
    # Load image series and labels for URAND
    urand_series = load_images(dataset_name, case_id, "image_series")
    urand_labels = load_images(dataset_name, case_id, "label")
    
    # Combine URAND data
    if urand_series is not None and urand_labels is not None:
        urand_imageset = {**urand_series, **urand_labels}
        print(f'URAND data keys: {list(urand_imageset.keys())}')
    else:
        print("URAND dataset images not found")
        urand_imageset = None
    
    ################################################
    # 3. InVivo dataset
    case_id = '000198'
    dataset_name = 'invivo2D_set1'
    
    # Load image series for InVivo
    invivo_series = load_images(dataset_name, case_id, "image_series")
    
    
    # Not actually labels for invivo, but predictions from FIT_NLLS
    invivo_preds = load_images(dataset_name.upper(), case_id, type="prediction", method="FIT_NLLS")
    
    # The html template expects 'label_' keys. Rename the preds to labels
    invivo_labels = {key.replace('pred_', 'label_'): value for key, value in invivo_preds.items()}
    
    # Combine InVivo data
    if invivo_series is not None and invivo_labels is not None:
        invivo_imageset = {**invivo_series, **invivo_labels}
        print(f'InVivo data keys: {list(invivo_imageset.keys())}')
    else:
        print("InVivo dataset images not found")
        invivo_imageset = None

    # Return also the echo time values. Hardwired
    TE_vals = [26.4 + i * 13.2 for i in range(10)]   

    return render_template('explore_datasets.html', 
                          imagenet_imageset=imagenet_imageset,
                          urand_imageset=urand_imageset,
                          invivo_imageset=invivo_imageset, 
                          TE_vals=TE_vals)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
