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




def load_images(dataset, case_id, type, method=None, rotation=1):
    """
    More general image loading function that will eventually replace load_images().
    This version returns the images as base64 encoded PNG strings for direct embedding in HTML.
    
    Args:
        dataset (str): Dataset name (e.g., "IMAGENET_TEST_1k", "INVIVO2D_SET3")
        case_id (str): Case identifier (e.g., "000001")
        type (str): Type of data to load:
            - "image_series": Returns the series of input images (e.g., multi-echo data)
            - "label": Returns the ground truth parametric maps from /labels folder
            - "prediction": Returns prediction results (requires method parameter)
        method (str, optional): Prediction method name (e.g., "FIT_NLLS"). Required when type="prediction"
        rotation (int, optional): Number of 90-degree clockwise rotations to apply (default: 1)
    
    Returns:
        dict: Dictionary containing the loaded images as base64 encoded PNG strings.
    """

    # File prefixes vary by dataset type
    if dataset.lower().startswith("synth"):
        is_synth = True
        prefix = "synth"    
        S0_vmax = 1
        T_vmax = 4
        imgseries_display_scaling = 1.2  # Display scaling after normalization
    else:
        is_synth = False
        prefix = "invivo"
        S0_vmax = 0.75
        T_vmax = 2
        imgseries_display_scaling = 3  # Display scaling after normalization

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

        # Normalize the image series
        scale_factor = 1/img3d.max()
        img3d = img3d * scale_factor * imgseries_display_scaling
        result['image_scale_factor'] = scale_factor * imgseries_display_scaling

        num_volumes = img3d.shape[2]
        for i in range(num_volumes):
            result[f'image_{i}'] = img3d[:, :, i]
            
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
        
        # Normalize for display, but save the original scaling factors
        result['label_S0'] = data[:, :, 0, 0] / S0_vmax  # S0 parameter
        result['label_T'] = data[:, :, 0, 1] / T_vmax# T parameter
        result['label_S0_scale_factor'] = 1 / S0_vmax
        result['label_T_scale_factor'] = 1 / T_vmax
 
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
        
        result['pred_S0'] = data[:, :, 0, 0]  # S0 parameter
        result['pred_T'] = data[:, :, 0, 1] if data.shape[3] > 1 else data[:, :, 0, 0]  # T parameter

                # Normalize for display, but save the original scaling factors
        result['pred_S0'] = data[:, :, 0, 0] / S0_vmax  # S0 parameter
        result['pred_T'] = data[:, :, 0, 1] / T_vmax# T parameter
        result['pred_S0_scale_factor'] = 1 / S0_vmax
        result['pred_T_scale_factor'] = 1 / T_vmax
        
    else:
        raise ValueError(f"Unknown type: {type}. Must be 'image_series', 'label', or 'prediction'")
    
    # For each image in results, normalize, rotate, and convert to base64
    base64_data = {}
    for key, array in result.items():
        # Determine scaling and colormap based on image type
        if key.endswith('_scale_factor'):
            base64_data[key] = array # Just pass through scale factors
            continue  # Skip scale factor entries

        # Colormap. Make T values 
        cmap = 'viridis' if key.endswith('_T') else 'gray'
        
        # Normalize the array to 0-255 range
        normalized = np.clip(array, 0, 1)
        normalized = (normalized * 255).astype(np.uint8)        
        
        # Apply rotation if specified (k=-rotation means clockwise)
        if rotation != 0:
            normalized = np.rot90(normalized, k=-rotation)
        
        # Convert to base64 using the helper function
        img_str = array_to_base64(normalized, cmap=cmap, vmin=0, vmax=255)
        
        base64_data[key] = img_str
    
    return base64_data
    

def array_to_base64(array, cmap='gray', vmin=0, vmax=255):
    """Convert a 2D numpy array to a base64 encoded PNG string.
    Uses Matplotlib to render the image.
    
    Args:
        array: 2D numpy array to convert
        cmap: Colormap to use for the image (default: 'gray')
        vmin: Minimum value for color scaling (default: 0)
        vmax: Maximum value for color scaling (default: 255)
    
    Returns:
        str: Base64 encoded PNG image string
    """
    matplotlib.use('Agg')  # Use non-interactive backend

    # Create figure for this image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(array.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_str


@app.route('/')
def index():
    return render_template('index.html'


@app.route('/explore_datasets')
def explore_datasets():
    # Load all three sample datasets
    
    # 1. ImageNet dataset
    case_id = '000066'
    dataset_name = 'synth_imagenet_1k_train'
    
    # Load image series and labels for ImageNet
    imagenet_series = load_images_2(dataset_name, case_id, "image_series", rotation=1)
    imagenet_labels = load_images_2(dataset_name, case_id, "label", rotation=1)
    
    # Combine ImageNet data
    if imagenet_series is not None and imagenet_labels is not None:
        imagenet_images = {**imagenet_series, **imagenet_labels}
        print(f'ImageNet data keys: {list(imagenet_images.keys())}')
    else:
        print("ImageNet dataset images not found")
        imagenet_images = None
    
    # 2. URAND dataset
    dataset_name = 'synth_urand_1k_train'
    
    # Load image series and labels for URAND
    urand_series = load_images_2(dataset_name, case_id, "image_series", rotation=1)
    urand_labels = load_images_2(dataset_name, case_id, "label", rotation=1)
    
    # Combine URAND data
    if urand_series is not None and urand_labels is not None:
        urand_images = {**urand_series, **urand_labels}
        print(f'URAND data keys: {list(urand_images.keys())}')
    else:
        print("URAND dataset images not found")
        urand_images = None
    
    # 3. InVivo dataset
    case_id = '000198'
    dataset_name = 'invivo2D_set1'
    
    # Load image series for InVivo
    invivo_series = load_images_2(dataset_name, case_id, "image_series", rotation=0)
    
    # Not actually labels for invivo, but predictions from FIT_NLLS
    invivo_preds = load_images_2(dataset_name.upper(), case_id, type="prediction", method="FIT_NLLS", rotation=0)
    
    # The html template expects 'label_' keys. Rename the preds to labels
    invivo_labels = {key.replace('pred_', 'label_'): value for key, value in invivo_preds.items()}
    
    # Combine InVivo data
    if invivo_series is not None and invivo_labels is not None:
        invivo_images = {**invivo_series, **invivo_labels}
        print(f'InVivo data keys: {list(invivo_images.keys())}')
    else:
        print("InVivo dataset images not found")
        invivo_images = None

    # Display only the invivo images
    images = invivo_images

    # Return also the echo time values. Hardwired
    TE_vals = [26.4 + i * 13.2 for i in range(10)]   

    return render_template('explore_datasets.html', 
                          images=images, 
                          imagenet_images=imagenet_images,
                          urand_images=urand_images,
                          invivo_images=invivo_images,
                          dataset_type='invivo', 
                          TE_vals=TE_vals)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
