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


def get_available_cases():
    """Get list of available case IDs from the synth images directory."""
    images_dir = "static/datasets/synth_imagenet_1k_test/images"
    if not os.path.exists(images_dir):
        return []
    
    cases = set()
    for filename in os.listdir(images_dir):
        if filename.startswith('synth_') and filename.endswith('.nii.gz'):
            # Extract case ID from filename (e.g., 'synth_000012.nii.gz' -> '000012')
            case_id = filename.replace('synth_', '').replace('.nii.gz', '')
            cases.add(case_id)
    
    return sorted(list(cases))


def get_prediction_methods():
    """Get list of available prediction methods from the predictions directory."""
    preds_base_dir = "static/predictions/IMAGENET_TEST_1k"
    if not os.path.exists(preds_base_dir):
        return []
    
    methods = []
    for item in os.listdir(preds_base_dir):
        item_path = os.path.join(preds_base_dir, item)
        if os.path.isdir(item_path):
            methods.append(item)
    
    return sorted(methods)


def load_images(case_id, prediction_method='FIT_NLLS', reference_method='Ground_Truth'):
    """Load and return the four parametric images as numpy arrays in a dictionary."""
    
    # Define base directories
    images_dir = "static/datasets/synth_imagenet_1k_test/images"
    labels_dir = "static/datasets/synth_imagenet_1k_test/labels"
    preds_base_dir = "static/predictions/IMAGENET_TEST_1k"
    
    synth_file = os.path.join(images_dir, f"synth_{case_id}.nii.gz")
    
    # Determine reference file path
    if reference_method == 'Ground_Truth':
        ref_file = os.path.join(labels_dir, f"synth_{case_id}.nii.gz")
    else:
        ref_file = os.path.join(preds_base_dir, reference_method, f"preds_{case_id}.nii.gz")
    
    # Determine prediction file path
    preds_file = os.path.join(preds_base_dir, prediction_method, f"preds_{case_id}.nii.gz")

    # Check if files exist
    for file_path in [synth_file, ref_file, preds_file]:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
    
    # Load the NIfTI files
    synth_img = nib.load(synth_file)
    ref_img = nib.load(ref_file)
    preds_img = nib.load(preds_file)
    
    # Get the data arrays
    synth_data = synth_img.get_fdata()
    ref_data = ref_img.get_fdata()
    preds_data = preds_img.get_fdata()
    
    # Assuming 4D data (x, y, z, parameters) - take middle slice
    if len(ref_data.shape) == 4:
        middle_slice = ref_data.shape[2] // 2
        label_S0 = ref_data[:, :, middle_slice, 0]  # S0 parameter
        label_T = ref_data[:, :, middle_slice, 1] if ref_data.shape[3] > 1 else ref_data[:, :, middle_slice, 0]  # T parameter
        pred_S0 = preds_data[:, :, middle_slice, 0]    # S0 parameter
        pred_T = preds_data[:, :, middle_slice, 1] if preds_data.shape[3] > 1 else preds_data[:, :, middle_slice, 0]    # T parameter
        
        # Extract all synth volumes (assuming synth has multiple TEs in 4th dimension)
        synth_images = []
        if len(synth_data.shape) == 4:
            num_volumes = synth_data.shape[3]
            for i in range(num_volumes):
                synth_images.append(synth_data[:, :, middle_slice, i])
        else:
            synth_images.append(synth_data[:, :, middle_slice])
    else:
        # If 3D, just take middle slice
        middle_slice = ref_data.shape[2] // 2
        label_S0 = ref_data[:, :, middle_slice]
        label_T = label_S0  # Same slice if only 3D
        pred_S0 = preds_data[:, :, middle_slice]
        pred_T = pred_S0
        synth_images = [synth_data[:, :, middle_slice]]
    
    result = {
        'label_S0': label_S0,
        'label_T': label_T,
        'pred_S0': pred_S0,
        'pred_T': pred_T,
    }
    
    # Add synth images with indexed keys
    for i, synth_img in enumerate(synth_images):
        result[f'synth_{i}'] = synth_img
    
    return result



def load_invivo_images(case_id, prediction_method='FIT_NLLS'):
    """Load and return invivo images for all noise levels (0-9) including original and predicted parametric maps.
    
    Returns:
        dict: Dictionary with keys 0-9, each containing {'pred_S0': array, 'pred_T': array, 'orig_0': array, ...}
              Original images (orig_*) are the same for all noise levels.
    """
    
    # Define base directories
    images_dir = "static/datasets/invivo2D_set3/images"
    
    # Define original images file path
    orig_file = os.path.join(images_dir, f"invivo_{case_id}.nii.gz")
    
    # Check if original file exists
    if not os.path.exists(orig_file):
        print(f"File not found: {orig_file}")
        return None
    
    # Load the original images once
    orig_img = nib.load(orig_file)
    orig_data = orig_img.get_fdata()
    
    # Extract original images from middle slice
    middle_slice = orig_data.shape[2] // 2 if len(orig_data.shape) >= 3 else 0
    orig_images = []
    if len(orig_data.shape) == 4:
        num_volumes = orig_data.shape[3]
        for i in range(num_volumes):
            orig_images.append(orig_data[:, :, middle_slice, i])
    else:
        orig_images.append(orig_data[:, :, middle_slice])
    
    # Dictionary to store all noise levels
    all_noise_levels = {}
    
    # Loop through all noise levels (0-9)
    for noise_level in range(10):
        # Handle noise_level = 0 (no noise suffix) vs noise_level > 0
        if noise_level == 0:
            preds_base_dir = "static/predictions/INVIVO2D_SET3"
        else:
            preds_base_dir = f"static/predictions/INVIVO2D_SET3_NOISE_{noise_level}"
        
        # Define predictions file path
        preds_file = os.path.join(preds_base_dir, prediction_method, f"preds_{case_id}.nii.gz")
        
        # Check if predictions file exists
        if not os.path.exists(preds_file):
            print(f"File not found: {preds_file}")
            all_noise_levels[noise_level] = None
            continue
        
        # Load the predictions
        preds_img = nib.load(preds_file)
        preds_data = preds_img.get_fdata()
        
        # Extract predicted parameters from middle slice
        if len(preds_data.shape) == 4:
            pred_S0 = preds_data[:, :, middle_slice, 0]    # S0 parameter
            pred_T = preds_data[:, :, middle_slice, 1] if preds_data.shape[3] > 1 else preds_data[:, :, middle_slice, 0]    # T parameter
        else:
            # If 3D, just take middle slice
            pred_S0 = preds_data[:, :, middle_slice]
            pred_T = pred_S0
        
        # Create result dictionary for this noise level
        result = {
            'pred_S0': pred_S0,
            'pred_T': pred_T,
        }
        
        # Add original images with indexed keys (same for all noise levels)
        for i, orig_img in enumerate(orig_images):
            result[f'orig_{i}'] = orig_img
        
        all_noise_levels[noise_level] = result
    
    return all_noise_levels


def load_images_2(dataset, case_id, type, method=None, rotation=1):
    """More general image loading function that will eventually replace load_images().
    
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


def images_to_base64(images, rotation=1, t_max=4):
    """Convert the parametric images to base64 encoded PNGs.
    
    Args:
        images: Dictionary of image arrays to convert
        rotation: Number of 90-degree clockwise rotations to apply (default=1)
        t_max: Maximum value for T parameter scaling (default=4)
    """
    matplotlib.use('Agg')  # Use non-interactive backend
    
    image_data = {}
    
    # Define scaling for each parameter
    s0_vmin, s0_vmax = 0, 1
    t_vmin, t_vmax = 0, t_max
    
    for key, array in images.items():
        # Determine scaling and colormap based on image type
        if 'image' in key:
            # Synth images: use same scaling as S0 (0-1) and grayscale
            vmin, vmax = s0_vmin, s0_vmax
            cmap = 'gray'
        elif 'S0' in key:
            vmin, vmax = s0_vmin, s0_vmax
            cmap = 'gray'
        else:  # 'T' parameter
            vmin, vmax = t_vmin, t_vmax
            cmap = 'viridis'
        
        # Normalize the array to 0-255 range
        normalized = np.clip((array - vmin) / (vmax - vmin), 0, 1)
        normalized = (normalized * 255).astype(np.uint8)
        
        # Apply rotation if specified (k=-rotation means clockwise)
        if rotation != 0:
            normalized = np.rot90(normalized, k=-rotation)
        
        # Convert to base64 using the new function
        img_str = array_to_base64(normalized, cmap=cmap, vmin=0, vmax=255)
        
        image_data[key] = img_str
    
    return image_data


@app.route('/')
def index():
    cases = get_available_cases()
    return render_template('index.html', cases=cases)


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


@app.route('/invivo_datasets')
def invivo_datasets():
    
    # All prediction methods to display
    valid_methods = ['FIT_NLLS', 'CNN_IMAGENET', 'NN1D_URAND']
    
    # Load invivo example: case 000118
    case_id = '000118'
    
    # Load images for all prediction methods and all noise levels
    all_methods_images = {}
    
    for method in valid_methods:
        # Load images for all noise levels (0-9) at once
        all_noise_levels = load_invivo_images(case_id, method)
        
        # Convert images to base64 for web display (no rotation, t_max=2 for invivo)
        method_images = {}
        if all_noise_levels is not None:
            for noise_level, images in all_noise_levels.items():
                if images is not None:
                    image_data = images_to_base64(images, rotation=0, t_max=2)
                    method_images[noise_level] = image_data
                else:
                    print(f"Invivo images not found for case: {case_id}, method: {method}, noise_level: {noise_level}")
                    method_images[noise_level] = None
        else:
            print(f"Failed to load invivo images for case: {case_id}, method: {method}")
        
        all_methods_images[method] = method_images
    
    return render_template('invivo_datasets.html', 
                          all_methods_images=all_methods_images,
                          methods=valid_methods)


@app.route('/case/<case_id>')
def view_case(case_id):
    # Get prediction method from query parameter, default to FIT_NLLS
    prediction_method = request.args.get('method', 'FIT_NLLS')
    reference_method = request.args.get('reference', 'Ground_Truth')
    
    # Validate prediction method
    available_methods = get_prediction_methods()
    if prediction_method not in available_methods:
        prediction_method = 'FIT_NLLS'  # Fallback to default
    
    # Validate reference method - can be 'Ground_Truth' or any prediction method
    reference_options = ['Ground_Truth'] + available_methods
    if reference_method not in reference_options:
        reference_method = 'Ground_Truth'  # Fallback to default
    
    images = load_images(case_id, prediction_method, reference_method)
    if images is None:
        abort(404)
    
    # Get all available cases for navigation
    all_cases = get_available_cases()
    
    # Find current case position and determine previous/next
    try:
        current_index = all_cases.index(case_id)
        prev_case = all_cases[current_index - 1] if current_index > 0 else None
        next_case = all_cases[current_index + 1] if current_index < len(all_cases) - 1 else None
    except ValueError:
        # Case ID not found in the list
        prev_case = None
        next_case = None
    
    # Convert images to base64 for web display (with 1 CW rotation)
    image_data = images_to_base64(images, rotation=1)

    return render_template('case.html', 
                         case_id=case_id, 
                         images=image_data,
                         prev_case=prev_case,
                         next_case=next_case,
                         current_position=current_index + 1 if case_id in all_cases else 0,
                         total_cases=len(all_cases),
                         prediction_methods=available_methods,
                         selected_method=prediction_method,
                         reference_options=reference_options,
                         selected_reference=reference_method)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
