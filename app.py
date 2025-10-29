from flask import Flask, render_template, abort
import platform
import time
import shutil
import os
import socket
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

try:
    import psutil
except ImportError:
    psutil = None

app = Flask(__name__)
start_time = time.time()



def get_available_cases():
    """Get list of available case IDs from the synth images directory."""
    images_dir = "/home/pbolan/prj/prostate_t2map_modified/datasets/synth_imagenet_1k_test/images"
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
    preds_base_dir = "/home/pbolan/prj/prostate_t2map_modified/predictions/IMAGENET_TEST_1k"
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
    images_dir = "/home/pbolan/prj/prostate_t2map_modified/datasets/synth_imagenet_1k_test/images"
    labels_dir = "/home/pbolan/prj/prostate_t2map_modified/datasets/synth_imagenet_1k_test/labels"
    preds_base_dir = "/home/pbolan/prj/prostate_t2map_modified/predictions/IMAGENET_TEST_1k"
    
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
    images_dir = "/home/pbolan/prj/prostate_t2map_modified/datasets/invivo2D_set3/images"
    
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
            preds_base_dir = "/home/pbolan/prj/prostate_t2map_modified/predictions/INVIVO2D_SET3"
        else:
            preds_base_dir = f"/home/pbolan/prj/prostate_t2map_modified/predictions/INVIVO2D_SET3_NOISE_{noise_level}"
        
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


def images_to_base64(images, rotation=1, t_max=4):
    """Convert the parametric images to base64 encoded PNGs.
    
    Args:
        images: Dictionary of image arrays to convert
        rotation: Number of 90-degree clockwise rotations to apply (default=1)
        t_max: Maximum value for T parameter scaling (default=4)
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    image_data = {}
    
    # Define scaling for each parameter
    s0_vmin, s0_vmax = 0, 1
    t_vmin, t_vmax = 0, t_max
    
    for key, array in images.items():
        # Determine scaling and colormap based on image type
        if 'synth' in key:
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
        
        # Create figure for this image
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(normalized.T, origin='lower', cmap=cmap, vmin=0, vmax=255)
        ax.axis('off')
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        image_data[key] = img_str
    
    return image_data


@app.route('/')
def index():
    cases = get_available_cases()
    return render_template('index.html', cases=cases)


@app.route('/sample_datasets')
def sample_datasets():
    # Get dataset type from query parameter
    from flask import request
    dataset_type = request.args.get('type', 'imagenet')
    
    # Validate dataset type
    valid_types = ['invivo', 'imagenet', 'urand']
    if dataset_type not in valid_types:
        dataset_type = 'imagenet'  # Default to imagenet
    
    # Load case 000001 with default settings for display
    case_id = '000001'
    images = load_images(case_id, prediction_method='FIT_NLLS', reference_method='Ground_Truth')
    
    if images is not None:
        # Extract pixel values at coordinate (20, 20) for all synth images
        pixel_x, pixel_y = 20, 20
        synth_values = []
        for i in range(10):
            synth_key = f'synth_{i}'
            if synth_key in images:
                # Extract pixel value at (20, 20)
                value = float(images[synth_key][pixel_x, pixel_y])
                synth_values.append(value)
        
        # Convert all synth arrays to lists for JavaScript access
        # Need to rotate them the same way as the displayed images
        synth_arrays = []
        for i in range(10):
            synth_key = f'synth_{i}'
            if synth_key in images:
                # Rotate 90 degrees clockwise and transpose to match displayed orientation
                rotated = np.rot90(images[synth_key], k=-1).T
                synth_arrays.append(rotated.tolist())
        
        # Also convert S0 and T parameter arrays for JavaScript access
        s0_array = np.rot90(images['label_S0'], k=-1).T.tolist()
        t_array = np.rot90(images['label_T'], k=-1).T.tolist()
        
        # Convert images to base64 for web display (with 1 CW rotation)
        image_data = images_to_base64(images, rotation=1)
    else:
        print(f"Sample dataset images not found for type: {dataset_type}")
        image_data = None
        synth_values = []
        synth_arrays = []
        s0_array = []
        t_array = []
    
    return render_template('sample_datasets.html', 
                          images=image_data, 
                          dataset_type=dataset_type,
                          synth_values=synth_values,
                          synth_arrays=synth_arrays,
                          s0_array=s0_array,
                          t_array=t_array)


@app.route('/invivo_datasets')
def invivo_datasets():
    from flask import request
    
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
    from flask import request
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
