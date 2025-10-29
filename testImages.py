import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def load_images(case_id="000012"):
    """Load and return the four parametric images as numpy arrays in a dictionary."""
    
    # Define file paths
    static_dir = "static"
    labels_file = os.path.join(static_dir, f"labels_{case_id}.nii.gz")
    preds_file = os.path.join(static_dir, f"preds_{case_id}.nii.gz")
    synth_file = os.path.join(static_dir, f"synth_{case_id}.nii.gz")

    # Check if files exist
    for file_path in [labels_file, preds_file]:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
    
    # Load the NIfTI files
    labels_img = nib.load(labels_file)
    preds_img = nib.load(preds_file)
    synth_img = nib.load(synth_file) if os.path.exists(synth_file) else None
    
    # Get the data arrays
    labels_data = labels_img.get_fdata()
    preds_data = preds_img.get_fdata()
    
    print(f"Labels shape: {labels_data.shape}")
    print(f"Preds shape: {preds_data.shape}")
    if synth_img:
        synth_data = synth_img.get_fdata()
        print(f"Synth shape: {synth_data.shape}")
    
    # Assuming 4D data (x, y, z, parameters) - take middle slice
    if len(labels_data.shape) == 4:
        middle_slice = labels_data.shape[2] // 2
        labels_s0 = labels_data[:, :, middle_slice, 0]  # S0 parameter
        labels_t = labels_data[:, :, middle_slice, 1] if labels_data.shape[3] > 1 else labels_data[:, :, middle_slice, 0]  # T parameter
        preds_s0 = preds_data[:, :, middle_slice, 0]    # S0 parameter
        preds_t = preds_data[:, :, middle_slice, 1] if preds_data.shape[3] > 1 else preds_data[:, :, middle_slice, 0]    # T parameter
    else:
        # If 3D, just take middle slice
        middle_slice = labels_data.shape[2] // 2
        labels_s0 = labels_data[:, :, middle_slice]
        labels_t = labels_s0  # Same slice if only 3D
        preds_s0 = preds_data[:, :, middle_slice]
        preds_t = preds_s0
    
    return {
        'label_S0': labels_s0,
        'label_T': labels_t,
        'pred_S0': preds_s0,
        'pred_T': preds_t
    }

def plot_case(images, case_id="000012"):
    """Plot the four parametric images in a 2x2 grid."""
    
    if images is None:
        print("No images to plot")
        return
    
    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Case {case_id}: Labels (top) vs Predictions (bottom)", fontsize=16)
    
    # Get separate vmin/vmax for S0 and T parameters
    s0_vmin = 0
    s0_vmax = 2
    t_vmin = 0
    t_vmax = 4
    
    # Top row: Labels
    im1 = axes[0, 0].imshow(images['label_S0'].T, origin='lower', cmap='viridis', vmin=s0_vmin, vmax=s0_vmax)
    axes[0, 0].set_title("Labels - S0 Parameter")
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(images['label_T'].T, origin='lower', cmap='viridis', vmin=t_vmin, vmax=t_vmax)
    axes[0, 1].set_title("Labels - T Parameter")
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Bottom row: Predictions (match corresponding top row scaling)
    im3 = axes[1, 0].imshow(images['pred_S0'].T, origin='lower', cmap='viridis', vmin=s0_vmin, vmax=s0_vmax)
    axes[1, 0].set_title("Predictions - S0 Parameter")
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(images['pred_T'].T, origin='lower', cmap='viridis', vmin=t_vmin, vmax=t_vmax)
    axes[1, 1].set_title("Predictions - T Parameter")
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Print some basic stats
    print(f"\nLabels stats:")
    print(f"  S0: min={np.min(images['label_S0']):.3f}, max={np.max(images['label_S0']):.3f}, mean={np.mean(images['label_S0']):.3f}")
    print(f"  T:  min={np.min(images['label_T']):.3f}, max={np.max(images['label_T']):.3f}, mean={np.mean(images['label_T']):.3f}")

    print(f"\nPredictions stats:")
    print(f"  S0: min={np.min(images['pred_S0']):.3f}, max={np.max(images['pred_S0']):.3f}, mean={np.mean(images['pred_S0']):.3f}")
    print(f"  T:  min={np.min(images['pred_T']):.3f}, max={np.max(images['pred_T']):.3f}, mean={np.mean(images['pred_T']):.3f}")

def load_and_plot_case(case_id="000012"):
    """Load and plot labels and predictions for a given case."""
    images = load_images(case_id)
    plot_case(images, case_id)

if __name__ == "__main__":
    # Run the visualization
    load_and_plot_case("000012")