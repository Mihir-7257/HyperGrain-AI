"""
spatial_mapping.py
Implements pixel-by-pixel chemical mapping and spatial visualization as described in:
- Xuan et al. (2024), Figures 1, 7, 8: "Visualised protein distribution of single and multiple grains"
- Hu et al. (2025), Figures 9, 10: "Visualization of protein and moisture content"
Applies trained chemometric (PLSR) and deep learning (CLNet) models across 3D hyperspectral
cubes (WheatDual-4V) to generate pseudo-color chemical heatmaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2


def segment_grain_mask(cube, contrast_band_idx=200, threshold_ratio=0.15):
    """
    Otsu-based or adaptive thresholding on a selected near-infrared band (~800 nm)
    to segment individual wheat grain pixels from the black foam background tray.
    Args:
        cube: 3D np.ndarray (H, W, Bands)
        contrast_band_idx: band index corresponding to ~800 nm with strong grain reflection
    Returns:
        mask: 2D binary np.ndarray (H, W) where 1=grain, 0=background
    """
    slice_img = cube[:, :, contrast_band_idx].copy()
    
    # Normalize slice to 0-255 uint8
    norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply Gaussian blur followed by Otsu's thresholding
    blurred = cv2.GaussianBlur(norm, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    
    # If mask is empty or inverted, fallback to simple percentile threshold
    if np.sum(mask) < 10:
        thresh = np.percentile(slice_img, 60)
        mask = (slice_img > thresh).astype(np.uint8)
        
    return mask


def map_cube_to_chemical_heatmap(cube, model, hsi_wavelengths, model_wavelengths, 
                                 target_name="Protein", vmin=None, vmax=None):
    """
    Evaluates the trained regression model on every single pixel of the 3D hyperspectral cube
    to generate the spatial chemical distribution heatmap.
    Args:
        cube: 3D np.ndarray (H, W, Bands)
        model: Trained model object (PLSR, CLNet wrapper, etc.)
        hsi_wavelengths: 1D array of wavelengths corresponding to the cube (e.g. 400-1000 nm)
        model_wavelengths: 1D array of wavelengths the model expects (e.g. 950-1650 nm)
    Returns:
        heatmap: 2D np.ndarray (H, W) of predicted chemical percentages (NaN for background)
        mask: 2D binary np.ndarray (H, W)
    """
    H, W, C = cube.shape
    mask = segment_grain_mask(cube)
    
    # Common overlap/interpolation range
    # Interpolator maps cube's spectral profile to model wavelength domain
    f_interp = interp1d(hsi_wavelengths, cube, axis=-1, bounds_error=False, fill_value="extrapolate")
    cube_resampled = f_interp(model_wavelengths)  # (H, W, len(model_wavelengths))

    grain_pixels_y, grain_pixels_x = np.where(mask == 1)
    
    heatmap = np.full((H, W), np.nan, dtype=np.float32)
    
    if len(grain_pixels_y) > 0:
        pixel_spectra = cube_resampled[grain_pixels_y, grain_pixels_x, :]  # (N_pixels, Bands)
        
        # Predict pixel values
        predictions = model.predict(pixel_spectra)
        heatmap[grain_pixels_y, grain_pixels_x] = predictions
        
    return heatmap, mask


def plot_chemical_distribution(heatmap, mask, target_name="Protein", sample_name="Wheat Seed", 
                               save_path=None, cmap='jet'):
    """
    Generates a publication-quality pseudo-color distribution map with colorbar
    matching Xuan et al. (2024) Fig. 7 and Hu et al. (2025) Fig. 9/10.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Set background color to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')
    
    valid_vals = heatmap[~np.isnan(heatmap)]
    if len(valid_vals) > 0:
        vmin = np.percentile(valid_vals, 5)
        vmax = np.percentile(valid_vals, 95)
    else:
        vmin, vmax = 8.0, 16.0
        
    im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{target_name} Content (%)', fontsize=11, fontweight='bold')
    
    ax.set_title(f"Spatial Distribution of {target_name}\n({sample_name})", fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved chemical heatmap to {save_path}")
    else:
        plt.show()


def generate_multi_cultivar_comparison(cubes_dict, model, hsi_wl, model_wl, target_name="Protein", save_path="outputs/multi_cultivar_heatmap.png"):
    """
    Generates a multi-variety panel matching Xuan et al. (2024) Fig. 8 and Hu et al. (2025) Fig. 9:
    visualizing the spatial protein and moisture distribution across multiple wheat cultivars.
    """
    num_vars = len(cubes_dict)
    fig, axes = plt.subplots(1, num_vars, figsize=(3.5 * num_vars, 4), dpi=300)
    if num_vars == 1:
        axes = [axes]
        
    all_valid = []
    heatmaps = []
    
    for var_name, cube in cubes_dict.items():
        hm, mask = map_cube_to_chemical_heatmap(cube, model, hsi_wl, model_wl, target_name)
        heatmaps.append((var_name, hm))
        valid = hm[~np.isnan(hm)]
        if len(valid) > 0:
            all_valid.extend(valid)
            
    vmin = np.percentile(all_valid, 5) if len(all_valid) > 0 else 8.0
    vmax = np.percentile(all_valid, 95) if len(all_valid) > 0 else 16.0
    
    for ax, (var_name, hm) in zip(axes, heatmaps):
        ax.set_facecolor('black')
        im = ax.imshow(hm, cmap='jet', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax.set_title(f"{var_name.upper()}", fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'{target_name} Content (%)', fontsize=11, fontweight='bold')
    
    fig.suptitle(f"Visualized {target_name} Distribution Across Wheat Cultivars", fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved multi-cultivar comparison to {save_path}")


if __name__ == "__main__":
    from data_loader import load_wheat_chemical_data, load_wheat_hsi_cubes, spxy_split
    from models import TunedPLSR

    # 1. Train quick model on chemical data
    wl, X, yp, ym, _ = load_wheat_chemical_data()
    pls = TunedPLSR().fit(X, yp)

    # 2. Load sample HSI cubes
    cubes, labels, views, paths = load_wheat_hsi_cubes(max_samples_per_variety=1)
    hsi_wl = np.linspace(400, 1000, cubes[0].shape[-1])

    # 3. Generate sample single heatmap
    sample_cube = cubes[0]
    sample_label = labels[0]
    hm, mask = map_cube_to_chemical_heatmap(sample_cube, pls, hsi_wl, wl, target_name="Protein")
    plot_chemical_distribution(hm, mask, target_name="Protein", sample_name=sample_label, 
                               save_path="outputs/sample_seed_protein_heatmap.png")

    # 4. Generate multi-cultivar panel
    cubes_dict = {lbl: cb for lbl, cb in zip(labels, cubes)}
    generate_multi_cultivar_comparison(cubes_dict, pls, hsi_wl, wl, target_name="Protein",
                                       save_path="outputs/cultivar_protein_comparison.png")
