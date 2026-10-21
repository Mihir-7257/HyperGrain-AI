"""
data_loader.py
Handles loading and partitioning of:
1. Certified Wheat NIR chemical dataset (sensAIfood Perten: 150 samples with lab Protein % and Moisture %).
2. WheatDual-4V 3D Hyperspectral Image Cubes (1,600 kernels across 4 cultivars, 300 bands).
3. SPXY (Sample set Partitioning based on joint X-Y distances) and Stratified/Random 80/20 train-test splits.
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import train_test_split

def load_wheat_chemical_data(csv_path="data/sensAIfood_Perten/Wheat_sensAIfood_Perten.csv"):
    """
    Loads the wheat NIR spectroscopy dataset with reference wet-lab chemical values.
    Returns:
        wavelengths: np.ndarray of shape (num_bands,)
        X: np.ndarray of shape (num_samples, num_bands) - Reflectance/Absorbance spectra
        y_protein: np.ndarray of shape (num_samples,) - Reference Protein %
        y_moisture: np.ndarray of shape (num_samples,) - Reference Moisture %
        metadata: pd.DataFrame containing sample ID, variety, origin, etc.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Chemical data CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Extract spectral columns (numeric column headers, e.g. 950 to 1650)
    spec_col_names = [c for c in df.columns if c.replace('.', '', 1).isdigit()]
    wavelengths = np.array([float(c) for c in spec_col_names])
    
    # Sort wavelengths in ascending order
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    spec_col_names = [spec_col_names[i] for i in sort_idx]

    X = df[spec_col_names].values.astype(np.float32)
    y_protein = df['Protein'].values.astype(np.float32)
    y_moisture = df['Moisture'].values.astype(np.float32)

    meta_cols = [c for c in df.columns if c not in spec_col_names]
    metadata = df[meta_cols].copy()

    return wavelengths, X, y_protein, y_moisture, metadata


def augment_spectral_data(X, y_protein, y_moisture, target_samples=300, noise_level=0.003, random_state=42):
    """
    Spectroscopic Data Augmentation to scale dataset from 150 to 300, 400, or 500+ samples.
    Applies physics-informed variations:
    1. Instrument Gaussian noise (sensor thermal drift).
    2. Minor baseline offset & tilt (slight illumination variation).
    3. Convex combination of neighboring spectral curves.
    """
    np.random.seed(random_state)
    N, J = X.shape
    if target_samples <= N:
        return X, y_protein, y_moisture

    additional_needed = target_samples - N
    augmented_X = []
    augmented_yp = []
    augmented_ym = []

    wl_grid = np.linspace(-1, 1, J)

    for i in range(additional_needed):
        # Pick two random samples to blend slightly (Mixup)
        idx1, idx2 = np.random.choice(N, size=2, replace=False)
        alpha = np.random.beta(8, 2)  # heavily biased towards idx1 (near 0.8-0.95)
        
        blended_x = alpha * X[idx1] + (1 - alpha) * X[idx2]
        blended_yp = alpha * y_protein[idx1] + (1 - alpha) * y_protein[idx2]
        blended_ym = alpha * y_moisture[idx1] + (1 - alpha) * y_moisture[idx2]

        # Add minor instrument sensor Gaussian noise
        noise = np.random.normal(0, noise_level, size=J)
        
        # Add slight baseline slope / offset
        offset = np.random.normal(0, 0.005)
        slope = np.random.normal(0, 0.002) * wl_grid
        
        aug_x = blended_x + noise + offset + slope
        augmented_X.append(aug_x)
        augmented_yp.append(blended_yp)
        augmented_ym.append(blended_ym)

    X_total = np.vstack([X, np.array(augmented_X)])
    yp_total = np.concatenate([y_protein, np.array(augmented_yp)])
    ym_total = np.concatenate([y_moisture, np.array(augmented_ym)])

    return X_total, yp_total, ym_total


def spxy_split(X, y, test_size=0.2):
    """
    Sample set Partitioning based on joint X-Y distances (SPXY) algorithm.
    Used in Xuan et al. (2024) and Hu et al. (2025) to divide representative samples
    into calibration (training) and prediction (testing) sets.
    """
    N = X.shape[0]
    num_test = int(np.round(N * test_size))
    num_cal = N - num_test

    # Normalize X and Y to [0, 1] range to calculate joint Euclidean distance
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y_norm = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0) + 1e-8)

    # Compute Euclidean distance matrices
    dist_x = np.linalg.norm(X_norm[:, None, :] - X_norm[None, :, :], axis=-1)
    dist_y = np.linalg.norm(y_norm[:, None, :] - y_norm[None, :, :], axis=-1)

    # Max distance normalization
    d_x_max = np.max(dist_x)
    d_y_max = np.max(dist_y)
    
    dist_xy = (dist_x / (d_x_max + 1e-8)) + (dist_y / (d_y_max + 1e-8))

    # Initialize calibration set with the two samples having the largest distance
    i_max, j_max = np.unravel_index(np.argmax(dist_xy), dist_xy.shape)
    cal_idx = [i_max, j_max]
    remaining_idx = list(set(range(N)) - set(cal_idx))

    # Iteratively add samples that maximize the minimum distance to existing calibration set
    while len(cal_idx) < num_cal:
        d_min_to_cal = []
        for rem in remaining_idx:
            d_to_existing = [dist_xy[rem, c] for c in cal_idx]
            d_min_to_cal.append(min(d_to_existing))
        
        best_cand = remaining_idx[np.argmax(d_min_to_cal)]
        cal_idx.append(best_cand)
        remaining_idx.remove(best_cand)

    val_idx = remaining_idx
    return np.array(cal_idx), np.array(val_idx)


def load_wheat_hsi_cubes(base_dir="data/Hyperspectral data of wheat seeds/Hyperspectral data of wheat seeds/Hyperspectral dataset of wheat seeds", max_samples_per_variety=50):
    """
    Loads 3D hyperspectral image cubes of wheat seeds from the Mendeley WheatDual-4V dataset.
    Returns:
        cubes: list of 3D np.ndarray (H, W, 300)
        varieties: list of str ('yangmai6', 'fanmai8', 'xingmai13', 'jinan17')
        views: list of str ('L-1', 'L-2', 'S-1', 'S-2')
        file_paths: list of str
    """
    varieties_list = ['yangmai6', 'fanmai8', 'xingmai13', 'jinan17']
    cubes = []
    labels = []
    views = []
    paths = []

    for var in varieties_list:
        var_path = os.path.join(base_dir, var)
        if not os.path.isdir(var_path):
            continue
        
        view_dirs = sorted(os.listdir(var_path))
        count = 0
        for v in view_dirs:
            view_path = os.path.join(var_path, v)
            if not os.path.isdir(view_path):
                continue
            
            mat_files = sorted(glob.glob(os.path.join(view_path, "*.mat")))
            for mf in mat_files:
                try:
                    mat_data = scipy.io.loadmat(mf)
                    if 'hyperspectral_data' in mat_data:
                        cube = mat_data['hyperspectral_data']  # (H, W, 300)
                        cubes.append(cube)
                        labels.append(var)
                        views.append(v)
                        paths.append(mf)
                        count += 1
                        if max_samples_per_variety and count >= max_samples_per_variety:
                            break
                except Exception as e:
                    print(f"Error loading {mf}: {e}")
            if max_samples_per_variety and count >= max_samples_per_variety:
                break

    print(f"Loaded {len(cubes)} HSI cubes across {len(set(labels))} varieties.")
    return cubes, labels, views, paths


if __name__ == "__main__":
    wl, X, yp, ym, meta = load_wheat_chemical_data()
    print(f"Chemical data loaded: X shape={X.shape}, wavelengths={len(wl)} ({wl[0]}-{wl[-1]} nm)")
    print(f"Protein % range: [{yp.min():.2f}, {yp.max():.2f}], Moisture % range: [{ym.min():.2f}, {ym.max():.2f}]")
    
    cal_idx, val_idx = spxy_split(X, yp, test_size=0.2)
    print(f"SPXY split: Calibration={len(cal_idx)}, Prediction={len(val_idx)}")
    
    cubes, labels, views, paths = load_wheat_hsi_cubes(max_samples_per_variety=5)
    print(f"Sample HSI cube shape: {cubes[0].shape}, Varieties: {set(labels)}")
