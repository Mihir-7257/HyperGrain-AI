import os
import scipy.io
import pandas as pd
import numpy as np

print("=== 1. INSPECTING WHEAT CHEMICAL DATA (sensAIfood) ===")
wheat_csv = "data/sensAIfood_Perten/Wheat_sensAIfood_Perten.csv"
if os.path.exists(wheat_csv):
    df = pd.read_csv(wheat_csv)
    print("Columns:", df.columns[:10].tolist(), "...", df.columns[-5:].tolist())
    print("Shape:", df.shape)
    print("First few rows of metadata/target columns:")
    for col in df.columns:
        if any(term in col.lower() for term in ['prot', 'moist', 'water', 'sample', 'id', 'class', 'target']):
            print(f"  Target/Metadata column '{col}': min={df[col].min()}, max={df[col].max()}, mean={df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'}")
            
    # Check wavelengths
    numeric_cols = [c for c in df.columns if c.replace('.', '', 1).isdigit() or c.isdigit()]
    if numeric_cols:
        print(f"Spectral wavelength columns: {len(numeric_cols)} bands from {numeric_cols[0]} to {numeric_cols[-1]}")
    else:
        print("Checking first 5 column names:", list(df.columns[:5]))
else:
    print(f"Not found: {wheat_csv}")

print("\n=== 2. INSPECTING WHEAT 3D HSI IMAGE CUBE (Mendeley) ===")
sample_mat = "data/Hyperspectral data of wheat seeds/Hyperspectral data of wheat seeds/Hyperspectral dataset of wheat seeds/yangmai6/L-1/region_000.mat"
if os.path.exists(sample_mat):
    mat_data = scipy.io.loadmat(sample_mat)
    print("Keys in .mat:", [k for k in mat_data.keys() if not k.startswith('__')])
    for k in mat_data:
        if not k.startswith('__'):
            val = mat_data[k]
            print(f"Key '{k}': type={type(val)}, shape={getattr(val, 'shape', 'no shape')}, dtype={getattr(val, 'dtype', 'no dtype')}")
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                print(f"  Min={np.min(val)}, Max={np.max(val)}, Mean={np.mean(val)}")
else:
    print(f"Not found: {sample_mat}")
