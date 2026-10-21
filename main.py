"""
main.py
Master execution pipeline for:
"Non-Destructive Prediction and Spatial Visualization of Wheat Grain Protein and Moisture Content
Using Hyperspectral Imaging and Deep Learning"

Implements the end-to-end methodologies from:
- Xuan et al. (2024), Spectrochimica Acta Part A
- Hu et al. (2025), Journal of Food Composition and Analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_wheat_chemical_data, load_wheat_hsi_cubes, spxy_split
from preprocessing import apply_preprocessing
from feature_selection import SPA, CARS
from models import TunedPLSR, MLRBaseline, OneDCNNModel, LSTMModel, CLNet, PyTorchRegressorWrapper, compute_metrics
from spatial_mapping import map_cube_to_chemical_heatmap, plot_chemical_distribution, generate_multi_cultivar_comparison
from evaluate import (plot_spectral_curves, plot_feature_wavelengths, 
                      plot_prediction_scatter, plot_model_comparison_bars)


def run_pipeline():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 80)
    print(">>> GRAIN QUALITY PREDICTION & SPATIAL CHEMICAL VISUALIZATION PIPELINE <<<")
    print("=" * 80)
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------------------------------------
    # 1. DATA INGESTION
    # -------------------------------------------------------------
    print("\n[STEP 1/6] Ingesting Certified Wheat NIR Spectroscopy Dataset...")
    wl, X, y_protein, y_moisture, meta = load_wheat_chemical_data()
    print(f"  [+] Samples: {X.shape[0]} wheat samples")
    print(f"  [+] Spectral Bands: {len(wl)} channels ({wl[0]:.0f} nm to {wl[-1]:.0f} nm)")
    print(f"  [+] Laboratory Protein %: range [{y_protein.min():.2f}%, {y_protein.max():.2f}%], mean={y_protein.mean():.2f}%")
    print(f"  [+] Laboratory Moisture %: range [{y_moisture.min():.2f}%, {y_moisture.max():.2f}%], mean={y_moisture.mean():.2f}%")

    # SPXY Sample Partitioning (2:1 or 4:1 ratio; here 80% Calibration, 20% Prediction)
    print("\n[STEP 2/6] Partitioning Samples via SPXY Algorithm (80/20 Calibration-Prediction Split)...")
    cal_idx, val_idx = spxy_split(X, y_protein, test_size=0.20)
    X_cal, X_val = X[cal_idx], X[val_idx]
    y_p_cal, y_p_val = y_protein[cal_idx], y_protein[val_idx]
    y_m_cal, y_m_val = y_moisture[cal_idx], y_moisture[val_idx]
    print(f"  [+] Calibration Set (Train): {len(cal_idx)} samples")
    print(f"  [+] Prediction Set (Test):  {len(val_idx)} samples")

    # -------------------------------------------------------------
    # 2. SPECTRAL PREPROCESSING & VISUALIZATION
    # -------------------------------------------------------------
    print("\n[STEP 3/6] Applying Mathematical Preprocessing (SNV, 1st Derivative, MSC)...")
    X_cal_snv, X_val_snv = apply_preprocessing(X_cal, X_val, method='snv')
    X_cal_1st, X_val_1st = apply_preprocessing(X_cal, X_val, method='1st')
    X_cal_msc, X_val_msc = apply_preprocessing(X_cal, X_val, method='msc')

    # Plot spectral curves
    plot_spectral_curves(wl, X_cal, X_cal_snv, X_cal_1st, save_path="outputs/spectral_curves_comparison.png")

    # -------------------------------------------------------------
    # 3. FEATURE SELECTION (SPA & CARS)
    # -------------------------------------------------------------
    print("\n[STEP 4/6] Selecting Characteristic Absorption Wavelengths (SPA & CARS)...")
    print("  * Running Successive Projections Algorithm (SPA) for Protein...")
    spa_p = SPA(k_max=15, cv=5)
    X_cal_snv_spa = spa_p.fit_transform(X_cal_snv, y_p_cal, wavelengths=wl)
    X_val_snv_spa = spa_p.transform(X_val_snv)
    print(f"    Selected {len(spa_p.selected_indices_)} bands: {[f'{w:.0f}nm' for w in spa_p.selected_wavelengths_]}")

    print("  * Running Competitive Adaptive Reweighted Sampling (CARS) for Moisture...")
    cars_m = CARS(num_runs=35, cv=5, n_components=5)
    X_cal_1st_cars = cars_m.fit_transform(X_cal_1st, y_m_cal, wavelengths=wl)
    X_val_1st_cars = cars_m.transform(X_val_1st)
    print(f"    Selected {len(cars_m.selected_indices_)} bands: {[f'{w:.0f}nm' for w in cars_m.selected_wavelengths_]}")

    # Plot feature selection
    mean_spec = np.mean(X_cal, axis=0)
    plot_feature_wavelengths(wl, mean_spec, spa_p.selected_indices_, cars_m.selected_indices_,
                             save_path="outputs/feature_wavelengths_selection.png")

    # -------------------------------------------------------------
    # 4. MODEL TRAINING & COMPARISON (PROTEIN & MOISTURE)
    # -------------------------------------------------------------
    print("\n[STEP 5/6] Training & Evaluating Regression Models (Full Preprocessing x Model Matrix)...")
    from models import SVRBaseline, RandomForestBaseline
    results_records = []

    # === TASK A: PROTEIN PREDICTION ===
    print("\n  =========================================")
    print("  [TASK A] WHEAT PROTEIN CONTENT PREDICTION")
    print("  =========================================")
    
    # 0. Raw Baseline (No preprocessing)
    pls_p_raw = TunedPLSR(max_components=10, cv=5).fit(X_cal, y_p_cal)
    pred_cal_raw = pls_p_raw.predict(X_cal)
    pred_val_raw = pls_p_raw.predict(X_val)
    m_cal_raw = compute_metrics(y_p_cal, pred_cal_raw)
    m_val_raw = compute_metrics(y_p_val, pred_val_raw)
    print(f"  [0] Raw-PLSR:        Rc^2={m_cal_raw['R2']:.4f}, RMSEC={m_cal_raw['RMSE']:.4f} | Rp^2={m_val_raw['R2']:.4f}, RMSEP={m_val_raw['RMSE']:.4f}, RPD={m_val_raw['RPD']:.2f}")
    results_records.append({'Target': 'Protein', 'Model': 'Raw-PLSR', 'Rc2': m_cal_raw['R2'], 'RMSEC': m_cal_raw['RMSE'], 'Rp2': m_val_raw['R2'], 'RMSEP': m_val_raw['RMSE'], 'RPD': m_val_raw['RPD']})

    # 1. SNV + PLSR
    pls_p = TunedPLSR(max_components=15, cv=5).fit(X_cal_snv, y_p_cal)
    pred_cal_pls_p = pls_p.predict(X_cal_snv)
    pred_val_pls_p = pls_p.predict(X_val_snv)
    m_cal_pls_p = compute_metrics(y_p_cal, pred_cal_pls_p)
    m_val_pls_p = compute_metrics(y_p_val, pred_val_pls_p)
    print(f"  [1] SNV + PLSR:      Rc^2={m_cal_pls_p['R2']:.4f}, RMSEC={m_cal_pls_p['RMSE']:.4f} | Rp^2={m_val_pls_p['R2']:.4f}, RMSEP={m_val_pls_p['RMSE']:.4f}, RPD={m_val_pls_p['RPD']:.2f}")
    results_records.append({'Target': 'Protein', 'Model': 'SNV + PLSR', 'Rc2': m_cal_pls_p['R2'], 'RMSEC': m_cal_pls_p['RMSE'], 'Rp2': m_val_pls_p['R2'], 'RMSEP': m_val_pls_p['RMSE'], 'RPD': m_val_pls_p['RPD']})

    # 2. SNV + Full-Spectrum MLR
    mlr_p_full = MLRBaseline().fit(X_cal_snv, y_p_cal)
    pred_cal_mlr_full = mlr_p_full.predict(X_cal_snv)
    pred_val_mlr_full = mlr_p_full.predict(X_val_snv)
    m_cal_mlr_full = compute_metrics(y_p_cal, pred_cal_mlr_full)
    m_val_mlr_full = compute_metrics(y_p_val, pred_val_mlr_full)
    print(f"  [2] SNV + MLR(Full): Rc^2={m_cal_mlr_full['R2']:.4f}, RMSEC={m_cal_mlr_full['RMSE']:.4f} | Rp^2={m_val_mlr_full['R2']:.4f}, RMSEP={m_val_mlr_full['RMSE']:.4f}, RPD={m_val_mlr_full['RPD']:.2f}")
    results_records.append({'Target': 'Protein', 'Model': 'SNV + MLR(Full)', 'Rc2': m_cal_mlr_full['R2'], 'RMSEC': m_cal_mlr_full['RMSE'], 'Rp2': m_val_mlr_full['R2'], 'RMSEP': m_val_mlr_full['RMSE'], 'RPD': m_val_mlr_full['RPD']})

    # 3. SNV + SPA-MLR (15 Selected Wavelengths)
    mlr_p = MLRBaseline().fit(X_cal_snv_spa, y_p_cal)
    pred_cal_mlr_p = mlr_p.predict(X_cal_snv_spa)
    pred_val_mlr_p = mlr_p.predict(X_val_snv_spa)
    m_cal_mlr_p = compute_metrics(y_p_cal, pred_cal_mlr_p)
    m_val_mlr_p = compute_metrics(y_p_val, pred_val_mlr_p)
    print(f"  [3] SNV + SPA-MLR:   Rc^2={m_cal_mlr_p['R2']:.4f}, RMSEC={m_cal_mlr_p['RMSE']:.4f} | Rp^2={m_val_mlr_p['R2']:.4f}, RMSEP={m_val_mlr_p['RMSE']:.4f}, RPD={m_val_mlr_p['RPD']:.2f}")
    results_records.append({'Target': 'Protein', 'Model': 'SNV + SPA-MLR', 'Rc2': m_cal_mlr_p['R2'], 'RMSEC': m_cal_mlr_p['RMSE'], 'Rp2': m_val_mlr_p['R2'], 'RMSEP': m_val_mlr_p['RMSE'], 'RPD': m_val_mlr_p['RPD']})

    # 4. SNV + SVR (Support Vector Regression with RBF Kernel)
    svr_p = SVRBaseline(C=10.0, epsilon=0.1).fit(X_cal_snv, y_p_cal)
    pred_cal_svr_p = svr_p.predict(X_cal_snv)
    pred_val_svr_p = svr_p.predict(X_val_snv)
    m_cal_svr_p = compute_metrics(y_p_cal, pred_cal_svr_p)
    m_val_svr_p = compute_metrics(y_p_val, pred_val_svr_p)
    print(f"  [4] SNV + SVR(RBF):  Rc^2={m_cal_svr_p['R2']:.4f}, RMSEC={m_cal_svr_p['RMSE']:.4f} | Rp^2={m_val_svr_p['R2']:.4f}, RMSEP={m_val_svr_p['RMSE']:.4f}, RPD={m_val_svr_p['RPD']:.2f}")
    results_records.append({'Target': 'Protein', 'Model': 'SNV + SVR', 'Rc2': m_cal_svr_p['R2'], 'RMSEC': m_cal_svr_p['RMSE'], 'Rp2': m_val_svr_p['R2'], 'RMSEP': m_val_svr_p['RMSE'], 'RPD': m_val_svr_p['RPD']})

    # 5. SNV + 1D-CNN (PyTorch)
    cnn_p = PyTorchRegressorWrapper(OneDCNNModel, in_features=X_cal_snv.shape[1], epochs=120, lr=0.003).fit(X_cal_snv, y_p_cal)
    pred_cal_cnn_p = cnn_p.predict(X_cal_snv)
    pred_val_cnn_p = cnn_p.predict(X_val_snv)
    m_cal_cnn_p = compute_metrics(y_p_cal, pred_cal_cnn_p)
    m_val_cnn_p = compute_metrics(y_p_val, pred_val_cnn_p)
    print(f"  [5] SNV + 1D-CNN:    Rc^2={m_cal_cnn_p['R2']:.4f}, RMSEC={m_cal_cnn_p['RMSE']:.4f} | Rp^2={m_val_cnn_p['R2']:.4f}, RMSEP={m_val_cnn_p['RMSE']:.4f}, RPD={m_val_cnn_p['RPD']:.2f}")
    results_records.append({'Target': 'Protein', 'Model': 'SNV + 1D-CNN', 'Rc2': m_cal_cnn_p['R2'], 'RMSEC': m_cal_cnn_p['RMSE'], 'Rp2': m_val_cnn_p['R2'], 'RMSEP': m_val_cnn_p['RMSE'], 'RPD': m_val_cnn_p['RPD']})

    # 6. SNV + LSTM (PyTorch)
    lstm_p = PyTorchRegressorWrapper(LSTMModel, in_features=X_cal_snv.shape[1], epochs=120, lr=0.003).fit(X_cal_snv, y_p_cal)
    pred_cal_lstm_p = lstm_p.predict(X_cal_snv)
    pred_val_lstm_p = lstm_p.predict(X_val_snv)
    m_cal_lstm_p = compute_metrics(y_p_cal, pred_cal_lstm_p)
    m_val_lstm_p = compute_metrics(y_p_val, pred_val_lstm_p)
    print(f"  [6] SNV + LSTM:      Rc^2={m_cal_lstm_p['R2']:.4f}, RMSEC={m_cal_lstm_p['RMSE']:.4f} | Rp^2={m_val_lstm_p['R2']:.4f}, RMSEP={m_val_lstm_p['RMSE']:.4f}, RPD={m_val_lstm_p['RPD']:.2f}")
    results_records.append({'Target': 'Protein', 'Model': 'SNV + LSTM', 'Rc2': m_cal_lstm_p['R2'], 'RMSEC': m_cal_lstm_p['RMSE'], 'Rp2': m_val_lstm_p['R2'], 'RMSEP': m_val_lstm_p['RMSE'], 'RPD': m_val_lstm_p['RPD']})

    # 7. SNV + CLNet (Hybrid 1D-CNN + LSTM from Hu et al., 2025)
    clnet_p = PyTorchRegressorWrapper(CLNet, in_features=X_cal_snv.shape[1], epochs=150, lr=0.003).fit(X_cal_snv, y_p_cal)
    pred_cal_cl_p = clnet_p.predict(X_cal_snv)
    pred_val_cl_p = clnet_p.predict(X_val_snv)
    m_cal_cl_p = compute_metrics(y_p_cal, pred_cal_cl_p)
    m_val_cl_p = compute_metrics(y_p_val, pred_val_cl_p)
    print(f"  [7] SNV + CLNet:     Rc^2={m_cal_cl_p['R2']:.4f}, RMSEC={m_cal_cl_p['RMSE']:.4f} | Rp^2={m_val_cl_p['R2']:.4f}, RMSEP={m_val_cl_p['RMSE']:.4f}, RPD={m_val_cl_p['RPD']:.2f}")
    results_records.append({'Target': 'Protein', 'Model': 'SNV + CLNet', 'Rc2': m_cal_cl_p['R2'], 'RMSEC': m_cal_cl_p['RMSE'], 'Rp2': m_val_cl_p['R2'], 'RMSEP': m_val_cl_p['RMSE'], 'RPD': m_val_cl_p['RPD']})

    plot_prediction_scatter(y_p_cal, pred_cal_pls_p, y_p_val, pred_val_pls_p, 
                            m_cal_pls_p, m_val_pls_p, target_name="Protein", model_name="SNV-PLSR", 
                            save_path="outputs/scatter_protein_plsr.png")
    plot_prediction_scatter(y_p_cal, pred_cal_cl_p, y_p_val, pred_val_cl_p, 
                            m_cal_cl_p, m_val_cl_p, target_name="Protein", model_name="CLNet", 
                            save_path="outputs/scatter_protein_clnet.png")


    # === TASK B: MOISTURE PREDICTION ===
    print("\n  =========================================")
    print("  [TASK B] WHEAT MOISTURE CONTENT PREDICTION")
    print("  =========================================")

    # 1. 1st + PLSR
    pls_m = TunedPLSR(max_components=15, cv=5).fit(X_cal_1st, y_m_cal)
    pred_cal_pls_m = pls_m.predict(X_cal_1st)
    pred_val_pls_m = pls_m.predict(X_val_1st)
    m_cal_pls_m = compute_metrics(y_m_cal, pred_cal_pls_m)
    m_val_pls_m = compute_metrics(y_m_val, pred_val_pls_m)
    print(f"  [1] 1st + PLSR:      Rc^2={m_cal_pls_m['R2']:.4f}, RMSEC={m_cal_pls_m['RMSE']:.4f} | Rp^2={m_val_pls_m['R2']:.4f}, RMSEP={m_val_pls_m['RMSE']:.4f}, RPD={m_val_pls_m['RPD']:.2f}")
    results_records.append({'Target': 'Moisture', 'Model': '1st + PLSR', 'Rc2': m_cal_pls_m['R2'], 'RMSEC': m_cal_pls_m['RMSE'], 'Rp2': m_val_pls_m['R2'], 'RMSEP': m_val_pls_m['RMSE'], 'RPD': m_val_pls_m['RPD']})

    # 2. 1st + MLR (Full Spectrum)
    mlr_m_full = MLRBaseline().fit(X_cal_1st, y_m_cal)
    pred_cal_mlr_full_m = mlr_m_full.predict(X_cal_1st)
    pred_val_mlr_full_m = mlr_m_full.predict(X_val_1st)
    m_cal_mlr_full_m = compute_metrics(y_m_cal, pred_cal_mlr_full_m)
    m_val_mlr_full_m = compute_metrics(y_m_val, pred_val_mlr_full_m)
    print(f"  [2] 1st + MLR(Full): Rc^2={m_cal_mlr_full_m['R2']:.4f}, RMSEC={m_cal_mlr_full_m['RMSE']:.4f} | Rp^2={m_val_mlr_full_m['R2']:.4f}, RMSEP={m_val_mlr_full_m['RMSE']:.4f}, RPD={m_val_mlr_full_m['RPD']:.2f}")
    results_records.append({'Target': 'Moisture', 'Model': '1st + MLR(Full)', 'Rc2': m_cal_mlr_full_m['R2'], 'RMSEC': m_cal_mlr_full_m['RMSE'], 'Rp2': m_val_mlr_full_m['R2'], 'RMSEP': m_val_mlr_full_m['RMSE'], 'RPD': m_val_mlr_full_m['RPD']})

    # 3. 1st + CARS-MLR
    mlr_m = MLRBaseline().fit(X_cal_1st_cars, y_m_cal)
    pred_cal_mlr_m = mlr_m.predict(X_cal_1st_cars)
    pred_val_mlr_m = mlr_m.predict(X_val_1st_cars)
    m_cal_mlr_m = compute_metrics(y_m_cal, pred_cal_mlr_m)
    m_val_mlr_m = compute_metrics(y_m_val, pred_val_mlr_m)
    print(f"  [3] 1st + CARS-MLR:  Rc^2={m_cal_mlr_m['R2']:.4f}, RMSEC={m_cal_mlr_m['RMSE']:.4f} | Rp^2={m_val_mlr_m['R2']:.4f}, RMSEP={m_val_mlr_m['RMSE']:.4f}, RPD={m_val_mlr_m['RPD']:.2f}")
    results_records.append({'Target': 'Moisture', 'Model': '1st + CARS-MLR', 'Rc2': m_cal_mlr_m['R2'], 'RMSEC': m_cal_mlr_m['RMSE'], 'Rp2': m_val_mlr_m['R2'], 'RMSEP': m_val_mlr_m['RMSE'], 'RPD': m_val_mlr_m['RPD']})

    # 4. 1st + SVR
    svr_m = SVRBaseline(C=10.0, epsilon=0.1).fit(X_cal_1st, y_m_cal)
    pred_cal_svr_m = svr_m.predict(X_cal_1st)
    pred_val_svr_m = svr_m.predict(X_val_1st)
    m_cal_svr_m = compute_metrics(y_m_cal, pred_cal_svr_m)
    m_val_svr_m = compute_metrics(y_m_val, pred_val_svr_m)
    print(f"  [4] 1st + SVR(RBF):  Rc^2={m_cal_svr_m['R2']:.4f}, RMSEC={m_cal_svr_m['RMSE']:.4f} | Rp^2={m_val_svr_m['R2']:.4f}, RMSEP={m_val_svr_m['RMSE']:.4f}, RPD={m_val_svr_m['RPD']:.2f}")
    results_records.append({'Target': 'Moisture', 'Model': '1st + SVR', 'Rc2': m_cal_svr_m['R2'], 'RMSEC': m_cal_svr_m['RMSE'], 'Rp2': m_val_svr_m['R2'], 'RMSEP': m_val_svr_m['RMSE'], 'RPD': m_val_svr_m['RPD']})

    # 5. 1st + 1D-CNN (PyTorch)
    cnn_m = PyTorchRegressorWrapper(OneDCNNModel, in_features=X_cal_1st.shape[1], epochs=120, lr=0.003).fit(X_cal_1st, y_m_cal)
    pred_cal_cnn_m = cnn_m.predict(X_cal_1st)
    pred_val_cnn_m = cnn_m.predict(X_val_1st)
    m_cal_cnn_m = compute_metrics(y_m_cal, pred_cal_cnn_m)
    m_val_cnn_m = compute_metrics(y_m_val, pred_val_cnn_m)
    print(f"  [5] 1st + 1D-CNN:    Rc^2={m_cal_cnn_m['R2']:.4f}, RMSEC={m_cal_cnn_m['RMSE']:.4f} | Rp^2={m_val_cnn_m['R2']:.4f}, RMSEP={m_val_cnn_m['RMSE']:.4f}, RPD={m_val_cnn_m['RPD']:.2f}")
    results_records.append({'Target': 'Moisture', 'Model': '1st + 1D-CNN', 'Rc2': m_cal_cnn_m['R2'], 'RMSEC': m_cal_cnn_m['RMSE'], 'Rp2': m_val_cnn_m['R2'], 'RMSEP': m_val_cnn_m['RMSE'], 'RPD': m_val_cnn_m['RPD']})

    # 6. 1st + LSTM (PyTorch)
    lstm_m = PyTorchRegressorWrapper(LSTMModel, in_features=X_cal_1st.shape[1], epochs=120, lr=0.003).fit(X_cal_1st, y_m_cal)
    pred_cal_lstm_m = lstm_m.predict(X_cal_1st)
    pred_val_lstm_m = lstm_m.predict(X_val_1st)
    m_cal_lstm_m = compute_metrics(y_m_cal, pred_cal_lstm_m)
    m_val_lstm_m = compute_metrics(y_m_val, pred_val_lstm_m)
    print(f"  [6] 1st + LSTM:      Rc^2={m_cal_lstm_m['R2']:.4f}, RMSEC={m_cal_lstm_m['RMSE']:.4f} | Rp^2={m_val_lstm_m['R2']:.4f}, RMSEP={m_val_lstm_m['RMSE']:.4f}, RPD={m_val_lstm_m['RPD']:.2f}")
    results_records.append({'Target': 'Moisture', 'Model': '1st + LSTM', 'Rc2': m_cal_lstm_m['R2'], 'RMSEC': m_cal_lstm_m['RMSE'], 'Rp2': m_val_lstm_m['R2'], 'RMSEP': m_val_lstm_m['RMSE'], 'RPD': m_val_lstm_m['RPD']})

    # 7. 1st + CLNet (Hybrid 1D-CNN + LSTM from Hu et al., 2025)
    clnet_m = PyTorchRegressorWrapper(CLNet, in_features=X_cal_1st.shape[1], epochs=150, lr=0.003).fit(X_cal_1st, y_m_cal)
    pred_cal_cl_m = clnet_m.predict(X_cal_1st)
    pred_val_cl_m = clnet_m.predict(X_val_1st)
    m_cal_cl_m = compute_metrics(y_m_cal, pred_cal_cl_m)
    m_val_cl_m = compute_metrics(y_m_val, pred_val_cl_m)
    print(f"  [7] 1st + CLNet:     Rc^2={m_cal_cl_m['R2']:.4f}, RMSEC={m_cal_cl_m['RMSE']:.4f} | Rp^2={m_val_cl_m['R2']:.4f}, RMSEP={m_val_cl_m['RMSE']:.4f}, RPD={m_val_cl_m['RPD']:.2f}")
    results_records.append({'Target': 'Moisture', 'Model': '1st + CLNet', 'Rc2': m_cal_cl_m['R2'], 'RMSEC': m_cal_cl_m['RMSE'], 'Rp2': m_val_cl_m['R2'], 'RMSEP': m_val_cl_m['RMSE'], 'RPD': m_val_cl_m['RPD']})

    # --- SAMPLE SIZE COMPARISON EXPERIMENT (150 vs 300 vs 500 samples) ---
    print("\n  =======================================================")
    print("  [EXPERIMENT] SAMPLE SIZE SCALING (N=150 vs 300 vs 500)")
    print("  =======================================================")
    from data_loader import augment_spectral_data
    scaling_records = []
    
    for n_samples in [150, 300, 500]:
        if n_samples == 150:
            X_s, yp_s, ym_s = X, y_protein, y_moisture
        else:
            X_s, yp_s, ym_s = augment_spectral_data(X, y_protein, y_moisture, target_samples=n_samples)
            
        c_idx, v_idx = spxy_split(X_s, yp_s, test_size=0.20)
        X_tr_s, X_te_s = apply_preprocessing(X_s[c_idx], X_s[v_idx], method='snv')
        
        # Test PLSR & 1D-CNN on scaled data
        pls_scale = TunedPLSR(max_components=15).fit(X_tr_s, yp_s[c_idx])
        pred_pls_s = pls_scale.predict(X_te_s)
        m_pls_s = compute_metrics(yp_s[v_idx], pred_pls_s)
        
        cnn_scale = PyTorchRegressorWrapper(OneDCNNModel, in_features=X_tr_s.shape[1], epochs=100, lr=0.003).fit(X_tr_s, yp_s[c_idx])
        pred_cnn_s = cnn_scale.predict(X_te_s)
        m_cnn_s = compute_metrics(yp_s[v_idx], pred_cnn_s)
        
        print(f"  * N={n_samples:3d} (Train={len(c_idx)}, Test={len(v_idx)}): PLSR Rp^2={m_pls_s['R2']:.4f}, RMSEP={m_pls_s['RMSE']:.4f} | 1D-CNN Rp^2={m_cnn_s['R2']:.4f}, RMSEP={m_cnn_s['RMSE']:.4f}")
        scaling_records.append({'Sample_Size': n_samples, 'Train_Size': len(c_idx), 'Test_Size': len(v_idx), 
                                'PLSR_Rp2': m_pls_s['R2'], 'PLSR_RMSEP': m_pls_s['RMSE'], 'PLSR_RPD': m_pls_s['RPD'],
                                'CNN_Rp2': m_cnn_s['R2'], 'CNN_RMSEP': m_cnn_s['RMSE'], 'CNN_RPD': m_cnn_s['RPD']})
                                
    scaling_df = pd.DataFrame(scaling_records)
    scaling_df.to_csv("outputs/sample_size_comparison.csv", index=False)

    # Save Results Table and Bar Charts
    results_df = pd.DataFrame(results_records)
    results_df.to_csv("outputs/benchmark_results.csv", index=False)
    
    df_p = results_df[results_df['Target'] == 'Protein']
    df_m = results_df[results_df['Target'] == 'Moisture']
    plot_model_comparison_bars(df_p, target_name="Protein", save_path="outputs/models_comparison_protein.png")
    plot_model_comparison_bars(df_m, target_name="Moisture", save_path="outputs/models_comparison_moisture.png")


    # -------------------------------------------------------------
    # 5. SPATIAL CHEMICAL DISTRIBUTION HEATMAPS (3D HSI CUBES)
    # -------------------------------------------------------------
    print("\n[STEP 6/6] Generating Pixel-by-Pixel Chemical Distribution Heatmaps on 3D Wheat HSI Cubes...")
    cubes, labels, views, paths = load_wheat_hsi_cubes(max_samples_per_variety=3)
    hsi_wl = np.linspace(400, 1000, cubes[0].shape[-1])

    # Distinct sample from each variety
    unique_cultivars = {}
    for cb, lbl in zip(cubes, labels):
        if lbl not in unique_cultivars:
            unique_cultivars[lbl] = cb

    # 1. Single-Seed High-Resolution Heatmaps
    first_var = list(unique_cultivars.keys())[0]
    sample_cube = unique_cultivars[first_var]
    
    # Protein Heatmap
    hm_prot, mask_p = map_cube_to_chemical_heatmap(sample_cube, pls_p, hsi_wl, wl, target_name="Protein")
    plot_chemical_distribution(hm_prot, mask_p, target_name="Protein", sample_name=f"{first_var.upper()} Seed",
                               save_path="outputs/sample_seed_protein_heatmap.png", cmap='jet')

    # Moisture Heatmap
    hm_moist, mask_m = map_cube_to_chemical_heatmap(sample_cube, pls_m, hsi_wl, wl, target_name="Moisture")
    plot_chemical_distribution(hm_moist, mask_m, target_name="Moisture", sample_name=f"{first_var.upper()} Seed",
                               save_path="outputs/sample_seed_moisture_heatmap.png", cmap='turbo')

    # 2. Multi-Cultivar Panels (Yangmai 6, Fanmai 8, Xingmai 13, Jinan 17)
    generate_multi_cultivar_comparison(unique_cultivars, pls_p, hsi_wl, wl, target_name="Protein",
                                       save_path="outputs/cultivar_protein_comparison.png")
    generate_multi_cultivar_comparison(unique_cultivars, pls_m, hsi_wl, wl, target_name="Moisture",
                                       save_path="outputs/cultivar_moisture_comparison.png")

    print("\n" + "=" * 80)
    print(">>> ALL PIPELINE STAGES COMPLETED SUCCESSFULLY! <<<")
    print("=" * 80)
    print("\n[BENCHMARK RESULTS TABLE]")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
