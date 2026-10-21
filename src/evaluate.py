"""
evaluate.py
Produces academic evaluation metrics and publication-ready scientific visualizations
matching Xuan et al. (2024) and Hu et al. (2025):
1. Preprocessed Spectral Curves (Raw vs SNV vs 1st Derivative with absorption peaks).
2. Selected Characteristic Wavelengths (SPA & CARS).
3. Scatter Plots (Measured vs Predicted) for Calibration and Prediction sets.
4. Model Performance Radar and Bar Charts (PLSR, MLR, 1D-CNN, LSTM, CLNet).
5. Comprehensive Academic Results Table (Rc^2, RMSEC, Rp^2, RMSEP, RPD).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


def plot_spectral_curves(wavelengths, X_raw, X_snv, X_1st, save_path="outputs/spectral_curves_comparison.png"):
    """
    Plots raw and preprocessed spectra, highlighting key biochemical absorption peaks:
    - O-H absorption peaks (~930, 970, 1450 nm for moisture)
    - N-H absorption peaks (~915, 1480 nm for protein)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # 1. Raw Spectra
    for i in range(min(50, len(X_raw))):
        axes[0].plot(wavelengths, X_raw[i], color='#3366cc', alpha=0.35, linewidth=0.8)
    axes[0].plot(wavelengths, np.mean(X_raw, axis=0), color='#cc0000', linewidth=2.0, label='Mean Spectrum')
    axes[0].set_title("(a) Raw Wheat NIR Spectra", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Wavelength (nm)", fontsize=11)
    axes[0].set_ylabel("Reflectance / Intensity", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    # 2. SNV Preprocessed Spectra (Protein)
    for i in range(min(50, len(X_snv))):
        axes[1].plot(wavelengths, X_snv[i], color='#109618', alpha=0.35, linewidth=0.8)
    axes[1].plot(wavelengths, np.mean(X_snv, axis=0), color='#004411', linewidth=2.0, label='Mean SNV')
    axes[1].axvline(x=1480, color='red', linestyle=':', linewidth=1.5, label='N-H Peak (1480 nm)')
    axes[1].set_title("(b) SNV Preprocessed (Scatter Corrected)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Wavelength (nm)", fontsize=11)
    axes[1].set_ylabel("Standardized Value", fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')

    # 3. 1st Derivative Preprocessed Spectra (Moisture)
    for i in range(min(50, len(X_1st))):
        axes[2].plot(wavelengths, X_1st[i], color='#ff9900', alpha=0.35, linewidth=0.8)
    axes[2].plot(wavelengths, np.mean(X_1st, axis=0), color='#b35900', linewidth=2.0, label='Mean 1st-Deriv')
    axes[2].axvline(x=1450, color='blue', linestyle=':', linewidth=1.5, label='O-H Peak (1450 nm)')
    axes[2].set_title("(c) 1st Derivative (Inflection Peaks)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Wavelength (nm)", fontsize=11)
    axes[2].set_ylabel("Derivative Value", fontsize=11)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved spectral comparison plot to {save_path}")


def plot_feature_wavelengths(wavelengths, mean_spectrum, spa_indices, cars_indices, 
                             save_path="outputs/feature_wavelengths_selection.png"):
    """
    Plots full spectrum with SPA and CARS selected wavelengths highlighted.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300, sharex=True)
    
    # Plot 1: SPA Selection
    ax1.plot(wavelengths, mean_spectrum, color='#333333', linewidth=1.8, label='Mean Spectrum')
    for idx in spa_indices:
        ax1.axvline(x=wavelengths[idx], color='#d9534f', linestyle='--', alpha=0.8, linewidth=1.4)
    ax1.scatter(wavelengths[spa_indices], mean_spectrum[spa_indices], color='#d9534f', s=60, zorder=5, 
                label=f'SPA Selected Bands ({len(spa_indices)})')
    ax1.set_title("(a) Successive Projections Algorithm (SPA) Feature Bands", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Intensity", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # Plot 2: CARS Selection
    ax2.plot(wavelengths, mean_spectrum, color='#333333', linewidth=1.8, label='Mean Spectrum')
    for idx in cars_indices:
        ax2.axvline(x=wavelengths[idx], color='#0275d8', linestyle='--', alpha=0.8, linewidth=1.4)
    ax2.scatter(wavelengths[cars_indices], mean_spectrum[cars_indices], color='#0275d8', s=60, zorder=5, 
                label=f'CARS Selected Bands ({len(cars_indices)})')
    ax2.set_title("(b) Competitive Adaptive Reweighted Sampling (CARS) Feature Bands", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Wavelength (nm)", fontsize=11)
    ax2.set_ylabel("Intensity", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved feature wavelength plot to {save_path}")


def plot_prediction_scatter(y_cal_true, y_cal_pred, y_val_true, y_val_pred, 
                            m_cal, m_val, target_name="Protein", model_name="PLSR",
                            save_path="outputs/scatter_prediction.png"):
    """
    Generates academic Measured vs Predicted scatter plot matching Fig. 6 in Xuan et al. (2024)
    and Figs. 7 & 8 in Hu et al. (2025).
    """
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    
    all_vals = np.concatenate([y_cal_true, y_val_true, y_cal_pred, y_val_pred])
    min_val = np.min(all_vals) - 0.5
    max_val = np.max(all_vals) + 0.5
    
    # 1:1 perfect prediction diagonal line
    ax.plot([min_val, max_val], [min_val, max_val], color='#444444', linestyle='--', linewidth=1.5, label='1:1 Line')
    
    # Scatter points - Using Train Split (80%) and Test Split (20%)
    ax.scatter(y_cal_true, y_cal_pred, color='#2b5c8f', edgecolors='black', s=45, alpha=0.75, 
               label=f'Train Split (80%) (R²={m_cal["R2"]:.3f}, RMSEC={m_cal["RMSE"]:.3f})')
    ax.scatter(y_val_true, y_val_pred, color='#e6550d', edgecolors='black', s=65, marker='^', alpha=0.9, 
               label=f'Test Split (20%) (R²={m_val["R2"]:.3f}, RMSEP={m_val["RMSE"]:.3f})')
    
    ax.set_title(f"{model_name} Prediction: {target_name} Content\n(RPD = {m_val['RPD']:.2f})", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel(f"Measured Reference {target_name} (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"Predicted {target_name} (%)", fontsize=11, fontweight='bold')
    
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction scatter plot to {save_path}")


def plot_model_comparison_bars(results_df, target_name="Protein", save_path="outputs/models_comparison_bars.png"):
    """
    Bar chart comparing Rp^2, RMSEP, and RPD across models with cleanly rotated, non-overlapping labels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=300)
    
    models = results_df['Model'].values
    rp2 = results_df['Rp2'].values
    rmsep = results_df['RMSEP'].values
    rpd = results_df['RPD'].values
    x_indices = np.arange(len(models))

    # 1. Rp^2 (Higher is better)
    bars1 = axes[0].bar(x_indices, rp2, color='#1e40af', edgecolor='#0f172a', width=0.55)
    axes[0].set_title(f"Test Split Determination (Rp²)", fontsize=11, fontweight='bold', pad=8)
    axes[0].set_ylabel("Rp² Value", fontsize=10, fontweight='bold')
    min_rp2 = min(0.0, float(np.min(rp2)) - 0.1)
    axes[0].set_ylim([min_rp2, 1.12])
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars1:
        val = b.get_height()
        y_text = val + 0.02 if val >= 0 else val - 0.08
        va = 'bottom' if val >= 0 else 'top'
        axes[0].text(b.get_x() + b.get_width()/2., y_text, f"{val:.3f}", 
                     ha='center', va=va, fontsize=8.5, fontweight='bold')

    # 2. RMSEP (Lower is better)
    bars2 = axes[1].bar(x_indices, rmsep, color='#ea580c', edgecolor='#0f172a', width=0.55)
    axes[1].set_title(f"Test Split Error (RMSEP %)", fontsize=11, fontweight='bold', pad=8)
    axes[1].set_ylabel("RMSEP (%)", fontsize=10, fontweight='bold')
    axes[1].set_ylim([0, max(rmsep) * 1.25])
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width()/2., b.get_height() + 0.03, f"{b.get_height():.3f}", 
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # 3. RPD (Higher is better, >2.5 excellent)
    bars3 = axes[2].bar(x_indices, rpd, color='#16a34a', edgecolor='#0f172a', width=0.55)
    axes[2].axhline(y=2.5, color='red', linestyle='--', linewidth=1.5, label='Threshold (RPD=2.5)')
    axes[2].set_title(f"Residual Prediction Deviation (RPD)", fontsize=11, fontweight='bold', pad=8)
    axes[2].set_ylabel("RPD", fontsize=10, fontweight='bold')
    axes[2].set_ylim([0, max(rpd) * 1.20])
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    axes[2].legend(loc='upper right', fontsize=8.5)
    for b in bars3:
        axes[2].text(b.get_x() + b.get_width()/2., b.get_height() + 0.25, f"{b.get_height():.2f}", 
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    for ax in axes:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(models, rotation=42, ha='right', rotation_mode='anchor', fontsize=9, fontweight='bold')

    fig.suptitle(f"Model Comparison for {target_name} Content Prediction (80/20 Train-Test Split)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved model comparison bars to {save_path}")
