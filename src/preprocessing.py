"""
preprocessing.py
Implements spectral signal preprocessing techniques as analyzed in:
- Xuan et al. (2024): MSC, 1st Derivative, SG smoothing.
- Hu et al. (2025): SNV (best for Protein), 1st Derivative (best for Moisture).
"""

import numpy as np
from scipy.signal import savgol_filter

def snv(X):
    """
    Standard Normal Variate (SNV) transformation.
    Normalizes each individual spectrum by subtracting its mean and dividing by its standard deviation.
    Eliminates scattering variations and baseline shifts caused by grain particle size and curvature.
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    # Avoid division by zero
    std[std == 0] = 1.0
    return (X - mean) / std


class MSC:
    """
    Multiplicative Scatter Correction (MSC).
    Corrects multiplicative and additive light scattering variations by regressing
    each spectrum against an ideal reference spectrum (typically the mean calibration spectrum).
    """
    def __init__(self):
        self.ref_spectrum = None

    def fit(self, X):
        self.ref_spectrum = np.mean(X, axis=0)
        return self

    def transform(self, X):
        if self.ref_spectrum is None:
            self.ref_spectrum = np.mean(X, axis=0)
            
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            # Linear fit: X[i] = m * ref + c
            fit = np.polyfit(self.ref_spectrum, X[i], 1, full=True)
            slope = fit[0][0]
            intercept = fit[0][1]
            if abs(slope) < 1e-8:
                slope = 1.0
            X_msc[i] = (X[i] - intercept) / slope
        return X_msc

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def savitzky_golay_derivative(X, window_length=11, polyorder=2, deriv=1):
    """
    Savitzky-Golay filter to compute the 1st or 2nd derivative of spectral curves.
    Removes baseline slope drift and accentuates subtle chemical absorption peaks (e.g. O-H, N-H).
    """
    # window_length must be odd and greater than polyorder
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 3
        if window_length % 2 == 0:
            window_length += 1
            
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)


def savitzky_golay_smooth(X, window_length=11, polyorder=2):
    """
    Savitzky-Golay smoothing filter to suppress high-frequency instrument noise.
    """
    return savitzky_golay_derivative(X, window_length=window_length, polyorder=polyorder, deriv=0)


def apply_preprocessing(X_train, X_test=None, method='snv', **kwargs):
    """
    Unified dispatcher for spectral preprocessing methods.
    Supports: 'raw', 'snv', 'msc', '1st', 'sg'.
    Avoids data leakage by fitting parameters (e.g. for MSC) on training data only.
    """
    method = method.lower()
    
    if method == 'raw':
        return (X_train, X_test) if X_test is not None else X_train
    
    elif method == 'snv':
        X_train_p = snv(X_train)
        X_test_p = snv(X_test) if X_test is not None else None
        return (X_train_p, X_test_p) if X_test is not None else X_train_p
        
    elif method == 'msc':
        msc_transformer = MSC()
        X_train_p = msc_transformer.fit_transform(X_train)
        X_test_p = msc_transformer.transform(X_test) if X_test is not None else None
        return (X_train_p, X_test_p) if X_test is not None else X_train_p
        
    elif method in ['1st', 'derivative', '1st_derivative']:
        w = kwargs.get('window_length', 11)
        p = kwargs.get('polyorder', 2)
        X_train_p = savitzky_golay_derivative(X_train, window_length=w, polyorder=p, deriv=1)
        X_test_p = savitzky_golay_derivative(X_test, window_length=w, polyorder=p, deriv=1) if X_test is not None else None
        return (X_train_p, X_test_p) if X_test is not None else X_train_p
        
    elif method in ['sg', 'smooth', 'smoothing']:
        w = kwargs.get('window_length', 11)
        p = kwargs.get('polyorder', 2)
        X_train_p = savitzky_golay_smooth(X_train, window_length=w, polyorder=p)
        X_test_p = savitzky_golay_smooth(X_test, window_length=w, polyorder=p) if X_test is not None else None
        return (X_train_p, X_test_p) if X_test is not None else X_train_p
        
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


if __name__ == "__main__":
    from data_loader import load_wheat_chemical_data, spxy_split
    wl, X, yp, ym, _ = load_wheat_chemical_data()
    cal_idx, val_idx = spxy_split(X, yp)
    X_train, X_test = X[cal_idx], X[val_idx]
    
    print("Testing Preprocessing methods on Wheat NIR data:")
    for m in ['raw', 'snv', 'msc', '1st', 'sg']:
        xtr, xte = apply_preprocessing(X_train, X_test, method=m)
        print(f"  Method [{m.upper()}]: Train shape={xtr.shape}, Test shape={xte.shape}, Mean={xtr.mean():.4f}, Std={xtr.std():.4f}")
