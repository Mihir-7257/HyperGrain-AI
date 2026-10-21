"""
models.py
Implements regression models for grain Protein and Moisture content prediction:
1. Chemometric Baselines:
   - PLSR (Partial Least Squares Regression with LV cross-validation)
   - MLR (Multiple Linear Regression)
2. Deep Learning Models:
   - 1D-CNN (Local feature extraction)
   - LSTM (Long-range sequential wavelength relationships)
   - CLNet (SOTA Hybrid 1D-CNN + LSTM from Hu et al., 2025)
3. Evaluator:
   - R^2 (Rc^2, Rp^2), RMSE (RMSEC, RMSEP), RPD (Residual Prediction Deviation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold


def compute_metrics(y_true, y_pred):
    """
    Computes R^2, RMSE, and RPD metrics as defined in Xuan et al. (2024) and Hu et al. (2025).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    sd_true = np.std(y_true)
    rpd_sd = sd_true / (rmse + 1e-8)
    
    # Paper formula: RPD = 1 / sqrt(1 - R^2) for R^2 < 1.0
    if r2 < 1.0 and r2 >= 0:
        rpd_formula = 1.0 / np.sqrt(1.0 - r2 + 1e-8)
    elif r2 >= 1.0:
        rpd_formula = 25.0
    else:
        rpd_formula = rpd_sd
        
    return {
        'R2': float(r2),
        'RMSE': float(rmse),
        'RPD': float(rpd_formula),
        'RPD_SD': float(rpd_sd)
    }


class TunedPLSR:
    """
    Partial Least Squares Regression with automatic Latent Variable (LV) selection
    via K-fold cross-validation, optimizing RMSECV.
    """
    def __init__(self, max_components=15, cv=5):
        self.max_components = max_components
        self.cv = cv
        self.best_n_components = 5
        self.model = None

    def fit(self, X, y):
        max_c = min(self.max_components, X.shape[1], X.shape[0] - 2)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        best_rmsecv = float('inf')
        best_c = 1

        for c in range(1, max_c + 1):
            fold_rmses = []
            for tr_idx, val_idx in kf.split(X):
                pls = PLSRegression(n_components=c)
                pls.fit(X[tr_idx], y[tr_idx])
                preds = pls.predict(X[val_idx]).flatten()
                fold_rmses.append(np.mean((y[val_idx] - preds) ** 2))
            rmsecv = np.sqrt(np.mean(fold_rmses))
            if rmsecv < best_rmsecv:
                best_rmsecv = rmsecv
                best_c = c

        self.best_n_components = best_c
        self.model = PLSRegression(n_components=best_c)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()


class MLRBaseline:
    """
    Multiple Linear Regression baseline (with slight L2 regularizer for numerical stability).
    """
    def __init__(self, alpha=1e-4):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()


class SVRBaseline:
    """
    Support Vector Regression (SVR) with RBF kernel.
    Commonly used in chemometrics for non-linear spectral mapping.
    """
    def __init__(self, C=10.0, epsilon=0.1):
        from sklearn.svm import SVR
        self.model = SVR(kernel='rbf', C=C, epsilon=epsilon)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()


class RandomForestBaseline:
    """
    Random Forest Regressor baseline.
    """
    def __init__(self, n_estimators=100, random_state=42):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()


# ==========================================
# DEEP LEARNING ARCHITECTURES (PYTORCH)
# ==========================================

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class OneDCNNModel(nn.Module):
    """
    1D-CNN regression model as depicted in Hu et al. (2025), Figure 1.
    Three convolutional blocks with kernel sizes 7, 5, and 3, followed by pooling and dense layers.
    """
    def __init__(self, in_features=141, out_features=1):
        super().__init__()
        # Input shape: (Batch, 1, in_features)
        self.block1 = nn.Sequential(
            Conv1DBlock(1, 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            Conv1DBlock(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            Conv1DBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_features)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)  # (B, 128, 1)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class LSTMModel(nn.Module):
    """
    LSTM regression model as depicted in Hu et al. (2025), Figure 2.
    Single-layer LSTM with hidden_size=128 processing sequential wavelengths.
    """
    def __init__(self, in_features=141, hidden_size=128, out_features=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        if x.ndim == 2:
            # (Batch, Seq_Len, 1)
            x = x.unsqueeze(-1)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use last hidden state
        last_hidden = hn[-1]  # (Batch, hidden_size)
        return self.fc(last_hidden)


class CLNet(nn.Module):
    """
    CLNet: Hybrid 1D-CNN + LSTM Network (Hu et al., 2025, Figure 3).
    Extracts local spectral absorption peaks via 1D-CNN, models long-range
    wavelength dependencies with an LSTM layer, and performs chemical regression.
    """
    def __init__(self, in_features=141, out_features=1, lstm_hidden=128):
        super().__init__()
        # 1D-CNN Feature Extractor
        self.block1 = nn.Sequential(
            Conv1DBlock(1, 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            Conv1DBlock(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            Conv1DBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Adapt feature map length to sequence for LSTM
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        
        # Regression Head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_features)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (Batch, 1, In_Features)
            
        # 1D-CNN feature extraction
        f = self.block1(x)
        f = self.block2(f)
        f = self.block3(f)
        f = self.adaptive_pool(f)  # (Batch, 128, 16)
        
        # Transpose for LSTM: (Batch, Seq_Len=16, Feat_Dim=128)
        f = f.permute(0, 2, 1)
        
        # LSTM sequence modeling
        lstm_out, (hn, cn) = self.lstm(f)
        last_hidden = hn[-1]  # (Batch, lstm_hidden)
        
        out = self.fc(last_hidden)
        return out


# ==========================================
# PYTORCH WRAPPER FOR SCIKIT-LEARN INTERFACE
# ==========================================

class PyTorchRegressorWrapper:
    """
    Clean wrapper providing .fit() and .predict() methods for PyTorch models,
    with automatic target normalization and learning rate scheduling.
    """
    def __init__(self, model_class, in_features=141, out_features=1, epochs=150, batch_size=16, lr=0.003, weight_decay=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model_class
        self.in_features = in_features
        self.out_features = out_features
        self.model = model_class(in_features=in_features, out_features=out_features).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_history = []
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X, y):
        # Normalize target y for smooth gradient backpropagation
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) if float(np.std(y)) > 1e-6 else 1.0
        y_scaled = (y - self.y_mean) / self.y_std

        self.model.train()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_scaled, dtype=torch.float32)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(-1)
            
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        criterion = nn.MSELoss()
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            scheduler.step()
            avg_loss = total_loss / len(dataset)
            self.loss_history.append(avg_loss)
            
        return self

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds_scaled = self.model(X_t).cpu().numpy()
        # Unscale back to physical chemical percentage
        preds = preds_scaled * self.y_std + self.y_mean
        return preds.flatten() if preds.shape[1] == 1 else preds


if __name__ == "__main__":
    from data_loader import load_wheat_chemical_data, spxy_split
    from preprocessing import apply_preprocessing

    wl, X, yp, ym, _ = load_wheat_chemical_data()
    cal_idx, val_idx = spxy_split(X, yp)
    X_tr, X_val = X[cal_idx], X[val_idx]
    y_tr, y_val = yp[cal_idx], yp[val_idx]

    # Preprocess with SNV
    X_tr_p, X_val_p = apply_preprocessing(X_tr, X_val, method='snv')

    print("=== Testing PLSR Baseline ===")
    pls = TunedPLSR()
    pls.fit(X_tr_p, y_tr)
    pred_val = pls.predict(X_val_p)
    m = compute_metrics(y_val, pred_val)
    print(f"PLSR Best LVs={pls.best_n_components}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, RPD={m['RPD']:.2f}")

    print("=== Testing CLNet (1D-CNN + LSTM) ===")
    clnet = PyTorchRegressorWrapper(CLNet, in_features=X_tr_p.shape[1], epochs=100, lr=0.002)
    clnet.fit(X_tr_p, y_tr)
    pred_val_cl = clnet.predict(X_val_p)
    m_cl = compute_metrics(y_val, pred_val_cl)
    print(f"CLNet: R2={m_cl['R2']:.4f}, RMSE={m_cl['RMSE']:.4f}, RPD={m_cl['RPD']:.2f}")
