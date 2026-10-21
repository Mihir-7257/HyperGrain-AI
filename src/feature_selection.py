"""
feature_selection.py
Implements characteristic wavelength selection algorithms from the research papers:
1. SPA (Successive Projections Algorithm) - Araújo et al. (2001), Xuan et al. (2024), Hu et al. (2025).
2. CARS (Competitive Adaptive Reweighted Sampling) - Li et al. (2009), Hu et al. (2025).
3. PCA (Principal Component Analysis) - Baseline dimensionality reduction.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


class SPA:
    """
    Successive Projections Algorithm (SPA) for wavelength selection.
    Uses vector orthogonal projection to select a subset of spectral variables
    with minimal collinearity, choosing the optimal subset size via cross-validated MLR.
    """
    def __init__(self, k_max=15, cv=5):
        self.k_max = k_max
        self.cv = cv
        self.selected_indices_ = None
        self.selected_wavelengths_ = None
        self.rmse_curve_ = []

    def fit(self, X, y, wavelengths=None):
        N, J = X.shape
        k_max = min(self.k_max, J - 1, N - 2)
        
        # Center X and y
        X_mean = np.mean(X, axis=0)
        X_c = X - X_mean
        
        # Candidate projection sequences starting from each variable j
        # To keep computation fast, test candidate starting variables
        step = max(1, J // 30)
        candidate_starts = list(range(0, J, step))
        if 0 not in candidate_starts:
            candidate_starts.insert(0, 0)
            
        best_overall_rmse = float('inf')
        best_overall_vars = None
        best_rmse_curve = None

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for init_j in candidate_starts:
            # Step 1: Projection sequence starting from init_j
            vars_chain = [init_j]
            v_chain = []
            
            # Remaining projection matrix
            P = np.copy(X_c)
            # Normalize starting vector
            norm_j = np.linalg.norm(P[:, init_j])
            if norm_j < 1e-10:
                continue
            v_chain.append(P[:, init_j] / norm_j)
            
            for k in range(1, k_max):
                # Project all remaining columns orthogonal to the latest chosen vector
                v_last = v_chain[-1]
                # P = P - v_last * (v_last^T * P)
                proj = np.outer(v_last, v_last.T @ P)
                P = P - proj
                
                # Zero out already selected variables
                for sel in vars_chain:
                    P[:, sel] = 0.0
                    
                # Variable with the maximum remaining projected vector length is selected
                norms = np.linalg.norm(P, axis=0)
                next_var = int(np.argmax(norms))
                if next_var in vars_chain or norms[next_var] < 1e-10:
                    break
                    
                vars_chain.append(next_var)
                v_chain.append(P[:, next_var] / norms[next_var])

            # Step 2: Cross-validate MLR for subsets of size 1 to len(vars_chain)
            rmse_curve = []
            for k in range(1, len(vars_chain) + 1):
                subset = vars_chain[:k]
                fold_errors = []
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X[train_idx][:, subset], X[val_idx][:, subset]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    
                    reg = LinearRegression().fit(X_tr, y_tr)
                    y_pred = reg.predict(X_val)
                    fold_errors.append(np.mean((y_val - y_pred) ** 2))
                rmse_curve.append(np.sqrt(np.mean(fold_errors)))

            min_idx = int(np.argmin(rmse_curve))
            if rmse_curve[min_idx] < best_overall_rmse:
                best_overall_rmse = rmse_curve[min_idx]
                best_overall_vars = vars_chain[:min_idx + 1]
                best_rmse_curve = rmse_curve

        self.selected_indices_ = sorted(best_overall_vars) if best_overall_vars is not None else list(range(min(10, J)))
        self.rmse_curve_ = best_rmse_curve
        if wavelengths is not None:
            self.selected_wavelengths_ = wavelengths[self.selected_indices_]
            
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, wavelengths=None):
        return self.fit(X, y, wavelengths=wavelengths).transform(X)


class CARS:
    """
    Competitive Adaptive Reweighted Sampling (CARS).
    Selects optimal wavelengths based on absolute regression coefficients in PLS,
    exponentially decreasing function (EDF), and adaptive reweighted sampling.
    """
    def __init__(self, num_runs=50, cv=5, n_components=5):
        self.num_runs = num_runs
        self.cv = cv
        self.n_components = n_components
        self.selected_indices_ = None
        self.selected_wavelengths_ = None
        self.rmsecv_history_ = []

    def fit(self, X, y, wavelengths=None):
        N, J = X.shape
        # Parameterize Exponentially Decreasing Function (EDF)
        # r_i = a * exp(-k * i)
        a = (J / 2.0) ** (1.0 / (self.num_runs - 1))
        k = np.log(a) / (self.num_runs - 1)

        active_indices = np.arange(J)
        best_rmsecv = float('inf')
        best_subset = active_indices
        rmsecv_list = []

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for i in range(self.num_runs):
            # 1. Fit PLS on current active variables
            X_sub = X[:, active_indices]
            n_comp = min(self.n_components, X_sub.shape[1], X_sub.shape[0] - 2)
            if n_comp < 1:
                break
                
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_sub, y)
            coef = np.abs(pls.coef_.flatten())

            # 2. Cross-validation RMSECV
            fold_errors = []
            for tr_idx, v_idx in kf.split(X):
                p_sub = PLSRegression(n_components=n_comp)
                p_sub.fit(X[tr_idx][:, active_indices], y[tr_idx])
                preds = p_sub.predict(X[v_idx][:, active_indices]).flatten()
                fold_errors.append(np.mean((y[v_idx] - preds) ** 2))
            rmsecv = np.sqrt(np.mean(fold_errors))
            rmsecv_list.append(rmsecv)

            if rmsecv < best_rmsecv:
                best_rmsecv = rmsecv
                best_subset = active_indices.copy()

            # 3. Ratio of wavelengths to retain via EDF
            ratio = np.exp(-k * (i + 1))
            num_retain = max(2, int(np.round(J * ratio)))
            if num_retain >= len(active_indices):
                num_retain = max(2, len(active_indices) - 1)

            # 4. Adaptive Reweighted Sampling (ARS)
            weights = coef / (np.sum(coef) + 1e-8)
            # Sample based on weights
            sampled_pos = np.random.choice(len(active_indices), size=num_retain, replace=True, p=weights)
            unique_pos = np.unique(sampled_pos)
            active_indices = active_indices[unique_pos]
            if len(active_indices) <= 2:
                break

        self.selected_indices_ = sorted(best_subset)
        self.rmsecv_history_ = rmsecv_list
        if wavelengths is not None:
            self.selected_wavelengths_ = wavelengths[self.selected_indices_]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, wavelengths=None):
        return self.fit(X, y, wavelengths=wavelengths).transform(X)


class PCAReducer:
    """
    Principal Component Analysis (PCA) baseline dimensionality reduction.
    """
    def __init__(self, n_components=0.99):
        self.pca = PCA(n_components=n_components)
        self.selected_indices_ = None

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X, y=None):
        return self.pca.fit_transform(X)


if __name__ == "__main__":
    from data_loader import load_wheat_chemical_data, spxy_split
    from preprocessing import apply_preprocessing

    wl, X, yp, ym, _ = load_wheat_chemical_data()
    cal_idx, val_idx = spxy_split(X, yp)
    X_train, X_test = X[cal_idx], X[val_idx]
    y_train, y_test = yp[cal_idx], yp[val_idx]

    # Preprocess with SNV
    X_train_snv, X_test_snv = apply_preprocessing(X_train, X_test, method='snv')

    print("Running Successive Projections Algorithm (SPA)...")
    spa = SPA(k_max=15)
    X_train_spa = spa.fit_transform(X_train_snv, y_train, wavelengths=wl)
    print(f"  SPA selected {len(spa.selected_indices_)} wavelengths:")
    print(" ", [f"{w:.0f}nm" for w in spa.selected_wavelengths_])

    print("Running Competitive Adaptive Reweighted Sampling (CARS)...")
    cars = CARS(num_runs=30, n_components=5)
    X_train_cars = cars.fit_transform(X_train_snv, y_train, wavelengths=wl)
    print(f"  CARS selected {len(cars.selected_indices_)} wavelengths:")
    print(" ", [f"{w:.0f}nm" for w in cars.selected_wavelengths_])
