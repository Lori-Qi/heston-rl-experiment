import numpy as np
import warnings
import os
import json
import tensorflow as tf 

SMALL_POSITIVE_EPSILON = 1e-9 

class PriceSimulatorBase:
    def simulate_one_day(self, historical_prices):
        raise NotImplementedError('Subclasses must implement this methods')

    def simulate_days(self, historical_prices, days = 5):
        prices = []
        current_history = historical_prices.copy()

        for _ in range(days):
            next_price = self.simulate_one_day(current_history)
            prices.append(next_price)
            current_history = np.append(current_history, next_price)

        return np.array(prices)
    

class EnvHestonSimulator(PriceSimulatorBase):
    SMALL_POSITIVE_EPSILON = 1e-9

    def __init__(self, mu, kappa, theta, xi, rho, dt=1 / 252):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = np.clip(rho, -1.0 + self.SMALL_POSITIVE_EPSILON, 1.0 - self.SMALL_POSITIVE_EPSILON)
        self.dt = dt
        self.sqrt_dt = np.sqrt(dt)
        self._v_for_step = None
        self._validate_true_parameters()

    def _validate_true_parameters(self):
        feller_lhs = 2 * self.kappa * self.theta
        feller_rhs = self.xi**2 * (1 - 1e-6)
        if not (feller_lhs >= feller_rhs):
            warnings.warn(
                f"Feller EnvSim TRUE PARAMS warning: {feller_lhs:.4g} vs "
                f"{self.xi**2:.4g}",
                UserWarning,
            )
        if self.theta < 0 or self.kappa < 0 or self.xi <= self.SMALL_POSITIVE_EPSILON:
            raise ValueError("EnvSim: True variance params non-neg, xi strictly pos.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

    def set_initial_variance(self, v0):
        self._v_for_step = float(max(0.0, v0))

    def _generate_correlated_normals(self):
        Z1, Z2 = np.random.normal(0, 1, 2)
        Z_v = Z1
        Z_S = (self.rho * Z1 + np.sqrt(max(0, 1 - self.rho**2)) * Z2)
        return Z_S, Z_v

    def _internal_simulate_heston_step(self, S_prev, v_prev):
        Z_S, Z_v = self._generate_correlated_normals()
        dW_S = self.sqrt_dt * Z_S
        dW_v = self.sqrt_dt * Z_v

        v_prev_nn = max(v_prev, 0.0)
        sqrt_v_prev = np.sqrt(v_prev_nn)

        v_n = (
            v_prev
            + self.kappa * (self.theta - v_prev_nn) * self.dt
            + self.xi * sqrt_v_prev * dW_v
        )
        v_n = max(v_n, 0.0)

        S_n = S_prev * np.exp((self.mu - 0.5 * v_prev_nn) * self.dt + sqrt_v_prev * dW_S)
        S_n = max(float(S_n), self.SMALL_POSITIVE_EPSILON)

        return S_n, v_n

    def simulate_one_day(self, historical_prices):
        if self._v_for_step is None:
            raise RuntimeError("EnvSim: _v_for_step not set.")
        if not isinstance(historical_prices, np.ndarray) or not historical_prices.size:
            raise ValueError("EnvSim: hp non-empty array.")

        S_prev = historical_prices[-1]
        if S_prev <= self.SMALL_POSITIVE_EPSILON:
            S_prev = self.SMALL_POSITIVE_EPSILON

        v_prev = self._v_for_step
        S_next, v_next = self._internal_simulate_heston_step(S_prev, v_prev)
        self._v_for_step = v_next

        return S_next


class EstGBMSimulator(PriceSimulatorBase):
    SMALL_POSITIVE_EPSILON = 1e-9

    def __init__(self, dt=1/252):
        self.dt = dt
        self.sqrt_dt = np.sqrt(dt)

        self.est_mu    = None
        self.est_sigma = None
        self.is_fitted = False

    def update_parameters(self, historical_prices, min_required=252):

        prices = np.asarray(historical_prices, dtype=np.float64)

        if prices.size < min_required:
            warnings.warn(f"Need at least {min_required} prices, got {prices.size}.")
            return False
        if np.any(prices <= self.SMALL_POSITIVE_EPSILON):
            warnings.warn("Historical prices contain non-positive values.")
            return False

        log_ret = np.diff(np.log(prices))
        var_lr = np.var(log_ret, ddof=1)
        if var_lr <= self.SMALL_POSITIVE_EPSILON:
            warnings.warn("Log-return variance is almost 0 and cannot estimate sigma.")
            return False

        sigma_hat = np.sqrt(var_lr / self.dt)
        mu_hat = (np.mean(log_ret) / self.dt) + 0.5 * sigma_hat**2

        self.est_sigma = float(sigma_hat)
        self.est_mu = float(mu_hat)
        self.is_fitted = True
        return True

    def _simulate_one_step(self, S_prev):
        Z = np.random.normal()
        d_logS = (self.est_mu - 0.5 * self.est_sigma**2) * self.dt \
                 + self.est_sigma * self.sqrt_dt * Z
        S_next = S_prev * np.exp(d_logS)
        return max(S_next, self.SMALL_POSITIVE_EPSILON)

    def simulate_one_day(self, historical_prices):
        if not self.is_fitted:
            warnings.warn("simulate_one_day called before fitting.", RuntimeWarning)
            return None

        if (not isinstance(historical_prices, np.ndarray)) or historical_prices.size == 0:
            raise ValueError("historical_prices must be a non-empty np.ndarray.")

        S_prev = float(historical_prices[-1])
        if S_prev <= self.SMALL_POSITIVE_EPSILON:
            S_prev = self.SMALL_POSITIVE_EPSILON

        S_next = self._simulate_one_step(S_prev)

        return S_next
    

class EstFailReason:
    SUCCESS = "SUCCESS"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_V_LAGGED_FOR_OLS = "INVALID_V_LAGGED_FOR_OLS"
    LINALG_ERROR_OLS = "LINALG_ERROR_OLS"

    KAPPA_INVALID = "KAPPA_INVALID"
    THETA_DENOMINATOR_ISSUE = "THETA_DENOMINATOR_ISSUE"
    THETA_INVALID = "THETA_INVALID"

    XI_DENOMINATOR_ISSUE = "XI_DENOMINATOR_ISSUE"
    XI_INVALID = "XI_INVALID"

    RHO_INSUFFICIENT_POINTS = "RHO_INSUFFICIENT_POINTS"
    RHO_STD_DEV_ZERO = "RHO_STD_DEV_ZERO"
    RHO_NAN_OR_INF = "RHO_NAN_OR_INF"

    FELLER_CONDITION_NOT_MET = "FELLER_CONDITION_NOT_MET"
    DATA_ALIGNMENT_ISSUE = "DATA_ALIGNMENT_ISSUE"



class EstHestonSimulator(PriceSimulatorBase):
    SMALL_POSITIVE_EPSILON = 1e-9

    def __init__(self, dt=1/252):
        self.dt = dt
        self.sqrt_dt = np.sqrt(dt)
        self.est_mu = self.est_kappa = self.est_theta = None
        self.est_xi = self.est_rho = None
        self.is_fitted = False
        self._v_for_step = None

    def set_initial_variance(self, v0):
        self._v_for_step = float(max(0.0, v0))

    def update_parameters(self, historical_prices, historical_variances):
        S_hist = np.asarray(historical_prices)
        v_hist = np.asarray(historical_variances)
        min_required = 252 # minimum use 1 year data

        valid_idx = np.where(
            (S_hist[:-1] > self.SMALL_POSITIVE_EPSILON) &
            (S_hist[1:] > self.SMALL_POSITIVE_EPSILON)
        )[0]

        if len(valid_idx) < min_required:
            return False, EstFailReason.INSUFFICIENT_DATA

        S_prev = S_hist[:-1][valid_idx]
        S_curr = S_hist[1:][valid_idx]
        log_returns = np.log(S_curr / S_prev)

        v_lagged = v_hist[:-1][valid_idx]
        delta_v = v_hist[1:][valid_idx] - v_lagged

        valid_v_idx = v_lagged > self.SMALL_POSITIVE_EPSILON
        if np.sum(valid_v_idx) < 2:
            return False, EstFailReason.INVALID_V_LAGGED_FOR_OLS

        Y = delta_v[valid_v_idx]
        X = v_lagged[valid_v_idx]
        A = np.vstack([X, np.ones(len(X))]).T

        try:
            beta_1, beta_0 = np.linalg.lstsq(A, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return False, EstFailReason.LINALG_ERROR_OLS

        kappa_hat = -beta_1 / self.dt
        if kappa_hat <= self.SMALL_POSITIVE_EPSILON:
            return False, EstFailReason.KAPPA_INVALID

        if abs(beta_1) < self.SMALL_POSITIVE_EPSILON:
            return False, EstFailReason.THETA_DENOMINATOR_ISSUE

        theta_hat = beta_0 / (-beta_1)
        if theta_hat <= self.SMALL_POSITIVE_EPSILON:
            return False, EstFailReason.THETA_INVALID

        residuals = Y - (beta_0 + beta_1 * X)
        denom_xi = np.sqrt(X * self.dt)
        valid_denom = denom_xi > self.SMALL_POSITIVE_EPSILON
        if np.sum(valid_denom) < 2:
            return False, EstFailReason.XI_DENOMINATOR_ISSUE

        Z_v = residuals[valid_denom] / denom_xi[valid_denom]
        if len(Z_v) < 2:
            return False, EstFailReason.XI_DENOMINATOR_ISSUE

        xi_hat = np.std(Z_v)
        if xi_hat <= self.SMALL_POSITIVE_EPSILON:
            return False, EstFailReason.XI_INVALID

        mu_hat = np.mean(log_returns / self.dt + 0.5 * v_lagged[:len(log_returns)])

        log_rho = log_returns[valid_v_idx]
        v_rho = X
        res_rho = residuals

        num_zs = log_rho - (mu_hat - 0.5 * v_rho) * self.dt
        den_zs = np.sqrt(v_rho * self.dt)
        num_zv = res_rho
        den_zv = xi_hat * np.sqrt(v_rho * self.dt)

        mask = (den_zs > self.SMALL_POSITIVE_EPSILON) & (den_zv > self.SMALL_POSITIVE_EPSILON)
        if np.sum(mask) < 2:
            return False, EstFailReason.RHO_INSUFFICIENT_POINTS

        Z_S = num_zs[mask] / den_zs[mask]
        Z_V = num_zv[mask] / den_zv[mask]

        if np.std(Z_S) < self.SMALL_POSITIVE_EPSILON or np.std(Z_V) < self.SMALL_POSITIVE_EPSILON:
            return False, EstFailReason.RHO_STD_DEV_ZERO

        rho_hat = np.corrcoef(Z_S, Z_V)[0, 1]
        if np.isnan(rho_hat) or np.isinf(rho_hat):
            return False, EstFailReason.RHO_NAN_OR_INF

        if 2 * kappa_hat * theta_hat < xi_hat ** 2 * (1 - 1e-6):
            return False, EstFailReason.FELLER_CONDITION_NOT_MET

        self.est_mu = mu_hat
        self.est_kappa = kappa_hat
        self.est_theta = theta_hat
        self.est_xi = xi_hat
        self.est_rho = rho_hat
        self.is_fitted = True

        return True, EstFailReason.SUCCESS

    def _internal_simulate_heston_step(self, S_prev, v_prev):
        if not self.is_fitted:
            raise RuntimeError("Simulator not fitted.")

        mu = self.est_mu
        k = self.est_kappa
        th = self.est_theta
        xi = self.est_xi
        rho = self.est_rho

        if any(p is None for p in [mu, k, th, xi, rho]):
            raise RuntimeError("Estimated parameters contain None.")

        S_prev = float(S_prev)
        v_prev = float(v_prev)

        if S_prev <= 0:
            S_prev = self.SMALL_POSITIVE_EPSILON
        if v_prev < 0:
            v_prev = 0.0

        Z1, Z2 = np.random.normal(0, 1, 2)
        Z_v = Z1
        rho_c = np.clip(rho, -1 + 1e-8, 1 - 1e-8)
        Z_S = rho_c * Z1 + np.sqrt(max(0, 1 - rho_c ** 2)) * Z2

        dW_S = self.sqrt_dt * Z_S
        dW_v = self.sqrt_dt * Z_v

        v_prev_nn = max(v_prev, 0.0)
        sqrt_v_prev = np.sqrt(v_prev_nn)

        v_n = v_prev + k * (th - v_prev_nn) * self.dt + xi * sqrt_v_prev * dW_v
        v_n = max(v_n, 0.0)

        S_n = S_prev * np.exp((mu - 0.5 * v_prev_nn) * self.dt + sqrt_v_prev * dW_S)
        S_n = max(S_n, self.SMALL_POSITIVE_EPSILON)

        return S_n, v_n

    def simulate_one_day(self, historical_prices):
        if not self.is_fitted:
            warnings.warn("simulate_one_day called before fitting.")
            return None

        if self._v_for_step is None:
            raise RuntimeError("Initial variance (_v_for_step) not set.")

        if not isinstance(historical_prices, np.ndarray) or historical_prices.size == 0:
            raise ValueError("historical_prices must be a non-empty np.ndarray.")

        S_prev = historical_prices[-1]
        if S_prev <= self.SMALL_POSITIVE_EPSILON:
            warnings.warn(f"S_prev ({S_prev:.4g}) <= 0; replacing with epsilon.")
            S_prev = self.SMALL_POSITIVE_EPSILON

        v_prev = self._v_for_step

        try:
            S_next, v_next = self._internal_simulate_heston_step(S_prev, v_prev)
        except Exception as e:
            warnings.warn(f"Simulation error: {e}", RuntimeWarning)
            return None

        self._v_for_step = v_next
        return S_next


class WGANPriceSimulator(PriceSimulatorBase):

    def __init__(self, model_dir="trained_heston_wgan", z_dim=32):
        self.z_dim = z_dim
        gen_path = f"{model_dir}/generator.h5"
        stats_path = f"{model_dir}/stats.json"

        if not os.path.exists(gen_path) or not os.path.exists(stats_path):
            raise FileNotFoundError("Generator weights or stats.json not found.")

        self.generator = tf.keras.models.load_model(gen_path, compile=False)
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.original_mean = np.array(stats["mean"], dtype=np.float32)
        self.original_std = np.array(stats["std"], dtype=np.float32)

        self.current_mean = self.original_mean.copy()
        self.current_std = self.original_std.copy()

        self._flat_norm = None
        self._cursor = 0
        self._path_length = None

        self._is_fitted = False
        self._recentered = False

    def recenter_to_historical_data(self, S_hist, v_hist):
        lnS_hist = np.log(np.maximum(S_hist, 1e-9))

        hist_flat = np.concatenate([lnS_hist, v_hist])

        self.current_mean = hist_flat.mean()
        self.current_std = hist_flat.std()

        if self.current_std < 1e-8:
            self.current_std = 1.0

        self._recentered = True
        self._is_fitted = True

        print(f"WGAN re-centered: mean={self.current_mean:.4f}, std={self.current_std:.4f}")

    @property
    def is_fitted(self):
        return self._is_fitted

    def reset(self):
        self._flat_norm = None
        self._cursor = 0

    def _sample_new_path(self):
        z = tf.random.normal((1, self.z_dim))
        flat_norm = self.generator(z, training=False).numpy()[0] 

        if self._recentered:
            normalized = (flat_norm + 3.0) / 6.0
            flat = normalized * self.current_std + self.current_mean
        else:
            flat = flat_norm * self.original_std + self.original_mean

        self._flat_norm = flat
        self._cursor = 0

        self._path_length = len(flat) // 2

    def simulate_one_day(self, historical_prices):
        if self._flat_norm is None or self._cursor >= self._path_length:
            self._sample_new_path()

        lnS_path = self._flat_norm[:self._path_length]

        lnS_current = lnS_path[self._cursor]
        price = np.exp(lnS_current)

        if len(self._flat_norm) > self._path_length:
            v_path = self._flat_norm[self._path_length:]
            if self._cursor < len(v_path):
                self._v_for_step = max(v_path[self._cursor], 1e-8)
            else:
                self._v_for_step = 0.04 

        self._cursor += 1
        return float(max(price, 1e-9))


class ReplaySimulator(PriceSimulatorBase):
    def __init__(self, price_path):
        self.path = list(price_path)
        self.idx = 0
        self._v_for_step = None
    
    def reset(self):
        self.idx = 0
    
    def simulate_one_day(self, hist):
        if self.idx >= len(self.path):
            raise RuntimeError("ReplaySimulator: path exhausted.")
        
        p = float(self.path[self.idx])
        self.idx += 1
        return p