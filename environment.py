import numpy as np
import warnings
from simulators import EnvHestonSimulator

# Heston Env
class Heston_Env:
    SMALL_POSITIVE_EPSILON = 1e-9

    def __init__(self, S0, W0, v0,
                 mu, kappa, theta, xi, rho,
                 r, T,
                 gamma = 1.5,
                 num_steps=504, future_steps=5, dt=1/252):

        self.S0 = S0
        self.W0 = W0
        self.v0 = v0

        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi

        self.rho = np.clip(
            rho,
            -1.0 + self.SMALL_POSITIVE_EPSILON,
            1.0 - self.SMALL_POSITIVE_EPSILON
        ) # correlation of St and vt
        self.gamma = gamma

        self.r = r # annulised
        self.T = T
        self.num_steps = num_steps
        self.future_steps = future_steps
        self.dt = dt
        self.sqrt_dt = np.sqrt(dt)
        self.daily_portfolio_returns_episode = [] # utility
        self.raw_daily_returns_episode = [] # raw return

        if not (2 * self.kappa * self.theta >= self.xi**2 * (1 - 1e-6)):
            warnings.warn(
                f"Feller condition not numerically met "
                f"({2*self.kappa*self.theta:.4f} vs {self.xi**2:.4f})",
                UserWarning
            )

        if self.v0 < 0:
            raise ValueError("Initial variance v0 must be non-negative.")

        if self.theta < 0 or self.kappa < 0 or self.xi <= self.SMALL_POSITIVE_EPSILON:
            raise ValueError("Variance parameters must be non-negative and xi strictly positive.")

        # hist data: S0 + 125 steps -> 126 steps
        self.S_hist = np.zeros(self.num_steps + 1)
        self.v_hist = np.zeros(self.num_steps + 1)
        self.W_hist = np.zeros(self.num_steps + 1)

        # future data: hist_data[-1] + 5 steps -> 6
        self.S_future = np.zeros(self.future_steps + 1)
        self.v_future = np.zeros(self.future_steps + 1)
        self.W_future = np.zeros(self.future_steps + 1)

        self.daily_portfolio_returns_episode = [] # now store the utilities

        self.S_hist[0] = self.S0
        self.v_hist[0] = self.v0
        self.W_hist[0] = self.W0

        self._generate_historical_paths()

        self.current_step_in_episode = 0
        self.is_done = False

        if 'EnvHestonSimulator' in globals() and callable(EnvHestonSimulator):
            self.true_simulator = EnvHestonSimulator(
                mu=self.mu,
                kappa=self.kappa,
                theta=self.theta,
                xi=self.xi,
                rho=self.rho,
                dt=self.dt
            )
        else:
            raise RuntimeError("EnvHestonSimulator not defined. Run simulator cell before this.")


    def _power_utility_return(self, r):
        if abs(self.gamma - 1.0) < 1e-12:
            base_utility = np.log1p(r)
        else:
            base_utility = ((1 + r) ** (1 - self.gamma) - 1) / (1 - self.gamma)

        return 5.0 * base_utility + 0.005  #

    def _generate_correlated_normals(self):
        Z1 = np.random.normal()
        Z2 = np.random.normal()

        Z_v = Z1
        Z_S = self.rho * Z1 + np.sqrt(max(0, 1 - self.rho**2)) * Z2

        return Z_S, Z_v

    def _simulate_true_step_internal(self, S_prev, v_prev):
        Z_S, Z_v = self._generate_correlated_normals()
        dW_S = self.sqrt_dt * Z_S
        dW_v = self.sqrt_dt * Z_v

        v_prev = max(v_prev, 0)
        sqrt_v = np.sqrt(v_prev)

        v_next = v_prev + self.kappa * (self.theta - v_prev) * self.dt + self.xi * sqrt_v * dW_v
        v_next = max(v_next, 0)

        S_next = S_prev * np.exp((self.mu - 0.5 * v_prev) * self.dt + sqrt_v * dW_S)
        S_next = max(S_next, self.SMALL_POSITIVE_EPSILON)

        return S_next, v_next

    def _generate_historical_paths(self):
        for t in range(1, self.num_steps + 1):
            S_prev = self.S_hist[t - 1]
            v_prev = self.v_hist[t - 1]
            self.S_hist[t], self.v_hist[t] = self._simulate_true_step_internal(S_prev, v_prev)
            self.W_hist[t] = self.W_hist[t - 1]

    def reset(self):
        self.S_future.fill(0)
        self.v_future.fill(0)
        self.W_future.fill(0)

        self.S_future[0] = self.S_hist[-1]
        self.v_future[0] = self.v_hist[-1]
        self.W_future[0] = self.W_hist[-1]

        self.daily_portfolio_returns_episode = []
        self.raw_daily_returns_episode = []
        self.current_step_in_episode = 0
        self.is_done = False

        return self.get_state()

    def step(self, action, simulator=None):
        if self.is_done:
            raise RuntimeError("Episode has ended, call reset().")

        sim = simulator or self.true_simulator

        # set initial variance v0
        if hasattr(sim, 'set_initial_variance'):
            sim.set_initial_variance(self.v_future[self.current_step_in_episode])

        S_next = sim.simulate_one_day(self.get_full_price_history())
        if S_next is None:
            raise RuntimeError(f"Simulator {type(sim).__name__} returned None.")

        if hasattr(sim, '_v_for_step'):
            v_next = sim._v_for_step
        else:
            v_next = self.v_future[self.current_step_in_episode]
            warnings.warn(f"{type(sim).__name__} missing _v_for_step; v_current used.")

        i = self.current_step_in_episode 
        self.v_future[i + 1] = v_next

        r_stock = (S_next - self.S_future[i]) / self.S_future[i] if self.S_future[i] != 0 else 0
        r_riskfree = self.r * self.dt
        r_portfolio = float(action) * r_stock + (1 - float(action)) * r_riskfree

        self.W_future[i + 1] = self.W_future[i] * (1 + r_portfolio)

        util_t = self._power_utility_return(r_portfolio)   
        self.daily_portfolio_returns_episode.append(util_t)
        self.raw_daily_returns_episode.append(r_portfolio) 


        self.current_step_in_episode += 1
        self.is_done = self.current_step_in_episode >= self.future_steps

        info = {
            'stock_price': self.S_future[self.current_step_in_episode],
            'variance': self.v_future[self.current_step_in_episode], # instaneous variance
            'sqrt_volatility': np.sqrt(max(self.v_future[self.current_step_in_episode], 0)),
            'wealth': self.W_future[self.current_step_in_episode],
            'stock_return_simulated': r_stock,
            'portfolio_return': r_portfolio,
            'action_taken': float(action),
            'daily_portfolio_returns_episode': self.daily_portfolio_returns_episode.copy()
        }

        return self.get_state(), util_t, self.is_done, info

    def get_full_price_history(self):
        if self.current_step_in_episode == 0:
            return self.S_hist
        return np.concatenate((
            self.S_hist,
            self.S_future[1 : self.current_step_in_episode + 1]
        ))

    def get_full_variance_history(self):
        if self.current_step_in_episode == 0:
            return self.v_hist
        return np.concatenate((
            self.v_hist,
            self.v_future[1 : self.current_step_in_episode + 1]
        ))

    def get_historical_price_data(self):
        return self.S_hist

    def get_historical_variance_data(self):
        return self.v_hist

    def get_state(self, window=21):
        S = self.S_future[self.current_step_in_episode]
        v = self.v_future[self.current_step_in_episode] # instaneous variance
        W = self.W_future[self.current_step_in_episode]
        sqrt_v = np.sqrt(max(v, 0))

        hist = self.get_full_price_history()
        realized_vol = 0

        if len(hist) > window + 1:
            hist = hist[-(window + 1):]
            hist = hist[hist > self.SMALL_POSITIVE_EPSILON]

            if len(hist) > 1:
                log_returns = np.diff(np.log(hist))
                if len(log_returns) > 0:
                    realized_vol = np.std(log_returns) / np.sqrt(self.dt)

        return np.array([S, W, sqrt_v, realized_vol], dtype=np.float32)

    def get_total_reward(self):
        return float(np.sum(self.daily_portfolio_returns_episode)) if self.daily_portfolio_returns_episode else 0.0
    

    def get_5days_return(self):
        if self.future_steps <= 0:
            return 0.0

        start_W = self.W_future[0]
        stop_idx = min(self.current_step_in_episode, self.future_steps)
        stop_W   = self.W_future[stop_idx]

        if start_W <= self.SMALL_POSITIVE_EPSILON:
            return 0.0
        return (stop_W - start_W) / start_W
