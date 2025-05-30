import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
import os 
from pathlib import Path 

from dqn_agent.agent import DQNAgent
from environment import Heston_Env
from simulators import EnvHestonSimulator, EstHestonSimulator, EstGBMSimulator, WGANPriceSimulator, ReplaySimulator, EstFailReason
from evaluation_metrics import evaluate_policy_in_world
from models_and_trainer import train_wgan_heston 

class TrainingManager:
    def __init__(
        self,
        env,
        agent,
        training_simulator=None,
        evaluation_simulator=None,
        max_episodes=1000,
        log_freq=10,
        eval_freq=2000,
        eval_episodes=1,
        save_freq=100,
        model_weights_path='dqn_agent_weights',
    ):
        self.env = env
        self.agent = agent
        self.training_simulator = training_simulator
        self.evaluation_simulator = evaluation_simulator
        self.max_episodes = max_episodes
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_freq = save_freq
        self.model_weights_path = model_weights_path

        self.episode_rewards = []
        self.avg_rewards_100 = []
        self.evaluation_scores = []
        self.evaluation_episodes = []
        self.training_start_time = None
        self.current_training_run_aborted_flag = False

    def train(self):
        sim_name = type(self.training_simulator).__name__ if self.training_simulator else 'EnvInternalDynamics'
        print(f"\n[INFO] Training started | Env: {type(self.env).__name__}, Simulator: {sim_name}")

        ready = True
        if hasattr(self.training_simulator, 'is_fitted') and not self.training_simulator.is_fitted:
            ready = False
        elif hasattr(self.training_simulator, 'is_loaded') and not self.training_simulator.is_loaded:
            ready = False
        if not ready:
            warnings.warn(f"[WARNING] Simulator {sim_name} is not ready. Training may fail.", RuntimeWarning)

        self.training_start_time = time.time()
        total_steps = 0
        self.current_training_run_aborted_flag = False

        for ep in range(1, self.max_episodes + 1):
            try:
                state = self.env.reset()
            except Exception as e:
                print(f"[ERROR] env.reset() failed at episode {ep}: {e}")
                warnings.warn("Training aborted due to reset failure.")
                self.current_training_run_aborted_flag = True
                break

            ep_reward, done, steps = 0, False, 0

            while not done:
                action, idx = self.agent.select_action(state)
                try:
                    next_state, reward, done, info = self.env.step(action, simulator=self.training_simulator)
                except RuntimeError as e_step:
                    print(f"[ERROR] env.step() failed at episode {ep}, step {steps+1}: {e_step}")
                    warnings.warn("Training aborted due to simulator failure.")
                    self.current_training_run_aborted_flag = True
                    done = True
                    break

                self.agent.step(state, idx, reward, next_state, done)
                state = next_state
                ep_reward += reward
                total_steps += 1
                steps += 1

            if self.current_training_run_aborted_flag:
                break

            self.episode_rewards.append(ep_reward)
            avg_r = sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:])
            self.avg_rewards_100.append(avg_r)
            duration = time.time() - self.training_start_time

            if ep % self.log_freq == 0:
                print(f"[EP {ep:4d}] Reward: {ep_reward:.4f} | Avg(100): {avg_r:.4f} | Epsilon: {self.agent.epsilon:.4f} | Total Steps: {total_steps} | Time: {duration:.1f}s")

            if ep % self.eval_freq == 0:
                eval_name = type(self.evaluation_simulator).__name__ if self.evaluation_simulator else 'EnvInternalDynamics'
                ready_eval = True
                if hasattr(self.evaluation_simulator, 'is_fitted') and not self.evaluation_simulator.is_fitted:
                    ready_eval = False
                elif hasattr(self.evaluation_simulator, 'is_loaded') and not self.evaluation_simulator.is_loaded:
                    ready_eval = False
                if not ready_eval:
                    warnings.warn(f"[WARNING] Eval simulator {eval_name} is not ready. Skipping evaluation.")
                else:
                    score = self.evaluate(num_episodes=self.eval_episodes)
                    if score != score:
                        print(f"[ERROR] Evaluation failed at episode {ep}. Training aborted.")
                        self.current_training_run_aborted_flag = True
                        break
                    self.evaluation_scores.append(score)
                    self.evaluation_episodes.append(ep)
                    print(f"[EVAL @ Ep {ep}] Score (sim: {eval_name}): {score:.4f}")

            if self.current_training_run_aborted_flag:
                break

            if ep % self.save_freq == 0:
                self.agent.save_weights(f"{self.model_weights_path}_ep{ep}")

        total_time = time.time() - self.training_start_time
        if self.current_training_run_aborted_flag:
            print(f"\n[ABORTED] Training stopped at episode {ep}. Duration: {total_time:.2f}s")
        else:
            print(f"\n[INFO] Training complete. Total time: {total_time:.2f}s")
            self.agent.save_weights(f"{self.model_weights_path}_final")
            self.visualize_training()

        return self.agent

    def evaluate(self, num_episodes=1):
        orig_eps = self.agent.epsilon
        self.agent.epsilon = 0
        rewards = []
        aborted = False

        for i in range(1, num_episodes + 1):
            try:
                if hasattr(self.evaluation_simulator, "reset"):
                    self.evaluation_simulator.reset()
                state = self.env.reset()
            except Exception as e:
                print(f"[ERROR] eval env.reset() failed at episode {i}: {e}")
                continue

            ep_r, done, steps = 0, False, 0

            while not done:
                action, _ = self.agent.select_action(state)
                try:
                    ns, r, done, info = self.env.step(action, simulator=self.evaluation_simulator)
                except RuntimeError as e_eval:
                    print(f"[ERROR] Evaluation failed at episode {i}, step {steps+1}: {e_eval}")
                    warnings.warn("Evaluation aborted.")
                    aborted = True
                    break
                state = ns
                ep_r += r
                steps += 1
                if steps >= self.env.future_steps:
                    done = True

            if aborted:
                break

            rewards.append(ep_r)

        self.agent.epsilon = orig_eps
        if aborted:
            return float('nan')
        return sum(rewards) / len(rewards) if rewards else 0

    def visualize_training(self):
        metrics = self.agent.get_metric()
        plot_configs = []

        if self.episode_rewards:
            plot_configs.append({
                "data": [self.episode_rewards, self.avg_rewards_100],
                "labels": ["Ep Reward", "Avg Reward (100eps)"],
                "title": "Episode Rewards",
                "xlabel": "Episode",
                "ylabel": "Reward",
                "colors": ["#1f77b4", "#d62728"],
            })

        if metrics.get("epsilon_history"):
            plot_configs.append({
                "data": [metrics["epsilon_history"]],
                "labels": ["Epsilon"],
                "title": "Epsilon Decay",
                "xlabel": "Agent Step",
                "ylabel": "Epsilon",
                "colors": ["#2ca02c"],
            })

        if metrics.get("loss_history"):
            loss_hist = metrics["loss_history"]
            p_data = [loss_hist]
            p_labels = ["Loss"]
            p_colors = ["#9467bd"]
            if len(loss_hist) > 50:
                smoothed = np.convolve(loss_hist, np.ones(50) / 50, mode="valid")
                p_data.append(smoothed)
                p_labels.append("Smoothed Loss (50)")
                p_colors.append("#ff7f0e")
            plot_configs.append({
                "data": p_data,
                "labels": p_labels,
                "colors": p_colors,
                "title": "Loss History (Log Scale)",
                "xlabel": "Learn Step",
                "ylabel": "Loss",
                "yscale": "log",
                "custom_plot_logic": True,
            })

        if self.evaluation_scores and self.evaluation_episodes:
            plot_configs.append({
                "data": [self.evaluation_scores],
                "labels": ["Eval Score"],
                "x_values": [self.evaluation_episodes],
                "title": "Evaluation Score",
                "xlabel": "Training Episode",
                "ylabel": "Avg Eval Score",
                "marker": "o",
                "colors": ["#17becf"],
            })

        if not plot_configs:
            print("[INFO] No data to visualize.")
            return

        fig, axs = plt.subplots(len(plot_configs), 1, figsize=(12, 4 * len(plot_configs)), squeeze=False)
        axs = axs.ravel()

        for idx, cfg in enumerate(plot_configs):
            ax = axs[idx]
            x_values_sets = cfg.get("x_values", [None] * len(cfg["data"]))
            if cfg.get("custom_plot_logic"):
                raw_y = cfg["data"][0]
                ax.plot(np.arange(len(raw_y)), raw_y, label=cfg["labels"][0], alpha=0.6, color=cfg["colors"][0])
                if len(cfg["data"]) > 1 and len(raw_y) > 50:
                    smooth_y = cfg["data"][1]
                    smooth_x = np.arange(len(smooth_y)) + (len(raw_y) - len(smooth_y)) // 2
                    ax.plot(smooth_x, smooth_y, label=cfg["labels"][1], color=cfg["colors"][1])
            else:
                for yd, lbl, clr, xd in zip(cfg["data"], cfg["labels"], cfg["colors"], x_values_sets):
                    if yd is None:
                        continue
                    x_vals = xd if xd is not None else np.arange(len(yd))
                    ax.plot(x_vals, yd, label=lbl, marker=cfg.get("marker", ""), linestyle=cfg.get("linestyle", "-"), alpha=0.7 if "Avg" not in lbl else 1.0, color=clr)

            ax.set_title(cfg["title"])
            ax.set_xlabel(cfg["xlabel"])
            ax.set_ylabel(cfg["ylabel"])
            if cfg.get("yscale"):
                ax.set_yscale(cfg["yscale"])
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()


def run_heston_rl_experiment(
        num_worlds=100,
        base_save_dir="heston_rl_experiment_output",
        true_env_S0=100.0,
        true_env_v0=0.04,
        true_env_mu=0.35,
        true_env_kappa=3.5,
        true_env_theta=0.07,
        true_env_xi=0.55,
        true_env_rho=-0.70,
        true_env_r_annual=0.025,
        hist_days_for_est=504,
        rl_action_space_config=np.linspace(-1, 1, 41),
        rl_max_train_episodes=1500,
        rl_eval_episodes_final=1,
        rl_env_episode_future_steps=5,
        rl_env_num_steps_lookback=126,
        global_wgan_dir="global_wgan",
        wgan_num_paths=10000,
        wgan_training_itts=2000,
):
    print(f"[INFO] Running Heston RL experiment: {num_worlds} worlds")
    os.makedirs(base_save_dir, exist_ok=True)

    if not Path(global_wgan_dir, "generator.h5").exists():
        print("[INFO] Training global WGAN...")
        train_wgan_heston(
            S0=true_env_S0, v0=true_env_v0, r=true_env_r_annual,
            kappa=true_env_kappa, theta=true_env_theta,
            xi=true_env_xi, rho=true_env_rho,
            T=1.0, dt=1/252, num_paths=wgan_num_paths,
            save_dir=global_wgan_dir, n_itts=wgan_training_itts)
        print(f"[INFO] Global WGAN saved at {global_wgan_dir}")
    else:
        print(f"[INFO] Global WGAN found: {global_wgan_dir}")

    global_wgan_sim = WGANPriceSimulator(global_wgan_dir, z_dim=32)
    dt = 1 / 252
    true_params = dict(mu=true_env_mu, kappa=true_env_kappa,
                       theta=true_env_theta, xi=true_env_xi,
                       rho=true_env_rho)

    warnings.filterwarnings("ignore", message=".*ReplaySimulator missing _v_for_step.*")
    results_all = []

    for w in range(num_worlds):
        print(f"[WORLD {w+1}/{num_worlds}]")
        row = {"world_id": w}
        save_dir = Path(base_save_dir, f"world_{w:03d}")
        save_dir.mkdir(parents=True, exist_ok=True)

        hist_sim = EnvHestonSimulator(**true_params, dt=dt)
        S_hist, v_hist = [true_env_S0], [true_env_v0]
        S, v = true_env_S0, true_env_v0
        for _ in range(hist_days_for_est):
            S, v = hist_sim._internal_simulate_heston_step(S, v)
            S_hist.append(S)
            v_hist.append(v)
        S_hist, v_hist = np.asarray(S_hist), np.asarray(v_hist)

        world_wgan_sim = WGANPriceSimulator(global_wgan_dir, z_dim=32)
        world_wgan_sim.recenter_to_historical_data(S_hist, v_hist)

        env = Heston_Env(
            S0=S_hist[0], W0=1000., v0=v_hist[0], mu=true_env_mu,
            kappa=true_env_kappa, theta=true_env_theta, xi=true_env_xi,
            rho=true_env_rho, r=true_env_r_annual, T=1.0,
            num_steps=rl_env_num_steps_lookback,
            future_steps=rl_env_episode_future_steps,
            dt=dt, gamma=2.0)
        env.S_hist[:] = S_hist[-(rl_env_num_steps_lookback+1):]
        env.v_hist[:] = v_hist[-(rl_env_num_steps_lookback+1):]

        future_prices = []
        hist_sim.set_initial_variance(v_hist[-1])
        S_next, v_next = S_hist[-1], v_hist[-1]
        for _ in range(rl_env_episode_future_steps):
            S_next, v_next = hist_sim._internal_simulate_heston_step(S_next, v_next)
            future_prices.append(S_next)
        replay_sim = ReplaySimulator(future_prices)

        est_heston = EstHestonSimulator(dt=dt)
        fit_ok_h, _ = est_heston.update_parameters(S_hist, v_hist)
        est_gbm = EstGBMSimulator(dt=dt)
        fit_ok_g = est_gbm.update_parameters(S_hist)

        sims = {
            "EnvTrue": hist_sim,
            "EstHeston": est_heston if fit_ok_h else None,
            "EstGBM": est_gbm if fit_ok_g else None,
            "WGAN": world_wgan_sim,
        }

        for tag, sim in sims.items():
            pref = f"{tag}_"
            if sim is None:
                row.update({pref + k: np.nan for k in ["R5", "SR5", "MD5"]})
                continue

            replay_sim.reset()
            agent = DQNAgent(state_size=4, action_space=rl_action_space_config)
            tm = TrainingManager(
                env=env, agent=agent,
                training_simulator=sim,
                evaluation_simulator=replay_sim,
                max_episodes=rl_max_train_episodes,
                eval_episodes=rl_eval_episodes_final,
                log_freq=200,
                model_weights_path=save_dir / f"agent_{tag}")
            tm.train()

            if tm.current_training_run_aborted_flag:
                row.update({pref + k: np.nan for k in ["R5", "SR5", "MD5"]})
            else:
                replay_sim.reset()
                metrics = evaluate_policy_in_world(env, agent, replay_sim, num_episodes=1)
                row.update({
                    pref + "R5": metrics.get("avg_ep_cum_return", np.nan),
                    pref + "SR5": metrics.get("avg_ep_sharpe_5d", np.nan),
                    pref + "MD5": metrics.get("avg_ep_maxdd", np.nan)
                })

        results_all.append(row)

    df = pd.DataFrame(results_all)
    print("\n[INFO] Experiment completed. Results:")
    print(df)
    csv_path = Path(base_save_dir, "aggregated_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Results saved to {csv_path}")
    return df
