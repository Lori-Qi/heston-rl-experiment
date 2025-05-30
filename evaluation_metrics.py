import numpy as np
from simulators import WGANPriceSimulator

SMALL_POSITIVE_EPSILON = 1e-9


def safe_format(value, fmt, default_str="N/A"):
    try:
        if value is None or np.isnan(value) or np.isinf(value):
            return default_str
        return f"{value:{fmt}}"
    except Exception:
        return default_str


def calculate_sharpe_ratio(daily_returns, risk_free_rate_daily, annualization_factor=1.0):
    daily_returns = np.asarray(daily_returns, dtype=float)
    if daily_returns.size < 3:
        return np.nan
    excess = daily_returns - risk_free_rate_daily
    std_er = np.std(excess, ddof=1)
    if std_er < SMALL_POSITIVE_EPSILON:
        return np.nan
    return np.mean(excess) / std_er * np.sqrt(annualization_factor)


def max_drawdown(wealth_path):
    w = np.asarray(wealth_path, dtype=float)
    if w.size < 2:
        return np.nan
    peak = np.maximum.accumulate(w)
    draw = (w - peak) / np.maximum(peak, SMALL_POSITIVE_EPSILON)
    return float(np.min(draw))


def evaluate_policy_in_world(env, agent, evaluation_simulator=None, num_episodes=50):
    sim_name = type(evaluation_simulator).__name__ if evaluation_simulator else "EnvInternalDynamics"
    print(f"\n--- Evaluating Policy: Env={type(env).__name__}, EvalSim={sim_name} ---")

    ready = True
    if hasattr(evaluation_simulator, "is_fitted") and not evaluation_simulator.is_fitted:
        ready = False
    elif hasattr(evaluation_simulator, "is_loaded") and not evaluation_simulator.is_loaded:
        ready = False
    elif isinstance(evaluation_simulator, WGANPriceSimulator):
        try:
            _ = evaluation_simulator.generate_log_return()
        except Exception:
            ready = False
    if evaluation_simulator and not ready:
        print("   Evaluation ABORTED: simulator not ready.")
        return {
            k: np.nan for k in [
                "avg_ep_cum_return", "avg_ep_sharpe_5d",
                "avg_ep_maxdd", "avg_ep_ceq",
                "episodes_completed", "run_aborted"
            ]
        }

    agent.epsilon, eps_bak = 0.0, agent.epsilon

    R5_all, SR5_all, MD5_all = [], [], []
    episodes_done, aborted = 0, False

    for ep in range(1, num_episodes + 1):
        try:
            if hasattr(evaluation_simulator, "reset"):
                evaluation_simulator.reset()
            state = env.reset()
            wealth_path = [env.W_future[0]]
        except Exception as e:
            print(f"Reset error ep {ep}: {e}")
            continue

        done = False
        while not done:
            act, _ = agent.select_action(state)
            try:
                ns, _, done, info = env.step(act, simulator=evaluation_simulator)
                wealth_path.append(info["wealth"])
            except RuntimeError as err:
                print(f"RuntimeError ep {ep}: {err}")
                aborted = True
                break
            state = ns
            if env.current_step_in_episode >= env.future_steps:
                done = True
        if aborted:
            break

        daily_rets = env.raw_daily_returns_episode.copy()
        R_5d = env.get_5days_return()
        rf_d = (1 + env.r)**(1/252) - 1
        SR_5d = calculate_sharpe_ratio(daily_rets, rf_d)
        MD_5d = max_drawdown(wealth_path)

        R5_all.append(R_5d)
        SR5_all.append(SR_5d)
        MD5_all.append(MD_5d)
        episodes_done += 1

        if ep % max(1, num_episodes // 5) == 0:
            print(f"  Ep {ep}/{num_episodes}: "
                  f"R5 {safe_format(R_5d, '.3%')}, "
                  f"SR5 {safe_format(SR_5d, '.3f')}, "
                  f"MD5 {safe_format(MD_5d, '.2%')}")

    agent.epsilon = eps_bak

    res = {
        "avg_ep_cum_return": np.nanmean(R5_all) if episodes_done else np.nan,
        "avg_ep_sharpe_5d": np.nanmean(SR5_all) if episodes_done else np.nan,
        "avg_ep_maxdd": np.nanmean(MD5_all) if episodes_done else np.nan,
        "episodes_completed": episodes_done,
        "run_aborted": aborted,
    }

    if aborted:
        print("--- Eval aborted due to simulator failure ---")
    else:
        print("--- Eval finished ---")
        print(f"    Avg R5  : {safe_format(res['avg_ep_cum_return'], '.3%')}")
        print(f"    Avg SR5 : {safe_format(res['avg_ep_sharpe_5d'], '.3f')}")
        print(f"    Avg MD5 : {safe_format(res['avg_ep_maxdd'], '.2%')}")

    return res