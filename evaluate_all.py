"""
evaluate_all.py
───────────────
Benchmark all policies in the policies/ directory against each other.
Handles different data-format expectations per policy gracefully.
"""

import os
import sys
import importlib.util
import traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import time

from simulator import FPLEnv, PlayerSimulator


# ── Data Loading ─────────────────────────────────────────────────────────
DATA_URL = ("https://raw.githubusercontent.com/vaastav/"
            "Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv")


def load_raw_data():
    """Download the raw 2023-24 merged_gw CSV."""
    print("Loading 2023-24 FPL data from GitHub ...")
    return pd.read_csv(DATA_URL)


def make_sim_df(raw_df):
    """Pre-process raw data for the FPLEnv simulator."""
    df = raw_df.rename(columns={"element": "player_id", "GW": "gameweek"}).copy()
    df["price"] = df["value"] / 10.0
    df["form"] = (
        df.groupby("player_id")["total_points"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["minutes_form"] = (
        df.groupby("player_id")["minutes"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    return df


def make_baseline_df(raw_df):
    """Pre-process raw data for baseline.py which expects 'price', 'form',
    'minutes_form', and 'points' columns."""
    df = raw_df.copy()
    df["price"] = df["value"] / 10.0
    df["form"] = (
        df.groupby("element")["total_points"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["minutes_form"] = (
        df.groupby("element")["minutes"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["points"] = df["total_points"]
    return df


def build_env_model(df, feature_cols):
    X = df[feature_cols].fillna(0.0).values
    y = df["total_points"].values.astype(float)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model


# ── Rollout Runner ───────────────────────────────────────────────────────
def run_rollout(env, policy):
    state = env.reset()
    policy.reset()
    total_reward = 0.0
    done = False
    while not done:
        try:
            transfers = policy.act(state)
            action = transfers[0] if (transfers and len(transfers) > 0) else None
        except Exception:
            action = None
        state, reward, done = env.step(action)
        total_reward += reward
    return total_reward


# ── Per-policy Evaluator ─────────────────────────────────────────────────
def evaluate_policy(name, module_path, raw_df, baseline_df, sim_df,
                    sim_model, n_rollouts=10):
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {name}")
    print(f"{'─'*60}")

    # 1. Import the module
    spec = importlib.util.spec_from_file_location(f"policy_{name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        PolicyClass = getattr(mod, "StudentPolicy")
        policy = PolicyClass()
    except Exception as e:
        print(f"  ✗ IMPORT FAILED: {e}")
        return None

    # 2. Fit the policy — try multiple data formats
    start_time = time.time()
    fitted = False

    # Strategy A: no-arg fit (vedh.py downloads its own data)
    if not fitted:
        try:
            import inspect
            sig = inspect.signature(PolicyClass.fit)
            params = list(sig.parameters.keys())
            # If fit() accepts 0 non-self args or has all defaults
            if len(params) == 1 or all(
                sig.parameters[p].default is not inspect.Parameter.empty
                for p in params[1:]
            ):
                policy.fit()
                fitted = True
                print(f"  ✓ Fitted with no-arg fit()")
        except Exception:
            pass

    # Strategy B: pass raw data (tejas.py does its own feature engineering)
    if not fitted:
        try:
            policy_fresh = PolicyClass()
            policy_fresh.fit(raw_df)
            policy = policy_fresh
            fitted = True
            print(f"  ✓ Fitted with raw data")
        except Exception:
            pass

    # Strategy C: pass pre-processed data (baseline.py)
    if not fitted:
        try:
            policy_fresh = PolicyClass()
            policy_fresh.fit(baseline_df)
            policy = policy_fresh
            fitted = True
            print(f"  ✓ Fitted with pre-processed data")
        except Exception:
            pass

    # Strategy D: pass sim_df (has player_id, gameweek, price, form, etc.)
    if not fitted:
        try:
            policy_fresh = PolicyClass()
            policy_fresh.fit(sim_df)
            policy = policy_fresh
            fitted = True
            print(f"  ✓ Fitted with simulator data")
        except Exception as e:
            print(f"  ✗ FIT FAILED (all strategies): {e}")
            traceback.print_exc()
            return None

    fit_time = time.time() - start_time
    print(f"  Fit time: {fit_time:.1f}s")

    # 3. Run rollouts
    env = FPLEnv(data=sim_df, model=sim_model, squad_size=5)
    rewards = []
    errors = 0
    for i in range(n_rollouts):
        np.random.seed(i)
        try:
            r = run_rollout(env, policy)
            rewards.append(r)
        except Exception as e:
            errors += 1
            if errors <= 2:
                print(f"  ⚠ Rollout {i} error: {e}")

    if not rewards:
        print(f"  ✗ All rollouts failed")
        return None

    avg = np.mean(rewards)
    std = np.std(rewards)
    print(f"  Results: avg={avg:.1f}  std={std:.1f}  "
          f"({len(rewards)}/{n_rollouts} rollouts succeeded)")

    return {
        "Policy": name,
        "Avg Reward": round(avg, 2),
        "Std": round(std, 2),
        "Min": round(min(rewards), 2),
        "Max": round(max(rewards), 2),
        "Fit Time (s)": round(fit_time, 1),
        "Rollouts OK": f"{len(rewards)}/{n_rollouts}",
    }


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    raw_df = load_raw_data()
    sim_df = make_sim_df(raw_df)
    baseline_df = make_baseline_df(raw_df)

    feature_cols = ["price", "form", "minutes_form"]
    sim_model = build_env_model(sim_df, feature_cols)

    # Discover policies
    policy_dir = "policies"
    policy_files = sorted([
        f for f in os.listdir(policy_dir)
        if f.endswith(".py") and not f.startswith("__")
    ])

    print(f"\nFound {len(policy_files)} policies: "
          f"{[f.replace('.py','') for f in policy_files]}\n")

    results = []

    for f in policy_files:
        name = f.replace(".py", "")
        path = os.path.join(policy_dir, f)
        res = evaluate_policy(name, path, raw_df, baseline_df, sim_df,
                              sim_model, n_rollouts=10)
        if res:
            results.append(res)

    # No-transfer baseline
    print(f"\n{'─'*60}")
    print(f"  Evaluating: NO-TRANSFER (baseline)")
    print(f"{'─'*60}")
    bl_rewards = []
    for i in range(10):
        np.random.seed(i)
        r = FPLEnv.run_no_transfer(FPLEnv(data=sim_df, model=sim_model, squad_size=5))
        bl_rewards.append(r)
    results.append({
        "Policy": "NO-TRANSFER",
        "Avg Reward": round(np.mean(bl_rewards), 2),
        "Std": round(np.std(bl_rewards), 2),
        "Min": round(min(bl_rewards), 2),
        "Max": round(max(bl_rewards), 2),
        "Fit Time (s)": 0.0,
        "Rollouts OK": "10/10",
    })

    # ── Final Leaderboard ────────────────────────────────────────────
    res_df = pd.DataFrame(results).sort_values("Avg Reward", ascending=False)
    res_df.index = range(1, len(res_df) + 1)
    res_df.index.name = "Rank"

    print("\n")
    print("╔" + "═"*62 + "╗")
    print("║" + "       🏆  FPL POLICY LEADERBOARD  🏆       ".center(62) + "║")
    print("╠" + "═"*62 + "╣")
    print("║" + f"  10 rollouts per policy  •  squad_size=5".ljust(62) + "║")
    print("╚" + "═"*62 + "╝")
    print()
    print(res_df.to_string())
    print()

    # Highlight winner
    winner = res_df.iloc[0]
    print(f"🥇 Winner: {winner['Policy']}  "
          f"(avg reward = {winner['Avg Reward']})")
    print()


if __name__ == "__main__":
    main()
