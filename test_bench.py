"""
test_bench.py
─────────────
Evaluation harness for StudentPolicy using the FPLEnv / PlayerSimulator
classes from simulator.py.

Runs 50 rollouts and prints the average total reward V(π).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from simulator import FPLEnv, PlayerSimulator
from policies.policy import StudentPolicy


def build_env_model(df, feature_cols):
    """
    Train a simple Ridge model on the training data so that the simulator's
    PlayerSimulator can produce synthetic points.
    """
    X = df[feature_cols].fillna(0.0).values
    y = df["total_points"].values.astype(float)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model


def run_rollout(env, policy):
    """Run a single episode with the policy, return total reward."""
    state = env.reset()
    policy.reset()
    total_reward = 0.0
    done = False
    while not done:
        transfers = policy.act(state)
        action = transfers[0] if transfers else None
        state, reward, done = env.step(action)
        total_reward += reward
    return total_reward


def main():
    # ── 1. Load data ──────────────────────────────────────────────────
    url = ("https://raw.githubusercontent.com/vaastav/"
           "Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv")
    print("[test_bench] Loading data ...")
    df = pd.read_csv(url)

    # Rename for simulator compatibility
    df = df.rename(columns={"element": "player_id", "GW": "gameweek"})
    df["price"] = df["value"] / 10.0
    df["form"] = (
        df.groupby("player_id")["total_points"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["minutes_form"] = (
        df.groupby("player_id")["minutes"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # ── 2. Fit reward model for the simulator ─────────────────────────
    feature_cols = ["price", "form", "minutes_form"]
    sim_model = build_env_model(df, feature_cols)

    # ── 3. Fit the StudentPolicy ──────────────────────────────────────
    print("[test_bench] Fitting StudentPolicy ...")
    policy = StudentPolicy()
    policy.fit()  # downloads its own data internally

    # ── 4. Build the evaluation environment ───────────────────────────
    # We use a smaller squad_size=5 to match the default FPLEnv from simulator.py
    env = FPLEnv(data=df, model=sim_model, squad_size=5)

    # ── 5. Run rollouts ──────────────────────────────────────────────
    n_rollouts = 5
    rewards = []
    print(f"[test_bench] Running {n_rollouts} rollouts ...")

    for i in range(n_rollouts):
        np.random.seed(i)  # reproducibility per rollout
        r = run_rollout(env, policy)
        rewards.append(r)
        if (i + 1) % 10 == 0:
            print(f"  rollout {i+1}/{n_rollouts}:  reward = {r:.1f}")

    # ── 6. Report ────────────────────────────────────────────────────
    avg = np.mean(rewards)
    std = np.std(rewards)
    print("\n" + "=" * 55)
    print(f"  Average Total Reward  V(π) = {avg:.2f}  (± {std:.2f})")
    print(f"  Min = {min(rewards):.2f}   Max = {max(rewards):.2f}")
    print("=" * 55)

    # Compare with a no-transfer baseline
    print("\n[test_bench] No-transfer baseline (50 rollouts) ...")
    baseline_rewards = []
    for i in range(n_rollouts):
        np.random.seed(i)
        r = FPLEnv.run_no_transfer(FPLEnv(data=df, model=sim_model, squad_size=5))
        baseline_rewards.append(r)
    baseline_avg = np.mean(baseline_rewards)
    baseline_std = np.std(baseline_rewards)
    print(f"  Baseline V(no-transfer) = {baseline_avg:.2f}  (± {baseline_std:.2f})")
    print(f"  Improvement = {avg - baseline_avg:+.2f} points\n")


if __name__ == "__main__":
    main()
