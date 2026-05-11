# Fantasy Premier League — Reinforcement Learning Final Project

DAEN/ISEN 489 Final Project, Texas A&M University.

This project applies reinforcement learning to the Fantasy Premier League (FPL) squad management problem. Three independent agent implementations are benchmarked against each other and a no-transfer baseline using a shared simulation environment built on historical 2023-24 FPL season data.

---

## Overview

Each gameweek, an FPL manager selects a squad and may transfer one player in exchange for another. The goal is to maximize cumulative predicted points over a full season. This project frames that decision as a sequential decision-making problem and explores several approaches — from greedy heuristics to full offline Implicit Q-Learning (IQL).

The environment simulates a simplified version of FPL:
- A 5-player squad (simplified from the real 15)
- A Ridge regression model trained on historical features predicts points per player per gameweek
- Gaussian noise is added to rewards to model real-world variance
- One transfer per gameweek is allowed; actions are `(player_out_id, player_in_id)` pairs

---

## Repository Structure

```
.
|- simulator.py         # Core FPL environment (FPLEnv, PlayerSimulator)
|- evaluate_all.py      # Benchmarks all policies and prints a leaderboard
|- test_bench.py        # Single-policy evaluation harness (defaults to vedh.py)
|- policies/
|   |- baseline.py      # Dummy baseline (predicts mean points, makes no transfers)
|   |- vedh.py          # IQL policy with Ridge reward model and tabular value function
|   |- tejas.py         # IQL policy with HistGBM reward model and PyTorch V/Q networks
|   |- michael.py       # Greedy NN policy with lineup and captaincy selection
|- report.pdf           # Written project report
```

---

## Environment

**`simulator.py`**

`PlayerSimulator` wraps a scikit-learn model and adds Gaussian noise to point predictions.

`FPLEnv` manages the episode loop:
- `reset()` — randomly selects an initial squad from available players
- `step(action)` — applies a `(player_out, player_in)` transfer (or `None`), computes the squad's predicted reward, and advances the gameweek
- `run_no_transfer(env)` — static method that runs a full season with no transfers

State: `(gameweek: int, squad: tuple[player_id, ...])`

The environment fetches pre-processed data containing `player_id`, `gameweek`, `price`, `form` (rolling 3-GW mean of total points), and `minutes_form` (rolling 3-GW mean of minutes played).

---

## Policies

All policies implement the same interface:

| Method | Description |
|---|---|
| `fit(train_data)` | Train any internal models on historical data |
| `reset()` | Reset per-episode state before each rollout |
| `act(state)` | Return a list of transfers `[(player_out_id, player_in_id)]` or `[]` |

### `baseline.py`

A do-nothing reference implementation. `fit()` records the mean points across all players and gameweeks. `act()` always returns an empty list (no transfers). Intended as a scaffold, not a competitive policy.

### `vedh.py` — IQL with Ridge Reward Model

**Reward model:** Ridge regression trained on 5 engineered features: `price`, `form`, `minutes_form`, `cum_points` (cumulative total points up to the current gameweek), and `opponent_strength` (historical average goals conceded by the opponent team).

**Value function:** A tabular value function `V(s)` is learned via expectile regression on offline trajectories constructed from the training data. States are coarsely bucketed into three season phases (early, mid, late). The expectile parameter tau = 0.7 gives an optimistic estimate of value.

**Transfer selection:** At inference, the policy evaluates all legal single-swap transfers by computing the marginal gain `r_in - r_out - penalty` and picks the best one exceeding zero. A same-position constraint is enforced, along with a 3-players-per-club limit and a budget check. Free transfers are tracked across gameweeks.

### `tejas.py` — Full IQL with PyTorch Networks

**Reward model:** `HistGradientBoostingRegressor` trained on 14 features per player per gameweek, including multi-window rolling averages (3 GW and 5 GW), goals, assists, clean sheets, bonus points, BPS, starts percentage, expected points (`xP`), home/away flag, normalized opponent difficulty, and positional encoding. 5-fold cross-validation is used to report out-of-fold MAE before training a final model on all data.

**Offline data collection:** 150 epsilon-greedy (epsilon = 0.30) rollouts are run against an internal FPL simulation environment. Each transition stores a 212-dimensional state vector (gameweek, budget, and 15 player feature vectors ordered by position and form) and an 11-dimensional action vector encoding the player-out and player-in features.

**IQL training:** A V-network and a Q-network (each a 3-layer MLP with LayerNorm) are trained for 25,000 gradient steps on a replay buffer of the collected transitions. The V-network is updated with asymmetric expectile loss (tau = 0.7); the Q-network is updated with standard TD error against the V-network's next-state estimate. A Polyak-averaged target Q-network stabilizes training.

**Transfer selection:** At inference, the IQL Q-network scores all valid transfers (constrained by position, budget, and club limit) and the no-transfer action. A transfer is executed only if its Q-score exceeds the no-transfer baseline by more than 0.01.

### `michael.py` — Greedy Neural Network Policy

**Reward model:** A 2-hidden-layer neural network (64 -> 32 -> 1) with ReLU activations and 10% dropout, trained with Adam and early stopping. The feature set (15 features) includes short-term and long-term form windows, goals, assists, clean sheets, BPS, ICT index, expected goals and assists, opponent strength, home flag, positional encoding, and log-scaled selection count. A temporal train/validation split (last 6 gameweeks held out) is used for early stopping.

**Transfer selection:** The policy iterates over every squad player and every eligible replacement of the same position. A transfer is made only if the predicted point improvement exceeds a threshold (default: 0.5 points). The top-20 candidates per position are considered.

**Lineup and captaincy:** An internal `_select_lineup` method selects the best starting 11 from the squad subject to minimum formation requirements (1 GK, 3 DEF, 2 MID, 1 FWD). The player with the highest predicted score is named captain and earns double points.

---

## Data

All policies use the publicly available FPL dataset maintained by [vaastav](https://github.com/vaastav/Fantasy-Premier-League). The 2023-24 merged gameweek CSV is fetched directly from GitHub at runtime:

```
https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv
```

No local data files are required.

---

## Running the Benchmark

Install dependencies first:

```bash
pip install numpy pandas scikit-learn torch
```

Then run the full benchmark across all policies:

```bash
python evaluate_all.py
```

This will:
1. Download the 2023-24 FPL data
2. Fit each policy using whatever data format it expects (the evaluator tries multiple strategies gracefully)
3. Run 10 rollouts per policy using the shared `FPLEnv`
4. Print a ranked leaderboard with average reward, standard deviation, min, max, fit time, and rollout success rate
5. Compute the no-transfer baseline for comparison

To run a quick evaluation of the `vedh` policy alone:

```bash
python test_bench.py
```

---

## Key Design Decisions

**Shared environment, independent policies.** Each policy does its own feature engineering and model training internally. The evaluator uses a common `FPLEnv` with a Ridge-based simulator model for fair comparison across all policies.

**Data format compatibility.** `evaluate_all.py` tries four fitting strategies in order: no-arg fit (for policies that self-download data), raw data, pre-processed baseline data, and simulator-formatted data. This allows policies with different internal pipelines to coexist.

**Simplified squad size.** The `FPLEnv` uses a 5-player squad rather than the real FPL 15, which reduces the combinatorial search space for the evaluation harness while preserving the transfer decision problem structure.

**Offline RL.** Both `vedh.py` and `tejas.py` use the IQL framework (Kostrikov et al., ICLR 2022), which learns from fixed offline data without requiring an interactive environment during training. This makes training stable and avoids policy-induced distribution shift.

---

## Team

- Vedh Jaishankar (`vedh.py`)
- Tejas (`tejas.py`)
- Michael (`michael.py`)
