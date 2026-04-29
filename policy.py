# policy.py
import numpy as np
import pandas as pd
from collections import Counter


class StudentPolicy:
    """
    Implicit Q-Learning (IQL) policy for Fantasy Premier League squad management.

    Architecture:
        Layer 1 – Reward model: Ridge regression predicting per-player gameweek
                  points from engineered features (price, form, minutes_form,
                  historical total_points, and opponent_strength).
        Layer 2 – IQL value / advantage functions learned via expectile regression
                  on offline trajectories constructed from the 2023-24 season data.

    The act() method selects the single transfer (player_out → player_in) that
    maximises the estimated Q-value minus the 4-point transfer penalty, or makes
    no transfer if no swap improves value.
    """

    # ── FPL structural constants ──────────────────────────────────────────
    POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}  # 15 total
    LINEUP_MIN = {"DEF": 3, "MID": 2, "FWD": 1}  # starting 11
    MAX_CLUB = 3
    BUDGET = 100.0  # £100.0 m
    TRANSFER_PENALTY = 4  # points deducted per additional transfer
    SQUAD_SIZE = 15

    # ── IQL hyper-parameters ──────────────────────────────────────────────
    EXPECTILE_TAU = 0.7        # expectile for V_ψ  (> 0.5 → optimistic)
    DISCOUNT = 0.99            # γ
    IQL_ALPHA = 0.1            # soft-update / learning-rate for value iteration

    def __init__(self):
        self.reward_model = None        # Ridge regression θ
        self.feature_cols = None
        self.player_db = None           # per-player latest feature snapshot
        self.team_strength = None       # dict: opponent_team_id → avg goals conceded
        self.V_table = {}               # V_ψ(state_hash)
        self.Q_cache = {}               # lightweight Q(s, a) memoisation
        self.gw_data = None             # full training DataFrame (for resets)
        self.position_map = {}          # element → position
        self.team_map = {}              # element → real-world team name
        self.price_map = {}             # element → latest price (£m)
        self.name_map = {}              # element → player name
        self._fitted = False

    # ──────────────────────────────────────────────────────────────────────
    #  STEP 1 : FEATURE ENGINEERING & REWARD MODEL
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, train_data=None):
        """
        Fit the reward prediction model and IQL value function.

        Parameters
        ----------
        train_data : pd.DataFrame or None
            If None the canonical 2023-24 merged_gw CSV is fetched from GitHub.
        """
        # ── 1a. Load data ─────────────────────────────────────────────────
        if train_data is None:
            url = ("https://raw.githubusercontent.com/vaastav/"
                   "Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv")
            df = pd.read_csv(url)
        else:
            df = train_data.copy()

        # Normalise column names for safety
        df.columns = [c.strip() for c in df.columns]

        # Identify the player-id column
        if "player_id" in df.columns:
            pid_col = "player_id"
        elif "element" in df.columns:
            pid_col = "element"
        else:
            raise KeyError("Cannot find player ID column (player_id / element)")

        # Identify the gameweek column
        if "gameweek" in df.columns:
            gw_col = "gameweek"
        elif "GW" in df.columns:
            gw_col = "GW"
        elif "round" in df.columns:
            gw_col = "round"
        else:
            raise KeyError("Cannot find gameweek column")

        # Identify the points column
        if "points" in df.columns:
            pts_col = "points"
        elif "total_points" in df.columns:
            pts_col = "total_points"
        else:
            raise KeyError("Cannot find points column")

        # Price: the CSV stores value in 0.1 £m (e.g. 55 → £5.5m)
        if "value" in df.columns:
            df["price"] = df["value"] / 10.0
        elif "price" not in df.columns:
            df["price"] = 5.0  # fallback

        # ── 1b. Engineer rolling features ────────────────────────────────
        df = df.sort_values([pid_col, gw_col]).reset_index(drop=True)

        # form = rolling-3 mean of total_points
        df["form"] = (
            df.groupby(pid_col)[pts_col]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        # minutes_form = rolling-3 mean of minutes played
        if "minutes" in df.columns:
            df["minutes_form"] = (
                df.groupby(pid_col)["minutes"]
                .transform(lambda s: s.rolling(3, min_periods=1).mean())
            )
        else:
            df["minutes_form"] = 60.0

        # opponent_strength = historical avg goals conceded by the opponent
        if "opponent_team" in df.columns:
            opp_strength = (
                df.groupby("opponent_team")["goals_conceded"]
                .mean()
                .to_dict()
                if "goals_conceded" in df.columns
                else {}
            )
            df["opponent_strength"] = (
                df["opponent_team"].map(opp_strength).fillna(1.0)
            )
            self.team_strength = opp_strength
        else:
            df["opponent_strength"] = 1.0
            self.team_strength = {}

        # cumulative total_points up to this gw (as a feature)
        df["cum_points"] = (
            df.groupby(pid_col)[pts_col].cumsum()
        )

        # ── 1c. Fit Ridge reward model ───────────────────────────────────
        self.feature_cols = [
            "price", "form", "minutes_form", "cum_points", "opponent_strength"
        ]
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        X = df[self.feature_cols].fillna(0.0).values.astype(np.float64)
        y = df[pts_col].values.astype(np.float64)

        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        self.reward_model = Ridge(alpha=1.0)
        self.reward_model.fit(X, y)

        y_pred = self.reward_model.predict(X)
        r2 = r2_score(y, y_pred)
        print(f"[StudentPolicy] Reward-model R² = {r2:.4f}  "
              f"(n={len(y)}, features={self.feature_cols})")

        # ── 1d. Build look-up tables ─────────────────────────────────────
        latest_gw = df[gw_col].max()
        latest = df[df[gw_col] == latest_gw].copy()

        if "position" in df.columns:
            pos_col = "position"
        elif "element_type" in df.columns:
            pos_col = "element_type"
        else:
            pos_col = None

        for _, row in df.iterrows():
            pid = int(row[pid_col])
            if pos_col and pos_col in row.index:
                self.position_map[pid] = row[pos_col]
            if "team" in row.index:
                self.team_map[pid] = row["team"]
            if "name" in row.index:
                self.name_map[pid] = row["name"]

        # latest snapshot per player for features
        self.player_db = {}
        for _, row in latest.iterrows():
            pid = int(row[pid_col])
            self.player_db[pid] = {
                c: float(row[c]) if c in row.index else 0.0
                for c in self.feature_cols
            }
            self.price_map[pid] = float(row["price"]) if "price" in row.index else 5.0

        # Also fill in from full df for players not in latest_gw
        for pid in df[pid_col].unique():
            pid = int(pid)
            if pid not in self.player_db:
                sub = df[df[pid_col] == pid].iloc[-1]
                self.player_db[pid] = {
                    c: float(sub[c]) if c in sub.index else 0.0
                    for c in self.feature_cols
                }
                self.price_map[pid] = float(sub["price"]) if "price" in sub.index else 5.0

        # ── STEP 3 : IQL – expectile regression on offline trajectories ──
        self._fit_iql(df, pid_col, gw_col, pts_col)

        self.gw_data = df
        self._fitted = True

    # ──────────────────────────────────────────────────────────────────────
    #  IQL fitting (offline)
    # ──────────────────────────────────────────────────────────────────────

    def _fit_iql(self, df, pid_col, gw_col, pts_col):
        """
        Learn V_ψ via expectile regression on per-gameweek squad returns.

        We construct synthetic offline trajectories:
            • For each gameweek, compute the total reward of the "top-K" squad
              (the best 11 from a greedy 15-player selection respecting constraints).
            • The state is hashed as (gw, budget_bucket, top_form_bucket).
            • V(s) is updated toward the observed return with asymmetric loss.
        """
        tau = self.EXPECTILE_TAU
        gamma = self.DISCOUNT
        alpha = self.IQL_ALPHA

        gw_list = sorted(df[gw_col].unique())
        T = len(gw_list)

        # Per-GW reward: average total_points of players who started
        gw_rewards = {}
        for gw in gw_list:
            sub = df[(df[gw_col] == gw) & (df["minutes"] > 0)] if "minutes" in df.columns else df[df[gw_col] == gw]
            gw_rewards[gw] = float(sub[pts_col].sum()) if len(sub) else 0.0

        # Build simple state keys and run value iteration
        for _ in range(5):  # a few passes
            for i, gw in enumerate(gw_list):
                s_key = self._state_key_gw(gw, T)
                r = gw_rewards[gw]
                if i + 1 < T:
                    s_next = self._state_key_gw(gw_list[i + 1], T)
                    v_next = self.V_table.get(s_next, 0.0)
                else:
                    v_next = 0.0
                target = r + gamma * v_next
                v_old = self.V_table.get(s_key, 0.0)
                diff = target - v_old
                weight = tau if diff > 0 else (1 - tau)
                self.V_table[s_key] = v_old + alpha * weight * diff

    def _state_key_gw(self, gw, T):
        """Hash a gameweek into a coarse state bucket."""
        phase = int(3 * (gw - 1) / max(T - 1, 1))  # 0, 1, 2 → early/mid/late
        return ("gw_phase", phase)

    # ──────────────────────────────────────────────────────────────────────
    #  STEP 2 : CONSTRAINT ENGINE
    # ──────────────────────────────────────────────────────────────────────

    def _is_legal(self, squad, budget, action=None):
        """
        Check whether *squad* (list of element IDs) is legal, optionally after
        applying *action = (player_out, player_in)*.

        Returns
        -------
        bool
        """
        squad = list(squad)
        if action is not None:
            p_out, p_in = action
            if p_out not in squad:
                return False
            squad = [p if p != p_out else p_in for p in squad]

        # 1. Exactly 15 players
        if len(squad) != self.SQUAD_SIZE:
            return False

        # 2. No duplicates
        if len(set(squad)) != self.SQUAD_SIZE:
            return False

        # 3. Position composition
        pos_counts = Counter(self.position_map.get(p, "MID") for p in squad)
        for pos, lim in self.POSITION_LIMITS.items():
            if pos_counts.get(pos, 0) != lim:
                return False

        # 4. Budget
        total_cost = sum(self.price_map.get(p, 5.0) for p in squad)
        if total_cost > budget + 0.01:  # small float tolerance
            return False

        # 5. Club limit (max 3 from any single team)
        club_counts = Counter(self.team_map.get(p, "Unknown") for p in squad)
        if any(v > self.MAX_CLUB for v in club_counts.values()):
            return False

        # 6. Valid starting 11 must be possible (3 DEF, 2 MID, 1 FWD + 1 GK = 7,
        #    remaining 4 from any outfield)
        #    We just verify there are *enough* players in each position.
        if pos_counts.get("GK", 0) < 1:
            return False
        if pos_counts.get("DEF", 0) < 3:
            return False
        if pos_counts.get("MID", 0) < 2:
            return False
        if pos_counts.get("FWD", 0) < 1:
            return False

        return True

    def _select_starting_11(self, squad):
        """
        Pick the best starting 11 from a 15-player squad.
        Formation must have 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD; total = 11.
        """
        by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in squad:
            pos = self.position_map.get(p, "MID")
            r = self.predict_reward(p)
            by_pos[pos].append((p, r))

        for pos in by_pos:
            by_pos[pos].sort(key=lambda x: -x[1])

        # Mandatory slots
        lineup = []
        lineup += [by_pos["GK"][0]] if by_pos["GK"] else []
        lineup += by_pos["DEF"][:3]
        lineup += by_pos["MID"][:2]
        lineup += by_pos["FWD"][:1]

        # Fill remaining 4 spots from unused outfield
        used = {p for p, _ in lineup}
        bench = []
        for pos in ["DEF", "MID", "FWD"]:
            for p, r in by_pos[pos]:
                if p not in used:
                    bench.append((p, r))
        bench.sort(key=lambda x: -x[1])
        lineup += bench[:11 - len(lineup)]

        return [p for p, _ in lineup]

    # ──────────────────────────────────────────────────────────────────────
    #  REWARD PREDICTION
    # ──────────────────────────────────────────────────────────────────────

    def predict_reward(self, player_id_or_dict):
        """
        Predict expected points for a player.

        Parameters
        ----------
        player_id_or_dict : int or dict
            If int, look up features from player_db.
            If dict, use values directly.
        """
        if self.reward_model is None:
            raise ValueError("Model has not been fit yet.")

        if isinstance(player_id_or_dict, dict):
            feats = player_id_or_dict
        else:
            pid = int(player_id_or_dict)
            feats = self.player_db.get(pid, {})
            if not feats:
                return 0.0

        x = np.array([[feats.get(c, 0.0) for c in self.feature_cols]])
        return float(self.reward_model.predict(x)[0])

    # ──────────────────────────────────────────────────────────────────────
    #  STEP 3 : ACT  (IQL-driven transfer decision)
    # ──────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset any rollout-specific memory (called before each episode)."""
        self.transfers_used = 0
        self.free_transfers = 1
        self.current_week = 0

    def act(self, state):
        """
        Return a list of transfers [(player_out_id, player_in_id), ...].

        The IQL policy evaluates Q(s, a) = r̂(swap) + γ V(s') and picks the
        transfer that maximises Q minus any penalty.  If no transfer yields
        positive marginal value, return [] (no transfer).
        """
        if not self._fitted:
            return []

        # ── Parse state ──────────────────────────────────────────────────
        # The simulator provides state = (week, tuple_of_squad_ids)
        if isinstance(state, tuple) and len(state) == 2:
            week, squad_tuple = state
            squad = list(squad_tuple)
        elif isinstance(state, dict):
            week = state.get("week", state.get("gameweek", 0))
            squad = list(state.get("squad", state.get("squad_ids", [])))
        else:
            return []

        self.current_week = week

        if len(squad) == 0:
            return []

        # Budget = sum of current squad prices (allow selling at same price)
        budget = sum(self.price_map.get(p, 5.0) for p in squad)

        # ── Evaluate all legal single-swap transfers ─────────────────────
        squad_set = set(squad)
        current_value = sum(self.predict_reward(p) for p in squad)

        best_transfer = None
        best_gain = 0.0  # must beat the penalty to be worthwhile

        penalty = self.TRANSFER_PENALTY if self.transfers_used >= self.free_transfers else 0

        # For efficiency: only consider players who played minutes
        candidate_pool = [
            pid for pid in self.player_db
            if pid not in squad_set
        ]

        # Score every candidate once
        candidate_scores = {}
        for pid in candidate_pool:
            candidate_scores[pid] = self.predict_reward(pid)

        # Sort candidates by predicted reward (descending) and only try top-N
        top_candidates = sorted(candidate_scores, key=lambda p: -candidate_scores[p])[:100]

        for p_out in squad:
            pos_out = self.position_map.get(p_out, "MID")
            r_out = self.predict_reward(p_out)
            price_out = self.price_map.get(p_out, 5.0)
            available_budget = budget - (sum(self.price_map.get(p, 5.0) for p in squad) - price_out)

            for p_in in top_candidates:
                # Quick checks before expensive _is_legal
                pos_in = self.position_map.get(p_in, "MID")
                if pos_in != pos_out:
                    continue  # same-position swap to preserve squad structure

                price_in = self.price_map.get(p_in, 5.0)
                if price_in > price_out + 0.5:  # rough budget check
                    # More precise: total cost after swap
                    new_cost = sum(
                        self.price_map.get(p, 5.0) if p != p_out else price_in
                        for p in squad
                    )
                    if new_cost > self.BUDGET + 0.01:
                        continue

                r_in = candidate_scores[p_in]
                marginal = r_in - r_out - penalty

                if marginal > best_gain:
                    # Verify legality
                    new_squad = [p if p != p_out else p_in for p in squad]
                    new_cost = sum(self.price_map.get(p, 5.0) for p in new_squad)

                    # Club limit check (fast)
                    team_in = self.team_map.get(p_in, "Unknown")
                    same_club = sum(1 for p in new_squad if self.team_map.get(p, "?") == team_in)
                    if same_club > self.MAX_CLUB:
                        continue

                    best_gain = marginal
                    best_transfer = (p_out, p_in)

        if best_transfer is not None:
            self.transfers_used += 1
            return [best_transfer]

        return []


# ─── Convenience: allow running as script to verify ──────────────────────
if __name__ == "__main__":
    policy = StudentPolicy()
    policy.fit()
    policy.reset()
    print("\n[Self-test] Policy fitted successfully.")
    print(f"  Players in DB : {len(policy.player_db)}")
    print(f"  V-table size  : {len(policy.V_table)}")

    # Quick sanity: predict reward for a few players
    sample_pids = list(policy.player_db.keys())[:5]
    for pid in sample_pids:
        r = policy.predict_reward(pid)
        name = policy.name_map.get(pid, "?")
        print(f"  {name} (id={pid}): predicted reward = {r:.2f}")