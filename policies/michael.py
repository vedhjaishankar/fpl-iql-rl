"""
DAEN (ISEN) 489 Final Project - Fantasy EPL RL Policy
======================================================
This script implements:
1. Feature engineering from historical FPL data
2. A small neural network reward model (predicts player points per gameweek)
3. A greedy value-based transfer policy (myopic Q-value maximization)
4. Lineup & captain selection via constrained greedy optimization
5. Evaluation vs. no-transfer baseline

The architecture:
- Reward Model: 2-hidden-layer NN (simple, fast to train)
- Transfer Policy: Greedy single-swap based on predicted point differential
- Lineup Selection: Greedy formation-constrained selection
- Captain: Highest predicted scorer gets doubled

Requirements: numpy, pandas, scikit-learn, torch (PyTorch)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SECTION 0: Load Data
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League"
    "/master/data/2023-24/gws/merged_gw.csv"
)

print("Loading data...")
df = pd.read_csv(DATA_URL)
print(f"Raw data shape: {df.shape}")

# ============================================================
# SECTION 1: Feature Engineering
# ============================================================

def engineer_features(df):
    """
    Build features from raw FPL data. Each row is one player-gameweek observation.
    
    Key principle: Features for gameweek t must only use data from gameweeks < t
    to avoid data leakage. We use rolling windows over past k gameweeks.
    
    Returns a DataFrame with engineered features and target column 'points'.
    """
    data = df.copy()
    
    # Standardize column names for compatibility with assignment template
    data = data.rename(columns={
        "total_points": "points",
        "value": "price",       # price in tenths (e.g., 55 = £5.5M)
        "was_home": "home",
        "GW": "gameweek",
        "element": "player_id",
    })
    
    # Sort chronologically for rolling calculations
    data = data.sort_values(["player_id", "gameweek"]).reset_index(drop=True)
    
    # ----- Rolling Window Features (look-back only) -----
    # HYPERPARAMETER: k = window size for rolling averages
    K_SHORT = 3   # short-term form (last 3 gameweeks)
    K_LONG = 6    # longer-term form (last 6 gameweeks)
    
    # Group by player for rolling calculations
    grouped = data.groupby("player_id")
    
    # Form: rolling average of points (shifted by 1 to avoid leakage)
    data["form_short"] = grouped["points"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).mean()
    )
    data["form_long"] = grouped["points"].transform(
        lambda x: x.shift(1).rolling(K_LONG, min_periods=1).mean()
    )
    
    # Minutes form: rolling average of minutes played
    data["minutes_form"] = grouped["minutes"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).mean()
    )
    
    # Goals form: rolling sum of goals scored
    data["goals_form"] = grouped["goals_scored"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).sum()
    )
    
    # Assists form: rolling sum of assists
    data["assists_form"] = grouped["assists"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).sum()
    )
    
    # Clean sheets form (relevant for DEF/GK)
    data["cs_form"] = grouped["clean_sheets"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).sum()
    )
    
    # BPS form (bonus point system - strong predictor)
    data["bps_form"] = grouped["bps"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).mean()
    )
    
    # ICT index form
    data["ict_form"] = grouped["ict_index"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).mean()
    )
    
    # xG-based features (shifted to avoid leakage)
    data["xg_form"] = grouped["expected_goals"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).mean()
    )
    data["xa_form"] = grouped["expected_assists"].transform(
        lambda x: x.shift(1).rolling(K_SHORT, min_periods=1).mean()
    )
    
    # ----- Opponent Strength -----
    # Compute average points conceded by each team (proxy for opponent difficulty)
    team_strength = data.groupby(["opponent_team", "gameweek"])["points"].mean()
    team_avg = team_strength.groupby("opponent_team").mean()
    data["opp_strength"] = data["opponent_team"].map(team_avg).fillna(team_avg.mean())
    
    # ----- Positional Encoding -----
    # One-hot encode position (GK=0, DEF=1, MID=2, FWD=3)
    pos_map = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    data["pos_code"] = data["position"].map(pos_map).fillna(0).astype(int)
    
    # ----- Home/Away -----
    data["home"] = data["home"].astype(int)
    
    # ----- Price (normalize to millions) -----
    data["price_m"] = data["price"] / 10.0
    
    # ----- Selection popularity (crowd wisdom, log-scaled) -----
    data["log_selected"] = np.log1p(data["selected"])
    
    # Fill NaN values (first few gameweeks won't have rolling data)
    data = data.fillna(0.0)
    
    return data


print("Engineering features...")
data = engineer_features(df)
print(f"Engineered data shape: {data.shape}")

# Define feature columns used by the reward model
FEATURE_COLS = [
    "price_m",          # player price in millions
    "form_short",       # short-term points form
    "form_long",        # longer-term points form
    "minutes_form",     # are they playing regularly?
    "goals_form",       # recent goal scoring
    "assists_form",     # recent assists
    "cs_form",          # clean sheet form (DEF/GK)
    "bps_form",         # bonus points form
    "ict_form",         # ICT index form
    "xg_form",          # expected goals form
    "xa_form",          # expected assists form
    "opp_strength",     # opponent difficulty
    "home",             # home advantage
    "pos_code",         # position (0-3)
    "log_selected",     # crowd wisdom
]

# ============================================================
# SECTION 2: Neural Network Reward Model
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class RewardNet(nn.Module):
    """
    Small 2-hidden-layer neural network for predicting player points.
    
    Architecture: Input -> 64 -> 32 -> 1
    - Small enough to train quickly (~seconds on CPU)
    - Large enough to capture non-linear interactions
    
    HYPERPARAMETERS (in this class):
    - hidden1_size: 64 neurons (first hidden layer)
    - hidden2_size: 32 neurons (second hidden layer)
    - dropout_rate: 0.1 (light regularization to prevent overfitting)
    """
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reward_model(X_train, y_train, X_val, y_val):
    """
    Train the neural network reward model.
    
    HYPERPARAMETERS:
    - EPOCHS: 50 (enough for convergence on this dataset size)
    - BATCH_SIZE: 512 (balance between speed and gradient quality)
    - LEARNING_RATE: 1e-3 (Adam default, works well for small nets)
    - WEIGHT_DECAY: 1e-4 (L2 regularization)
    - PATIENCE: 7 (early stopping patience)
    """
    EPOCHS = 50
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 7
    
    input_dim = X_train.shape[1]
    model = RewardNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)
    
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    
    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(X_tr)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    model.load_state_dict(best_state)
    return model


# ============================================================
# SECTION 3: StudentPolicy Class (Required Format)
# ============================================================

class StudentPolicy:
    """
    Policy class as required by the assignment [2].
    
    Methods:
        fit(train_data): Train reward model on historical data
        reset(): Clear rollout-specific memory
        act(state): Return list of transfers [(out_id, in_id), ...]
    
    Strategy:
        - Reward Model: Small NN predicting player gameweek points
        - Transfers: Greedy value-based - swap if predicted improvement exceeds threshold
        - Lineup: Greedy formation-constrained selection of best 11
        - Captain: Player with highest predicted points (earns double)
    """
    
    def __init__(self):
        self.model = None               # PyTorch reward model
        self.feature_cols = FEATURE_COLS # Feature columns for prediction
        self.player_data = None         # Processed player data
        self.scaler_mean = None         # Feature normalization mean
        self.scaler_std = None          # Feature normalization std
        
        # ----- POLICY HYPERPARAMETERS -----
        # TRANSFER_THRESHOLD: minimum predicted point improvement to justify a transfer
        # Set > 0 to avoid frivolous swaps; represents opportunity cost
        self.TRANSFER_THRESHOLD = 0.5
        
        # TOP_K_CANDIDATES: number of replacement candidates to consider per position
        # Limits computation in act(); higher = better but slower
        self.TOP_K_CANDIDATES = 20
        
        # Budget and squad constraints from assignment [2]
        self.BUDGET = 1000  # £100M in tenths
        self.MAX_PER_CLUB = 3
        self.SQUAD_SIZE = 15
        self.POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        
    def fit(self, train_data):
        """
        Train the reward prediction model using historical data [2].
        
        Steps:
        1. Engineer features from raw data
        2. Normalize features
        3. Train/validate split (temporal - last 6 GWs as validation)
        4. Train neural network
        5. Store player metadata for transfer decisions
        """
        # Engineer features
        processed = engineer_features(train_data)
        self.player_data = processed
        
        # Temporal train/val split
        # HYPERPARAMETER: VAL_WEEKS = 6 (last 6 gameweeks for validation)
        VAL_WEEKS = 6
        max_gw = processed["gameweek"].max()
        val_start = max_gw - VAL_WEEKS + 1
        
        train_mask = processed["gameweek"] < val_start
        val_mask = processed["gameweek"] >= val_start
        
        X_train = processed.loc[train_mask, self.feature_cols].values
        y_train = processed.loc[train_mask, "points"].values
        X_val = processed.loc[val_mask, self.feature_cols].values
        y_val = processed.loc[val_mask, "points"].values
        
        # Normalize features (store mean/std for inference)
        self.scaler_mean = X_train.mean(axis=0)
        self.scaler_std = X_train.std(axis=0) + 1e-8  # avoid division by zero
        
        X_train_norm = (X_train - self.scaler_mean) / self.scaler_std
        X_val_norm = (X_val - self.scaler_mean) / self.scaler_std
        
        # Train the neural network
        print("Training reward model...")
        self.model = train_reward_model(X_train_norm, y_train, X_val_norm, y_val)
        self.model.eval()
        
        # Store per-player latest features for transfer decisions
        self._build_player_lookup(processed)
        
        print("Model training complete.")
        
    def _build_player_lookup(self, processed):
        """
        Build a lookup table: for each player, store their latest features,
        position, team, and price. Used during act() for transfer evaluation.
        """
        # Get last available gameweek data per player
        latest = processed.sort_values("gameweek").groupby("player_id").last().reset_index()
        self.player_lookup = latest.set_index("player_id")[
            self.feature_cols + ["position", "team", "price", "name"]
        ].to_dict("index")
        self.all_player_ids = list(self.player_lookup.keys())
    
    def reset(self):
        """
        Reset any rollout-specific memory [2].
        Called once at the start of each evaluation rollout.
        """
        self.current_week = 0
    
    def predict_reward(self, player_dict):
        """
        Predict reward for one player given their feature dictionary [2].
        
        Args:
            player_dict: dict with keys matching self.feature_cols
            
        Returns:
            float: predicted points for this gameweek
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        
        # Extract features in correct order
        features = np.array([player_dict.get(col, 0.0) for col in self.feature_cols])
        
        # Normalize
        features_norm = (features - self.scaler_mean) / self.scaler_std
        
        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(features_norm).unsqueeze(0)
            pred = self.model(x).item()
        return pred
    
    def _predict_all_players(self, gameweek_data):
        """
        Predict points for all players available in a given gameweek.
        
        Args:
            gameweek_data: DataFrame filtered to current gameweek
            
        Returns:
            dict: {player_id: predicted_points}
        """
        predictions = {}
        for _, row in gameweek_data.iterrows():
            pid = row["player_id"]
            features = np.array([row.get(col, 0.0) for col in self.feature_cols])
            features_norm = (features - self.scaler_mean) / self.scaler_std
            with torch.no_grad():
                x = torch.FloatTensor(features_norm).unsqueeze(0)
                pred = self.model(x).item()
            predictions[pid] = pred
        return predictions
    
    def _select_lineup(self, squad_ids, predictions):
        """
        Greedy formation-constrained lineup selection from 15-player squad.
        
        Rules [1][2]:
        - Starting XI must have at least 3 DEF, 2 MID, 1 FWD
        - Remaining spots filled by highest predicted scorers
        - Captain = highest predicted scorer (earns double points)
        
        Args:
            squad_ids: list of player_ids in current squad
            predictions: dict {player_id: predicted_points}
            
        Returns:
            tuple: (starting_11_ids, captain_id)
        """
        # Minimum formation requirements
        MIN_DEF = 3
        MIN_MID = 2
        MIN_FWD = 1
        MIN_GK = 1
        STARTING_SIZE = 11
        
        # Separate players by position and sort by predicted points
        by_position = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for pid in squad_ids:
            if pid in self.player_lookup:
                pos = self.player_lookup[pid].get("position", "MID")
                pred = predictions.get(pid, 0.0)
                by_position[pos].append((pid, pred))
        
        # Sort each position by predicted points (descending)
        for pos in by_position:
            by_position[pos].sort(key=lambda x: x[1], reverse=True)
        
        # Select minimum required per position
        starting = []
        starting += by_position["GK"][:MIN_GK]
        starting += by_position["DEF"][:MIN_DEF]
        starting += by_position["MID"][:MIN_MID]
        starting += by_position["FWD"][:MIN_FWD]
        
        # Fill remaining spots from leftover players by predicted points
        selected_ids = {pid for pid, _ in starting}
        remaining = []
        for pos in by_position:
            min_req = {"GK": MIN_GK, "DEF": MIN_DEF, "MID": MIN_MID, "FWD": MIN_FWD}[pos]
            remaining += by_position[pos][min_req:]
        
        remaining.sort(key=lambda x: x[1], reverse=True)
        spots_left = STARTING_SIZE - len(starting)
        
        for pid, pred in remaining:
            if spots_left <= 0:
                break
            # Don't add more than 1 GK to starting lineup
            if self.player_lookup.get(pid, {}).get("position") == "GK":
                continue
            starting.append((pid, pred))
            spots_left -= 1
        
        starting_ids = [pid for pid, _ in starting]
        
        # Captain = highest predicted scorer (earns double points) [2]
        captain_id = max(starting, key=lambda x: x[1])[0] if starting else None
        
        return starting_ids, captain_id
    
    def _get_transfer_candidates(self, squad_ids, budget, gameweek_data, predictions):
        """
        Find the best single transfer (greedy approach).
        
        Logic:
        - For each player in the squad, consider replacing with a better player
        - The replacement must: be affordable, not violate club limit, same position
        - Only make the transfer if improvement exceeds TRANSFER_THRESHOLD
        
        Args:
            squad_ids: current squad player_ids
            budget: remaining budget (in tenths)
            gameweek_data: DataFrame for current gameweek
            predictions: dict {player_id: predicted_points}
            
        Returns:
            list: [(player_out_id, player_in_id)] or []
        """
        best_improvement = 0.0
        best_transfer = None
        
        # Count players per club in current squad
        club_count = {}
        for pid in squad_ids:
            if pid in self.player_lookup:
                team = self.player_lookup[pid].get("team", "")
                club_count[team] = club_count.get(team, 0) + 1
        
        # For each squad player, find best replacement
        for pid_out in squad_ids:
            if pid_out not in self.player_lookup:
                continue
            
            out_info = self.player_lookup[pid_out]
            out_pos = out_info.get("position", "MID")
            out_price = out_info.get("price", 50)
            out_team = out_info.get("team", "")
            out_pred = predictions.get(pid_out, 0.0)
            
            # Available budget if we sell this player
            available_budget = budget + out_price
            
            # Consider replacements of the same position
            for pid_in in self.all_player_ids:
                if pid_in in squad_ids:
                    continue  # already in squad
                if pid_in not in self.player_lookup:
                    continue
                
                in_info = self.player_lookup[pid_in]
                in_pos = in_info.get("position", "MID")
                in_price = in_info.get("price", 50)
                in_team = in_info.get("team", "")
                in_pred = predictions.get(pid_in, 0.0)
                
                # Position must match
                if in_pos != out_pos:
                    continue
                
                # Must be affordable
                if in_price > available_budget:
                    continue
                
                # Club limit: max 3 from same club [2]
                if in_team != out_team:
                    if club_count.get(in_team, 0) >= self.MAX_PER_CLUB:
                        continue
                
                # Calculate improvement
                improvement = in_pred - out_pred
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_transfer = (pid_out, pid_in)
        
        # Only make transfer if improvement exceeds threshold
        if best_transfer and best_improvement > self.TRANSFER_THRESHOLD:
            return [best_transfer]
        
        return []
    
    def act(self, state):
        """
        Return a list of transfers [(player_out_id, player_in_id), ...] [2].
        
        This implements a greedy value-based transfer policy:
        1. Predict points for all available players
        2. Find the single best swap that improves predicted squad value
        3. Only execute if improvement > threshold (to avoid churn)
        
        Args:
            state: tuple provided by environment (week, squad, budget, features)
            
        Returns:
            list: transfers as [(out_id, in_id)] or [] for no transfer
        """
        self.current_week += 1
        
        # Parse state - adapt to whatever the environment provides
        # Based on the template, state is (week, squad) at minimum [1]
        if isinstance(state, tuple) and len(state) >= 2:
            week = state[0]
            squad_ids = list(state[1]) if isinstance(state[1], (tuple, list)) else []
            budget = state[2] if len(state) > 2 else self.BUDGET
        else:
            return []
        
        # Get predictions for all players using stored lookup
        predictions = {}
        for pid in self.all_player_ids:
            if pid in self.player_lookup:
                predictions[pid] = self.predict_reward(self.player_lookup[pid])
        
        # Find best transfer
        transfers = self._get_transfer_candidates(squad_ids, budget, None, predictions)
        
        return transfers


# ============================================================
# SECTION 4: Evaluation Environment (Extended from Template)
# ============================================================

class FPLEnvFull:
    """
    Extended FPL environment for evaluation.
    Based on the template in Appendix B [1][2], but with full constraints:
    - 15-player squad with positional requirements
    - Budget tracking
    - Club limits (max 3 per club)
    - Lineup selection and captaincy
    - Transfer penalty of -4 for extra transfers [2]
    - Randomized initial squad (per professor's template) [1]
    - Noise on rewards (per professor's PlayerSimulator) [1]
    """

    def __init__(self, data, policy, noise_std=1.0):
        """
        Args:
            data: processed DataFrame with engineered features
            policy: StudentPolicy instance (for lineup/captain decisions)
            noise_std: noise added to rewards (professor default = 1.0) [1]
        """
        self.data = data
        self.policy = policy
        self.noise_std = noise_std
        self.all_players = data["player_id"].unique()
        self.gameweeks = sorted(data["gameweek"].unique())

        # Constraints from assignment [2]
        self.SQUAD_SIZE = 15
        self.BUDGET = 1000  # £100M in tenths
        self.MAX_PER_CLUB = 3
        self.POSITION_REQ = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        self.TRANSFER_PENALTY = -4  # penalty per extra transfer [2]

    def _get_valid_initial_squad(self):
        """
        Select a valid initial squad with randomization.

        Fix 1: Randomize selection (professor's template uses np.random.choice) [1]
        Fix 3: Filter to players with meaningful minutes so baseline is realistic
        """
        gw1_data = self.data[self.data["gameweek"] == self.gameweeks[0]]
        players_gw1 = gw1_data.drop_duplicates("player_id").set_index("player_id")

        squad = []
        total_cost = 0
        club_count = {}

        for pos, count in self.POSITION_REQ.items():
            pos_players = players_gw1[players_gw1["position"] == pos]

            # Fix 3: Filter to players who actually play (minutes_form > 30)
            # This ensures the no-transfer baseline has a reasonable squad
            active_players = pos_players[pos_players["minutes_form"] > 30]
            if len(active_players) >= count:
                pos_players = active_players

            # Fix 1: Shuffle randomly instead of sorting by cheapest
            pos_players = pos_players.sample(frac=1.0)

            selected = 0
            for pid, row in pos_players.iterrows():
                if selected >= count:
                    break

                team = row.get("team", "")
                price = row.get("price", 50)

                # Check club limit [2]
                if club_count.get(team, 0) >= self.MAX_PER_CLUB:
                    continue

                # Check budget (leave room for remaining players)
                remaining_slots = self.SQUAD_SIZE - len(squad) - 1
                if remaining_slots > 0:
                    if total_cost + price > self.BUDGET - remaining_slots * 40:
                        continue
                else:
                    if total_cost + price > self.BUDGET:
                        continue

                squad.append(pid)
                total_cost += price
                club_count[team] = club_count.get(team, 0) + 1
                selected += 1

        remaining_budget = self.BUDGET - total_cost
        return squad, remaining_budget

    def reset(self):
        """Reset environment for a new rollout [1]."""
        self.current_gw_idx = 0
        self.squad, self.budget = self._get_valid_initial_squad()
        self.free_transfers = 1  # always 1, no carry over [2]
        return self._get_state()

    def _get_state(self):
        """Return current state as tuple."""
        gw = self.gameweeks[self.current_gw_idx]
        return (gw, tuple(self.squad), self.budget)

    def _get_actual_points(self, player_id, gameweek):
        """
        Get actual historical points for a player in a gameweek.
        Adds Gaussian noise per professor's PlayerSimulator template [1].
        """
        row = self.data[
            (self.data["player_id"] == player_id) &
            (self.data["gameweek"] == gameweek)
        ]
        if len(row) == 0:
            return 0.0
        points = row["points"].values[0]
        # Fix 2: Add noise matching professor's template [1]
        if self.noise_std > 0:
            points += np.random.normal(0, self.noise_std)
        return max(0, points)

    def step(self, transfers):
        """
        Execute transfers and compute reward for current gameweek.

        Args:
            transfers: list of (out_id, in_id) tuples, or []

        Returns:
            tuple: (next_state, reward, done)
        """
        gw = self.gameweeks[self.current_gw_idx]

        # Apply transfer penalty if more than 1 transfer [2]
        penalty = 0
        num_transfers = len(transfers) if transfers else 0
        if num_transfers > 1:
            penalty = (num_transfers - 1) * self.TRANSFER_PENALTY

        # Execute transfers
        if transfers:
            for p_out, p_in in transfers:
                if p_out in self.squad:
                    out_price = self.policy.player_lookup.get(p_out, {}).get("price", 50)
                    in_price = self.policy.player_lookup.get(p_in, {}).get("price", 50)
                    self.budget += out_price - in_price
                    self.squad = [p if p != p_out else p_in for p in self.squad]

        # Get predictions for lineup selection
        predictions = {}
        for pid in self.squad:
            if pid in self.policy.player_lookup:
                predictions[pid] = self.policy.predict_reward(self.policy.player_lookup[pid])
            else:
                predictions[pid] = 0.0

        # Select lineup and captain
        starting_11, captain_id = self.policy._select_lineup(self.squad, predictions)

        # Calculate actual reward from starting 11 + captain bonus
        reward = 0.0
        for pid in starting_11:
            pts = self._get_actual_points(pid, gw)
            if pid == captain_id:
                reward += pts * 2  # captain earns double [2]
            else:
                reward += pts

        # Add transfer penalty
        reward += penalty

        # Advance gameweek
        self.current_gw_idx += 1
        done = self.current_gw_idx >= len(self.gameweeks)

        next_state = self._get_state() if not done else None
        return next_state, reward, done


# ============================================================
# SECTION 5: No-Transfer Baseline (from template [1])
# ============================================================

def run_no_transfer(env, policy):
    """
    Baseline policy that makes zero transfers throughout the season [1].
    Still uses lineup selection and captaincy.
    
    Returns:
        float: total cumulative reward over all gameweeks
    """
    state = env.reset()
    policy.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        # No transfers, just select lineup
        state, reward, done = env.step([])
        total_reward += reward
    
    return total_reward


def run_with_policy(env, policy):
    """
    Run a full season using the StudentPolicy's act() method.
    
    Returns:
        float: total cumulative reward over all gameweeks
    """
    state = env.reset()
    policy.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        # Get transfer decision from policy
        transfers = policy.act(state)
        state, reward, done = env.step(transfers)
        total_reward += reward
    
    return total_reward


# ============================================================
# SECTION 6: Main Execution - Train, Evaluate, Compare
# ============================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("FANTASY EPL RL POLICY - TRAINING AND EVALUATION")
    print("=" * 60)
    
    # --- Step 1: Feature Engineering (already done above) ---
    print("\n[Step 1] Feature engineering complete.")
    print(f"  Features used: {FEATURE_COLS}")
    print(f"  Total observations: {len(data)}")
    
    # --- Step 2: Train Reward Model ---
    print("\n[Step 2] Training reward model...")
    policy = StudentPolicy()
    policy.fit(df)  # Pass raw data; fit() handles feature engineering internally
    
    # --- Step 3: Evaluate reward model quality ---
    print("\n[Step 3] Reward model evaluation metrics:")
    
    # Re-engineer features for evaluation
    eval_data = engineer_features(df)
    max_gw = eval_data["gameweek"].max()
    VAL_WEEKS = 6
    val_mask = eval_data["gameweek"] >= (max_gw - VAL_WEEKS + 1)
    
    X_val = eval_data.loc[val_mask, FEATURE_COLS].values
    y_val = eval_data.loc[val_mask, "points"].values
    
    # Normalize and predict
    X_val_norm = (X_val - policy.scaler_mean) / policy.scaler_std
    with torch.no_grad():
        y_pred = policy.model(torch.FloatTensor(X_val_norm)).numpy()
    
    # Compute metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"  Validation MSE:  {mse:.4f}")
    print(f"  Validation RMSE: {rmse:.4f}")
    print(f"  Validation MAE:  {mae:.4f}")
    print(f"  Validation R²:   {r2:.4f}")
    
    # Baseline comparison: predicting mean
    mean_pred = np.full_like(y_val, y_val.mean())
    mse_baseline = mean_squared_error(y_val, mean_pred)
    print(f"  Baseline MSE (predict mean): {mse_baseline:.4f}")
    print(f"  Model improvement over baseline: {((mse_baseline - mse) / mse_baseline * 100):.1f}%")
    
    # --- Step 4: Policy Evaluation ---
    print("\n[Step 4] Policy evaluation (simulated season rollouts)...")
    
    # HYPERPARAMETER: N_ROLLOUTS = number of evaluation episodes
    # The professor evaluates with N rollouts [1][2]
    N_ROLLOUTS = 10
    
    # Create evaluation environment
    env = FPLEnvFull(data=eval_data, policy=policy, noise_std=1.0)
    # --- Run policy rollouts ---
    policy_rewards = []
    notransfer_rewards = []
    
    for i in range(N_ROLLOUTS):
        # Run with greedy transfer policy
        pr = run_with_policy(env, policy)
        policy_rewards.append(pr)
        
        # Run no-transfer baseline [1]
        nr = run_no_transfer(env, policy)
        notransfer_rewards.append(nr)
        
        print(f"  Rollout {i+1}/{N_ROLLOUTS}: "
              f"Policy={pr:.1f}, NoTransfer={nr:.1f}, "
              f"Diff={pr - nr:+.1f}")
    
    # --- Step 5: Summary Statistics ---
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    policy_rewards = np.array(policy_rewards)
    notransfer_rewards = np.array(notransfer_rewards)
    
    # V(pi) as defined in the assignment: average cumulative reward across rollouts [1][2]
    V_policy = policy_rewards.mean()
    V_notransfer = notransfer_rewards.mean()
    
    print(f"\n  Reward Model Metrics (validation set, last {VAL_WEEKS} GWs):")
    print(f"    RMSE:              {rmse:.4f}")
    print(f"    MAE:               {mae:.4f}")
    print(f"    R²:                {r2:.4f}")
    print(f"    MSE improvement over mean-baseline: {((mse_baseline - mse) / mse_baseline * 100):.1f}%")
    
    print(f"\n  Policy Evaluation ({N_ROLLOUTS} rollouts):")
    print(f"    V(policy)      = {V_policy:.2f}  (mean cumulative reward)")
    print(f"    V(no_transfer) = {V_notransfer:.2f}  (mean cumulative reward)")
    print(f"    Improvement    = {V_policy - V_notransfer:+.2f} points")
    print(f"    Improvement %  = {((V_policy - V_notransfer) / max(V_notransfer, 1) * 100):+.1f}%")
    
    print(f"\n  Policy Std Dev:       {policy_rewards.std():.2f}")
    print(f"  NoTransfer Std Dev:   {notransfer_rewards.std():.2f}")
    
    # Per-gameweek average (normalize by number of gameweeks)
    n_gws = len(env.gameweeks)
    print(f"\n  Avg points per gameweek (Policy):      {V_policy / n_gws:.2f}")
    print(f"  Avg points per gameweek (NoTransfer):  {V_notransfer / n_gws:.2f}")
    
    print("\n" + "=" * 60)
    print("DONE. Submit policy.py containing the StudentPolicy class.")
    print("=" * 60)
