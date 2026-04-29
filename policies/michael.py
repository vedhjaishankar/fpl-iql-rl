import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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
