import numpy as np
import pandas as pd
class StudentPolicy:
    def __init__(self):
        self.model = None
        self.feature_cols = None

    def fit(self, train_data):
        """
        Fit a reward prediction model using historical data.
        Students should replace this baseline with their own model.
        """
        # Example placeholder
        self.feature_cols = ["price", "form", "minutes_form"]
        X = train_data[self.feature_cols].fillna(0.0).values
        y = train_data["points"].values
        # Dummy baseline: predict the mean
        self.model = {"mean_points": float(np.mean(y))}
    def reset(self):
        """
        Reset any rollout-specific memory.
        """
        pass
    def predict_reward(self, player_dict):
        """
        Predict reward for one player from current features.
        """
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        # Dummy baseline
        return self.model["mean_points"]
    def act(self, state):
        """
        Return a list of transfers [(player_out_id, player_in_id), ...]
        """
        return []
