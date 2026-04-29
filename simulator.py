import numpy as np
import pandas as pd
class PlayerSimulator:
    def __init__(self, model, noise_std=1.0):
        self.model = model
        self.noise_std = noise_std

    def predict_points(self, features):
        pred = self.model.predict(features.reshape(1, -1))[0]
        noise = np.random.normal(0, self.noise_std)
        return max(0, pred + noise)
class FPLEnv:
    def __init__(self, data, model, squad_size=5):
        self.data = data
        self.model = model
        self.sim = PlayerSimulator(model)
        self.players = data["player_id"].unique()
        self.squad_size = squad_size
    def reset(self):
        self.week = self.data["gameweek"].min()
        self.squad = np.random.choice(self.players, self.squad_size, replace=False)
        return self._get_state()
    def _get_state(self):
        return (self.week, tuple(self.squad))
    def _get_features(self, player_id):
        row = self.data[
            (self.data["player_id"] == player_id) &
            (self.data["gameweek"] == self.week)
        ]
        if len(row) == 0:
            return None
        return row[["price","form","minutes_form"]].values[0]
        
    def step(self, action):
        if action is not None:
            p_out, p_in = action
            self.squad = np.array([
            p if p != p_out else p_in for p in self.squad
            ])

        reward = 0
        for p in self.squad:
            feat = self._get_features(p)
            if feat is not None:
                reward += self.sim.predict_points(feat)
        self.week += 1
        done = self.week > self.data["gameweek"].max()
        return self._get_state(), reward, done

    def run_no_transfer(env):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state, reward, done = env.step(None)
            total_reward += reward
        return total_reward