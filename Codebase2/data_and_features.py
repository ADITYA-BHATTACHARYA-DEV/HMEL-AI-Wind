# File 1: data_and_features.py

import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        df = pd.read_csv(self.filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df


class FeatureEngineer:
    def __init__(self):
        pass

    def add_circular_features(self, df):
        df['dir_sin'] = np.sin(np.deg2rad(df['Value']))
        df['dir_cos'] = np.cos(np.deg2rad(df['Value']))
        return df
