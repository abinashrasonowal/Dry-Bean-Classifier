import pandas as pd
import numpy as np
from dataloader import df
from sklearn.preprocessing import LabelEncoder

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle the indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    # Calculate the split index
    split_idx = int(len(indices) * (1 - test_size))
    
    # Split the indices
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        # Compute the mean and standard deviation to be used for later scaling.
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        # Perform standardization by centering and scaling.
        if self.mean_ is None or self.scale_ is None:
            raise Exception("The scaler has not been fitted yet.")
        
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled
    
    def fit_transform(self, X):
        # Fit to data, then transform it.
        self.fit(X)
        return self.transform(X)

def get_clean_dataset(df = df):
    le = LabelEncoder()
    df = df[(df["Area"] < 100000) | (df["ConvexArea"] < 100000)]  # Filtering
    df = df.copy()
    df["Class"] = le.fit_transform(df["Class"])  # Label Encoding
    return df
