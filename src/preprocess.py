"""
preprocess.py
Reusable preprocessing functions for the fraud detection pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    """Load the creditcard CSV and print a basic summary."""
    df = pd.read_csv(filepath)
    print(f"Shape        : {df.shape}")
    print(f"Fraud cases  : {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    print(f"Missing vals : {df.isnull().sum().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and log-transformed features."""
    df = df.copy()
    df['Hour']       = (df['Time'] // 3600) % 24          # hour of day
    df['Log_Amount'] = np.log1p(df['Amount'])              # compress skewed amount
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, StandardScaler]:
    """Scale Amount and Time columns; return df + fitted scalers."""
    scaler_amount = StandardScaler()
    scaler_time   = StandardScaler()
    df = df.copy()
    df['Amount'] = scaler_amount.fit_transform(df[['Amount']])
    df['Time']   = scaler_time.fit_transform(df[['Time']])
    return df, scaler_amount, scaler_time


def split_data(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
    """Stratified 70 / 15 / 15 train-val-test split."""
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """Apply SMOTE to training set ONLY to balance classes."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE  — Fraud: {y_res.sum()} | Legit: {(y_res==0).sum()}")
    return X_res, y_res
