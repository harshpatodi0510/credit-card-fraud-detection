"""
models.py
Model definitions: Isolation Forest, Autoencoder, Neural Network.
"""

import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X_train, contamination: float = 0.002):
    """Train Isolation Forest. contamination ≈ expected fraud rate."""
    model = IsolationForest(contamination=contamination, n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train)
    return model


def predict_isolation_forest(model, X):
    """Return binary predictions (1=fraud, 0=legit)."""
    raw = model.predict(X)
    return np.where(raw == -1, 1, 0)


def build_autoencoder(input_dim: int):
    """Bottleneck autoencoder trained on normal transactions only."""
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    bottleneck = tf.keras.layers.Dense(8, activation='relu')(x)   # compressed representation
    x = tf.keras.layers.Dense(32, activation='relu')(bottleneck)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)

    ae = tf.keras.Model(inputs, outputs)
    ae.compile(optimizer='adam', loss='mse')
    return ae


def get_reconstruction_errors(ae, X):
    """Mean squared reconstruction error per sample."""
    X_pred = ae.predict(X, verbose=0)
    return np.mean(np.power(X - X_pred, 2), axis=1)


def build_neural_network(input_dim: int):
    """Supervised binary classifier with dropout regularisation."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=input_dim, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model
