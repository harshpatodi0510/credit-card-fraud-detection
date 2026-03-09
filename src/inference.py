"""
inference.py
score_transaction() — single-transaction fraud scoring pipeline.
"""

import numpy as np
import pandas as pd


def score_transaction(transaction: dict, scaler_amount, scaler_time, model, threshold: float = 0.3) -> dict:
    """
    Score a single transaction dict against the trained neural network.

    Parameters
    ----------
    transaction   : dict with keys matching training feature columns
    scaler_amount : fitted StandardScaler for Amount
    scaler_time   : fitted StandardScaler for Time
    model         : trained Keras model
    threshold     : decision boundary (default 0.3 — tuned for high recall)

    Returns
    -------
    dict: fraud_probability, risk_label, risk_tier, threshold_used
    """
    df = pd.DataFrame([transaction])
    df['Amount'] = scaler_amount.transform(df[['Amount']])
    df['Time']   = scaler_time.transform(df[['Time']])

    prob = float(model.predict(df, verbose=0)[0][0])

    if prob >= 0.7:
        tier = "HIGH RISK"
    elif prob >= threshold:
        tier = "MEDIUM RISK"
    else:
        tier = "LOW RISK"

    return {
        "fraud_probability": round(prob, 4),
        "risk_label"       : "FRAUD" if prob >= threshold else "LEGITIMATE",
        "risk_tier"        : tier,
        "threshold_used"   : threshold,
    }
