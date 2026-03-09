# Credit Card Fraud Detection

Detecting fraudulent transactions using Anomaly Detection & Machine Learning.

## Dataset
Download `creditcard.csv` from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place it in the `data/` folder.

## Setup

```bash
# 1. Create virtual environment
python -m venv fraud_env

# 2. Activate it
# macOS/Linux:
source fraud_env/bin/activate
# Windows:
fraud_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open VS Code
code .
```

## Project Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv          ← place dataset here
├── notebooks/
│   ├── 01_eda.ipynb            ← Step 1: Explore the data
│   ├── 02_preprocessing.ipynb  ← Step 2: Clean & prepare
│   ├── 03_isolation_forest.ipynb
│   ├── 04_autoencoder.ipynb
│   └── 05_neural_network.ipynb
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── evaluate.py
│   └── inference.py
├── outputs/
│   ├── figures/
│   └── models/
├── requirements.txt
└── README.md
```

## Notebooks — Run in Order
1. `01_eda.ipynb` — Class imbalance, distributions, correlations
2. `02_preprocessing.ipynb` — Scaling, SMOTE, train/val/test split
3. `03_isolation_forest.ipynb` — Unsupervised baseline
4. `04_autoencoder.ipynb` — Semi-supervised deep learning
5. `05_neural_network.ipynb` — Supervised classifier (best performance)

## Key Metric: PR-AUC (not accuracy!)
Accuracy is misleading on this dataset (99.8% by predicting everything as legit).
Focus on **Recall** and **PR-AUC**.
