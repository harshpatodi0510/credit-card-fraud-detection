"""
app.py — Fraud Detection Streamlit Dashboard
Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────────
MODELS_PATH = '/Users/harshpatodi/Downloads/fraud-detection/outputs/models/'
DATA_PATH   = '/Users/harshpatodi/Downloads/fraud-detection/data/creditcard.csv'

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        color: #1565C0; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F0F4FF; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #1565C0;
    }
    .fraud-alert {
        background: #FFEBEE; border-radius: 10px;
        padding: 1.2rem; border-left: 6px solid #F44336;
        font-size: 1.1rem; font-weight: 600; color: #C62828;
    }
    .legit-alert {
        background: #E8F5E9; border-radius: 10px;
        padding: 1.2rem; border-left: 6px solid #4CAF50;
        font-size: 1.1rem; font-weight: 600; color: #2E7D32;
    }
    .medium-alert {
        background: #FFF8E1; border-radius: 10px;
        padding: 1.2rem; border-left: 6px solid #FF9800;
        font-size: 1.1rem; font-weight: 600; color: #E65100;
    }
    .section-title {
        font-size: 1.2rem; font-weight: 700;
        color: #1565C0; margin-top: 1rem;
        border-bottom: 2px solid #E3F2FD;
        padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model & artifacts (cached) ───────────────────────────
@st.cache_resource
def load_artifacts():
    model         = keras.models.load_model(f'{MODELS_PATH}neural_network.keras')
    scaler_amount = joblib.load(f'{MODELS_PATH}scaler_amount.pkl')
    scaler_time   = joblib.load(f'{MODELS_PATH}scaler_time.pkl')
    feature_cols  = joblib.load(f'{MODELS_PATH}feature_cols.pkl')
    return model, scaler_amount, scaler_time, feature_cols

@st.cache_data
def load_sample_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_shap_explainer(_model, _feature_cols):
    """Load SHAP explainer — cached so it only builds once."""
    df        = load_sample_data()
    df['Hour']       = (df['Time'] // 3600) % 24
    df['Log_Amount'] = np.log1p(df['Amount'])
    background = df[[c for c in _feature_cols]].sample(100, random_state=42).values

    def predict_fn(X):
        return _model.predict(X, verbose=0).flatten()

    explainer = shap.KernelExplainer(predict_fn, background)
    return explainer


# ── Scoring function ───────────────────────────────────────────
def score_transaction(raw: dict, model, scaler_amount, scaler_time,
                      feature_cols, threshold=0.3):
    df_in = pd.DataFrame([raw])
    df_in['Hour']       = (df_in['Time'] // 3600) % 24
    df_in['Log_Amount'] = np.log1p(df_in['Amount'])
    df_in['Amount']     = scaler_amount.transform(df_in[['Amount']])
    df_in['Time']       = scaler_time.transform(df_in[['Time']])
    df_in = df_in.reindex(columns=feature_cols, fill_value=0)
    prob  = float(model.predict(df_in.values, verbose=0)[0][0])

    if prob >= 0.7:
        tier, color = "HIGH RISK 🔴", "fraud"
    elif prob >= threshold:
        tier, color = "MEDIUM RISK 🟡", "medium"
    else:
        tier, color = "LOW RISK 🟢", "legit"

    return {
        "probability" : round(prob, 4),
        "label"       : "FRAUD" if prob >= threshold else "LEGITIMATE",
        "tier"        : tier,
        "color"       : color,
        "threshold"   : threshold,
        "input_df"    : df_in,
    }


def shap_waterfall(shap_vals, feature_names, feature_values, prob, n=12):
    """Plot a SHAP waterfall chart for a single transaction."""
    sv = pd.Series(shap_vals, index=feature_names)
    top = sv.abs().sort_values(ascending=False).head(n).index
    sv_top  = sv[top]
    val_top = feature_values[top]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ['#F44336' if v > 0 else '#2196F3' for v in sv_top.values]
    ax.barh(range(len(sv_top)), sv_top.values[::-1],
            color=colors[::-1], edgecolor='white', height=0.6)
    ax.set_yticks(range(len(sv_top)))
    ax.set_yticklabels(
        [f'{f} = {val_top[f]:.3f}' for f in top[::-1]], fontsize=9
    )
    ax.axvline(x=0, color='black', lw=1)
    ax.set_xlabel('SHAP Value  (→ fraud  |  ← legitimate)')
    ax.set_title(f'Why was this transaction scored {prob:.4f}?',
                 fontweight='bold', fontsize=11)

    red_patch  = mpatches.Patch(color='#F44336', label='Pushes toward FRAUD')
    blue_patch = mpatches.Patch(color='#2196F3', label='Pushes toward LEGITIMATE')
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc='lower right')
    plt.tight_layout()
    return fig


# ── Load everything ────────────────────────────────────────────
model, scaler_amount, scaler_time, feature_cols = load_artifacts()
df_raw = load_sample_data()

V_COLS = [f'V{i}' for i in range(1, 29)]

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90,
        value=0.30, step=0.05,
        help="Lower = catch more fraud (more false alarms). Higher = fewer false alarms (miss more fraud)."
    )
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"""
    | | |
    |---|---|
    | **Model** | Neural Network |
    | **PR-AUC** | 0.7630 (random) |
    | **PR-AUC** | 0.6488 (time-split) |
    | **Recall** | 75.68% |
    | **Features** | {len(feature_cols)} |
    """)
    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("""
    1. Pick a **sample transaction** or enter custom V-values  
    2. Click **Score Transaction**  
    3. View fraud probability + SHAP explanation  
    """)
    show_shap = st.checkbox("Show SHAP explanation", value=True,
                             help="Adds ~15 seconds — explains why the score was given")

# ══════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🔍 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time transaction scoring · Neural Network · SHAP Explainability</div>', unsafe_allow_html=True)

# ── Top metrics row ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
fraud_count  = int(df_raw['Class'].sum())
total        = len(df_raw)
col1.metric("Total Transactions",  f"{total:,}")
col2.metric("Fraud Cases",         f"{fraud_count:,}")
col3.metric("Fraud Rate",          f"{fraud_count/total*100:.3f}%")
col4.metric("Model PR-AUC",        "0.7630")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔎 Score a Transaction", "📊 Dataset Explorer", "📈 Model Performance"])

# ────────────────────────────────────────────────────────────
# TAB 1 — SCORE A TRANSACTION
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Select or Enter a Transaction</div>', unsafe_allow_html=True)

    mode = st.radio("Input mode", ["Pick a real example", "Enter custom values"],
                    horizontal=True)

    if mode == "Pick a real example":
        col_a, col_b = st.columns(2)
        with col_a:
            sample_type = st.selectbox(
                "Transaction type",
                ["Random Fraud", "Random Legitimate", "Highest Confidence Fraud",
                 "Lowest Confidence Fraud"]
            )
        with col_b:
            seed = st.number_input("Random seed", value=42, min_value=0, max_value=999)

        np.random.seed(seed)
        if sample_type == "Random Fraud":
            row = df_raw[df_raw['Class'] == 1].sample(1, random_state=seed).iloc[0]
        elif sample_type == "Random Legitimate":
            row = df_raw[df_raw['Class'] == 0].sample(1, random_state=seed).iloc[0]
        elif sample_type == "Highest Confidence Fraud":
            fraud_rows = df_raw[df_raw['Class'] == 1]
            row = fraud_rows.iloc[0]
        else:
            fraud_rows = df_raw[df_raw['Class'] == 1]
            row = fraud_rows.iloc[-1]

        transaction = row.drop('Class').to_dict()
        true_label  = int(row['Class'])
        st.info(f"**True label:** {'🔴 FRAUD' if true_label == 1 else '🟢 LEGITIMATE'}  |  "
                f"Time: {transaction['Time']:.0f}s  |  Amount: ${transaction['Amount']:.2f}")

    else:
        st.markdown("**Enter V1–V5 key features (others default to 0.0):**")
        c1, c2, c3, c4, c5 = st.columns(5)
        v_vals = {
            'V1' : c1.number_input('V1',  value=0.0, format="%.4f"),
            'V2' : c2.number_input('V2',  value=0.0, format="%.4f"),
            'V3' : c3.number_input('V3',  value=0.0, format="%.4f"),
            'V4' : c4.number_input('V4',  value=0.0, format="%.4f"),
            'V14': c5.number_input('V14', value=0.0, format="%.4f"),
        }
        c6, c7 = st.columns(2)
        time_val   = c6.number_input('Time (seconds)', value=50000.0)
        amount_val = c7.number_input('Amount ($)',      value=100.0)

        transaction = {f'V{i}': 0.0 for i in range(1, 29)}
        transaction.update(v_vals)
        transaction['Time']   = time_val
        transaction['Amount'] = amount_val
        true_label = None

    # Score button
    st.markdown("")
    if st.button("🚀 Score Transaction", type="primary", use_container_width=True):
        with st.spinner("Scoring transaction..."):
            result = score_transaction(
                transaction, model, scaler_amount, scaler_time,
                feature_cols, threshold
            )

        # Result card
        prob  = result['probability']
        label = result['label']
        tier  = result['tier']
        color = result['color']

        css_class = {"fraud": "fraud-alert", "medium": "medium-alert", "legit": "legit-alert"}[color]
        st.markdown(f"""
        <div class="{css_class}">
            {tier} &nbsp;|&nbsp; Fraud Probability: {prob:.4f} &nbsp;|&nbsp;
            Decision: {label} &nbsp;|&nbsp; Threshold: {threshold}
        </div>
        """, unsafe_allow_html=True)

        # Gauge-style probability bar
        st.markdown("")
        col_p1, col_p2, col_p3 = st.columns([1, 3, 1])
        with col_p2:
            bar_color = "#F44336" if prob >= threshold else "#4CAF50"
            st.markdown(f"**Fraud Probability: {prob:.4f}**")
            st.progress(prob)

        if true_label is not None:
            correct = (true_label == 1 and label == "FRAUD") or \
                      (true_label == 0 and label == "LEGITIMATE")
            st.markdown(
                f"**Prediction: {'✅ Correct' if correct else '❌ Incorrect'}**  "
                f"(True label: {'FRAUD' if true_label == 1 else 'LEGITIMATE'})"
            )

        # SHAP explanation
        if show_shap:
            st.markdown('<div class="section-title">🧠 SHAP Explanation — Why this score?</div>',
                        unsafe_allow_html=True)
            with st.spinner("Computing SHAP values (~15 seconds)..."):
                explainer  = load_shap_explainer(model, tuple(feature_cols))
                input_arr  = result['input_df'].values
                shap_vals  = explainer.shap_values(input_arr, nsamples=100)[0]
                feat_vals  = pd.Series(result['input_df'].values[0], index=feature_cols)

            fig = shap_waterfall(shap_vals, feature_cols, feat_vals, prob)
            st.pyplot(fig)

            # Top 3 reasons in plain English
            sv = pd.Series(shap_vals, index=feature_cols)
            top3 = sv.abs().sort_values(ascending=False).head(3)
            st.markdown("**Top 3 reasons for this score:**")
            for feat, _ in top3.items():
                direction = "toward FRAUD 🔴" if sv[feat] > 0 else "toward LEGITIMATE 🟢"
                st.markdown(f"- **{feat}** = {feat_vals[feat]:.4f} → pushed {direction} "
                            f"(SHAP: {sv[feat]:+.4f})")


# ────────────────────────────────────────────────────────────
# TAB 2 — DATASET EXPLORER
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Class distribution
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df_raw['Class'].value_counts().sort_index()
        ax.bar(['Legitimate', 'Fraud'], counts.values,
               color=['#2196F3', '#F44336'], edgecolor='white')
        ax.set_title('Class Distribution', fontweight='bold')
        ax.set_ylabel('Count')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        # Amount distribution by class
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df_raw[df_raw['Class']==0]['Amount'].clip(upper=500),
                bins=60, alpha=0.6, color='#2196F3', density=True, label='Legitimate')
        ax.hist(df_raw[df_raw['Class']==1]['Amount'].clip(upper=500),
                bins=40, alpha=0.7, color='#F44336', density=True, label='Fraud')
        ax.set_xlabel('Amount ($, capped at $500)')
        ax.set_ylabel('Density')
        ax.set_title('Amount Distribution by Class', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Fraud by hour
    st.markdown('<div class="section-title">Fraud Patterns Over Time</div>', unsafe_allow_html=True)
    df_raw['Hour'] = (df_raw['Time'] // 3600) % 24
    fraud_by_hour  = df_raw[df_raw['Class']==1].groupby('Hour').size()
    legit_by_hour  = df_raw[df_raw['Class']==0].groupby('Hour').size()
    fraud_rate     = (fraud_by_hour / (fraud_by_hour + legit_by_hour) * 100).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].bar(fraud_by_hour.index, fraud_by_hour.values, color='#F44336', alpha=0.8)
    axes[0].set_title('Fraud Count by Hour of Day', fontweight='bold')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('Fraud Count')
    axes[0].set_xticks(range(0, 24))

    axes[1].bar(fraud_rate.index, fraud_rate.values, color='#FF7043', edgecolor='white')
    axes[1].axhline(y=df_raw['Class'].mean()*100, color='navy',
                    linestyle='--', lw=1.5, label='Overall avg')
    axes[1].set_title('Fraud Rate (%) by Hour of Day', fontweight='bold')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('Fraud Rate (%)')
    axes[1].set_xticks(range(0, 24))
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)


# ────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Three-Model Comparison</div>', unsafe_allow_html=True)

    results_data = {
        'Model'    : ['Isolation Forest', 'Autoencoder', 'Neural Network'],
        'Type'     : ['Unsupervised', 'Semi-supervised', 'Supervised'],
        'Precision': [0.1964, 0.5224, 0.7568],
        'Recall'   : [0.2973, 0.4730, 0.7568],
        'F1'       : [0.2366, 0.4965, 0.7568],
        'PR-AUC'   : [0.1064, 0.5005, 0.7630],
        'ROC-AUC'  : [0.9450, 0.9561, 0.9800],
    }
    df_results = pd.DataFrame(results_data)

    # Styled table
    st.dataframe(
        df_results.set_index('Model').style
        .background_gradient(cmap='Blues', subset=['Precision','Recall','F1','PR-AUC','ROC-AUC'])
        .format({'Precision':'{:.4f}','Recall':'{:.4f}','F1':'{:.4f}',
                 'PR-AUC':'{:.4f}','ROC-AUC':'{:.4f}'}),
        use_container_width=True
    )

    # Bar chart comparison
    metrics = ['Precision', 'Recall', 'F1', 'PR-AUC']
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ['#FF9800', '#7B1FA2', '#1565C0']
    for i, (_, row) in enumerate(df_results.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (i-1)*width, vals, width,
                      label=row['Model'], color=colors[i],
                      alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f'{v:.2f}', ha='center', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — All Metrics', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Random vs Time split
    st.markdown('<div class="section-title">Random Split vs Time-Based Split</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Random Split PR-AUC",    "0.7630", help="Optimistic — model saw future data")
    col2.metric("Time-Based PR-AUC",      "0.6488", delta="-0.1142",
                help="Honest production estimate")
    col3.metric("Gap",                    "0.1142",
                help="Due to concept drift — fraud patterns shift over time")

    st.info("""
    **Why the gap matters:** The time-based split is the number you should report to stakeholders.
    Random splits overestimate performance because the model has implicitly seen future fraud patterns.
    In production, you always train on the past and predict on the future.
    """)
