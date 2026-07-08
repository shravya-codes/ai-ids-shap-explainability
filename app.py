import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI-Powered IDS with SHAP Explainability", layout="wide")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgboost_model.pkl")
    le = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    autoencoder = load_model("autoencoder.keras")
    with open("threshold.json") as f:
        threshold = json.load(f)["threshold"]
    with open("feature_columns.json") as f:
        feature_columns = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, le, scaler, autoencoder, threshold, feature_columns, explainer

@st.cache_data
def load_demo_data():
    return pd.read_csv("demo_traffic_sample.csv")

model, le, scaler, autoencoder, threshold, feature_columns, explainer = load_artifacts()
demo_df = load_demo_data()

# ---------- Header ----------
st.title("🛡️ AI-Powered Intrusion Detection System")
st.caption("XGBoost classification + SHAP explainability + Autoencoder anomaly detection — every alert comes with a *why*.")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Model Accuracy", "99.88%")
col_b.metric("Attack Categories Detected", "6")
col_c.metric("Anomaly Detection Layer", "Autoencoder")

st.divider()

# ---------- Sidebar: pick a traffic flow ----------
st.sidebar.header("Select Traffic Flow")
mode = st.sidebar.radio("Source", ["Random sample from dataset", "Pick by row index"])

if mode == "Random sample from dataset":
    if st.sidebar.button("🎲 Pull a new flow"):
        st.session_state.row_idx = np.random.randint(0, len(demo_df))
    if "row_idx" not in st.session_state:
        st.session_state.row_idx = 0
    row_idx = st.session_state.row_idx
else:
    row_idx = st.sidebar.number_input("Row index", 0, len(demo_df) - 1, 0)

row = demo_df.iloc[[row_idx]]
true_label = row["Category"].values[0] if "Category" in row.columns else "Unknown"
X_row = row[feature_columns]

# ---------- Run predictions ----------
pred_class_idx = model.predict(X_row)[0]
pred_proba = model.predict_proba(X_row)[0]
pred_label = le.classes_[pred_class_idx]

X_row_scaled = scaler.transform(X_row)
reconstruction = autoencoder.predict(X_row_scaled, verbose=0)
recon_error = float(np.mean(np.square(X_row_scaled - reconstruction)))
is_anomaly = recon_error > threshold

# ---------- Alert Panel ----------
st.subheader(f"Flow #{row_idx} — Alert Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("True Label", true_label)
label_color = "🔴" if pred_label != "BENIGN" else "🟢"
c2.metric("XGBoost Prediction", f"{label_color} {pred_label}")
c3.metric("Confidence", f"{pred_proba[pred_class_idx]*100:.1f}%")
anomaly_flag = "⚠️ ANOMALY" if is_anomaly else "✅ Normal"
c4.metric("Autoencoder Verdict", anomaly_flag, f"error: {recon_error:.4f}")

if pred_label != "BENIGN" or is_anomaly:
    st.warning(f"**Alert triggered.** XGBoost classified this flow as **{pred_label}**"
               + (f" and the autoencoder flagged it as statistically anomalous (reconstruction error {recon_error:.4f} > threshold {threshold:.4f})." if is_anomaly else "."))
else:
    st.success("No alert. Flow classified as BENIGN and reconstruction error within normal range.")

st.divider()

# ---------- SHAP Explanation ----------
st.subheader("🔍 Why did the model decide this? (SHAP Explanation)")

shap_values = explainer.shap_values(X_row)
# shap_values shape: (1, n_features, n_classes)

fig, ax = plt.subplots(figsize=(9, 6))
exp = shap.Explanation(
    values=shap_values[0, :, pred_class_idx],
    base_values=explainer.expected_value[pred_class_idx],
    data=X_row.iloc[0],
    feature_names=feature_columns
)
shap.plots.waterfall(exp, show=False, max_display=10)
st.pyplot(fig, clear_figure=True)

st.caption("Red bars push the prediction toward the predicted class; blue bars push away from it. "
           "This is the per-flow explanation a SOC analyst would use to justify or dismiss the alert.")

st.divider()

# ---------- Global feature importance ----------
with st.expander("📊 Global Feature Importance (across all traffic)"):
    st.write("Which features matter most across the whole dataset, on average, for attack classes:")
    sample_for_global = demo_df[feature_columns].sample(min(200, len(demo_df)), random_state=1)
    global_shap = explainer.shap_values(sample_for_global)
    global_importance = np.abs(global_shap[:, :, 1:]).mean(axis=(0, 2))
    imp_df = pd.DataFrame({"Feature": feature_columns, "Importance": global_importance}) \
                .sort_values("Importance", ascending=False).head(15)
    st.bar_chart(imp_df.set_index("Feature"))

st.divider()

# ---------- Reconstruction error distribution ----------
with st.expander("📈 Autoencoder Reconstruction Error Context"):
    st.write("Reconstruction error for this flow compared to typical benign traffic.")
    benign_sample = demo_df[demo_df["Category"] == "BENIGN"][feature_columns].sample(
        min(200, len(demo_df[demo_df["Category"] == "BENIGN"])), random_state=1)
    benign_scaled = scaler.transform(benign_sample)
    benign_recon = autoencoder.predict(benign_scaled, verbose=0)
    benign_errors = np.mean(np.square(benign_scaled - benign_recon), axis=1)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.hist(np.clip(benign_errors, 0, 2), bins=30, alpha=0.6, label="Benign traffic (sample)")
    ax2.axvline(threshold, color="red", linestyle="--", label="Anomaly threshold")
    ax2.axvline(min(recon_error, 2), color="black", linewidth=2, label="This flow")
    ax2.set_xlabel("Reconstruction Error")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

st.divider()
st.caption("Built by Shravya — XGBoost + SHAP + Autoencoder two-layer intrusion detection system, trained on CICIDS2017.")
