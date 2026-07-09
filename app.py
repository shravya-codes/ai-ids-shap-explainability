import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import shap
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI-Powered IDS | SHAP Explainability", page_icon="🛡️", layout="wide")

# ---------- Custom theme (red accent, dark, personality-forward) ----------
st.markdown("""
<style>
    .stApp { background-color: #0e0e10; }
    h1, h2, h3 { color: #f2f2f2 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a1d 0%, #221214 100%);
        border: 1px solid #3a1518;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: left;
    }
    .metric-label { font-size: 13px; color: #b0aeae; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
    .accent { color: #ff3b3b; }
    .alert-box-danger {
        background: linear-gradient(135deg, #2b0f10 0%, #3a1214 100%);
        border-left: 4px solid #ff3b3b;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffb3b3;
        font-size: 15px;
    }
    .alert-box-safe {
        background: linear-gradient(135deg, #0f2b16 0%, #123a1c 100%);
        border-left: 4px solid #2ecc71;
        border-radius: 10px;
        padding: 16px 20px;
        color: #b3ffcc;
        font-size: 15px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .badge-red { background: #3a1518; color: #ff5c5c; }
    .badge-green { background: #123a1c; color: #4ee888; }
    div[data-testid="stSidebar"] { background-color: #141416; }
</style>
""", unsafe_allow_html=True)


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

CLASS_COLORS = {
    "BENIGN": "#2ecc71", "Bot": "#ff9f43", "Brute Force": "#ff6b6b",
    "DDoS": "#ff3b3b", "DoS": "#e84118", "PortScan": "#feca57", "Web Attack": "#9b59b6"
}

# ---------- Header ----------
st.markdown("""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
    <div style="font-size:38px;">🛡️</div>
    <div>
        <h1 style="margin:0; font-size:32px;">AI-Powered Intrusion Detection <span class="accent">System</span></h1>
        <p style="color:#9a9a9a; margin:2px 0 0; font-size:14px;">XGBoost classification + SHAP explainability + Autoencoder anomaly detection — every alert comes with a <i>why</i>.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("Model accuracy", "99.88%"),
    ("Attack categories", "6"),
    ("Test flows analyzed", f"{len(demo_df):,}"),
    ("Anomaly layer", "Autoencoder"),
]
for col, (label, val) in zip([c1, c2, c3, c4], metrics):
    col.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
                  <div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)

st.write("")
st.markdown("---")

# ---------- Sidebar ----------
st.sidebar.markdown("### 🎛️ Traffic control panel")
mode = st.sidebar.radio("Source", ["Random sample", "Pick by row index", "Simulate live feed"])

if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0
if "feed_log" not in st.session_state:
    st.session_state.feed_log = []

if mode == "Random sample":
    if st.sidebar.button("🎲 Pull a new flow", use_container_width=True):
        st.session_state.row_idx = int(np.random.randint(0, len(demo_df)))
    row_idx = st.session_state.row_idx

elif mode == "Pick by row index":
    row_idx = st.sidebar.number_input("Row index", 0, len(demo_df) - 1, 0)
    st.session_state.row_idx = row_idx

else:
    st.sidebar.write("Streams a new flow every couple seconds.")
    run_feed = st.sidebar.toggle("▶ Start live feed", value=False)
    row_idx = st.session_state.row_idx
    if run_feed:
        st.session_state.row_idx = int(np.random.randint(0, len(demo_df)))
        row_idx = st.session_state.row_idx

st.sidebar.markdown("---")
st.sidebar.caption("Recent alerts")
for entry in st.session_state.feed_log[-6:][::-1]:
    color = "🔴" if entry["label"] != "BENIGN" else "🟢"
    st.sidebar.markdown(f"{color} `#{entry['idx']}` **{entry['label']}** — {entry['conf']:.0f}%")

# ---------- Run predictions ----------
row = demo_df.iloc[[row_idx]]
true_label = row["Category"].values[0] if "Category" in row.columns else "Unknown"
X_row = row[feature_columns]

pred_class_idx = model.predict(X_row)[0]
pred_proba = model.predict_proba(X_row)[0]
pred_label = le.classes_[pred_class_idx]

X_row_scaled = scaler.transform(X_row)
reconstruction = autoencoder.predict(X_row_scaled, verbose=0)
recon_error = float(np.mean(np.square(X_row_scaled - reconstruction)))
is_anomaly = recon_error > threshold

st.session_state.feed_log.append({"idx": row_idx, "label": pred_label, "conf": pred_proba[pred_class_idx]*100})

# ---------- Alert Panel ----------
st.markdown(f"### Flow `#{row_idx}` — live alert summary")

a1, a2, a3, a4 = st.columns(4)
a1.markdown(f"""<div class="metric-card"><div class="metric-label">True label</div>
            <div class="metric-value" style="font-size:20px;">{true_label}</div></div>""", unsafe_allow_html=True)

badge_class = "badge-red" if pred_label != "BENIGN" else "badge-green"
a2.markdown(f"""<div class="metric-card"><div class="metric-label">XGBoost prediction</div>
            <span class="badge {badge_class}">{pred_label}</span></div>""", unsafe_allow_html=True)

a3.markdown(f"""<div class="metric-card"><div class="metric-label">Confidence</div>
            <div class="metric-value" style="font-size:20px;">{pred_proba[pred_class_idx]*100:.1f}%</div></div>""", unsafe_allow_html=True)

anomaly_badge = "badge-red" if is_anomaly else "badge-green"
anomaly_text = "⚠ Anomaly" if is_anomaly else "✓ Normal"
a4.markdown(f"""<div class="metric-card"><div class="metric-label">Autoencoder verdict</div>
            <span class="badge {anomaly_badge}">{anomaly_text}</span></div>""", unsafe_allow_html=True)

st.write("")
if pred_label != "BENIGN" or is_anomaly:
    st.markdown(f"""<div class="alert-box-danger">🚨 <b>Alert triggered.</b> XGBoost classified this flow as <b>{pred_label}</b>
                {"and the autoencoder flagged it as statistically anomalous (reconstruction error " + f"{recon_error:.4f} > threshold {threshold:.4f})." if is_anomaly else "."}
                </div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="alert-box-safe">✅ No alert. Flow classified as BENIGN and reconstruction error within normal range.</div>""", unsafe_allow_html=True)

st.write("")
st.markdown("---")

# ---------- Confidence breakdown + gauge ----------
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.markdown("#### 📊 Class probability breakdown")
    proba_df = pd.DataFrame({"Category": le.classes_, "Probability": pred_proba * 100})
    proba_df = proba_df.sort_values("Probability", ascending=True)
    colors = [CLASS_COLORS.get(c, "#888") for c in proba_df["Category"]]
    fig_bar = go.Figure(go.Bar(
        x=proba_df["Probability"], y=proba_df["Category"], orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in proba_df["Probability"]],
        textposition="outside"
    ))
    fig_bar.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", xaxis=dict(range=[0, 105], showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown("#### 🎯 Reconstruction error gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=recon_error,
        number={"suffix": "", "font": {"color": "#ffffff"}},
        gauge={
            "axis": {"range": [0, max(threshold * 3, recon_error * 1.2, 0.5)], "tickcolor": "#888"},
            "bar": {"color": "#ff3b3b" if is_anomaly else "#2ecc71"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, threshold], "color": "#123a1c"},
                {"range": [threshold, max(threshold * 3, recon_error * 1.2, 0.5)], "color": "#3a1518"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": threshold}
        }
    ))
    fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=10),
                             paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0")
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ---------- SHAP Explanation ----------
st.markdown("#### 🔍 Why did the model decide this? (SHAP explanation)")

shap_values = explainer.shap_values(X_row)
sv = shap_values[0, :, pred_class_idx]
base_val = explainer.expected_value[pred_class_idx]

order = np.argsort(np.abs(sv))[::-1][:10]
shap_df = pd.DataFrame({
    "Feature": [feature_columns[i] for i in order],
    "Value": [X_row.iloc[0, i] for i in order],
    "SHAP": [sv[i] for i in order]
})
shap_df["Label"] = shap_df["Feature"] + " = " + shap_df["Value"].round(2).astype(str)
shap_df = shap_df.sort_values("SHAP")

fig_shap = go.Figure(go.Bar(
    x=shap_df["SHAP"], y=shap_df["Label"], orientation="h",
    marker_color=["#ff3b3b" if v > 0 else "#378ADD" for v in shap_df["SHAP"]],
    text=[f"{v:+.2f}" for v in shap_df["SHAP"]], textposition="outside"
))
fig_shap.update_layout(
    height=420, margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e0e0e0", xaxis=dict(showgrid=True, gridcolor="#2a2a2a", zeroline=True, zerolinecolor="#555"),
    yaxis=dict(showgrid=False)
)
st.plotly_chart(fig_shap, use_container_width=True)
st.caption("Red bars push the prediction toward the predicted class, blue bars push away from it — this is the per-flow explanation a SOC analyst uses to justify or dismiss the alert.")

st.markdown("---")

# ---------- Global importance + reconstruction context ----------
tab1, tab2 = st.tabs(["📈 Global feature importance", "📉 Reconstruction error landscape"])

with tab1:
    sample_for_global = demo_df[feature_columns].sample(min(200, len(demo_df)), random_state=1)
    global_shap = explainer.shap_values(sample_for_global)
    global_importance = np.abs(global_shap[:, :, 1:]).mean(axis=(0, 2))
    imp_df = pd.DataFrame({"Feature": feature_columns, "Importance": global_importance}) \
                .sort_values("Importance", ascending=True).tail(15)
    fig_global = go.Figure(go.Bar(
        x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
        marker_color="#ff6b6b"
    ))
    fig_global.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0"
    )
    st.plotly_chart(fig_global, use_container_width=True)

with tab2:
    benign_mask = demo_df["Category"] == "BENIGN"
    sample_benign = demo_df[benign_mask][feature_columns].sample(min(300, benign_mask.sum()), random_state=1)
    benign_scaled = scaler.transform(sample_benign)
    benign_recon = autoencoder.predict(benign_scaled, verbose=0)
    benign_errors = np.mean(np.square(benign_scaled - benign_recon), axis=1)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=np.clip(benign_errors, 0, 2), nbinsx=40, marker_color="#2ecc71",
                                     opacity=0.7, name="Benign traffic"))
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#ff3b3b",
                        annotation_text="threshold", annotation_font_color="#ff3b3b")
    fig_hist.add_vline(x=min(recon_error, 2), line_color="white", line_width=3,
                        annotation_text="this flow", annotation_font_color="white")
    fig_hist.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.caption("Built by Shravya — XGBoost + SHAP + Autoencoder two-layer intrusion detection system, trained on CICIDS2017.")

if mode == "Simulate live feed" and run_feed:
    time.sleep(2)
    st.rerun()
