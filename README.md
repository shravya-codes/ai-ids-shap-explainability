# AI-Powered IDS with SHAP Explainability

An intrusion detection system that doesn't just say "attack detected" — it tells you *why*.

Most IDS tools are black boxes: a flow comes in, a label comes out, and a SOC analyst is left guessing which features triggered the alert. This project pairs a high-accuracy XGBoost classifier with SHAP explainability so every single prediction comes with a feature-level breakdown, plus an autoencoder layer for catching anomalies the classifier has never seen before.

## Live Demo

**[ai-ids-shap-explainability.streamlit.app](https://ai-ids-shap-explainability.streamlit.app/)**

## Why this project

- **Explainability first.** Every alert includes a SHAP waterfall plot showing exactly which features pushed the model toward its decision — not just a confidence score.
- **Two-layer architecture.** A supervised XGBoost classifier catches known attack signatures. An unsupervised autoencoder, trained only on benign traffic, flags statistical anomalies — including attack types the classifier was never trained on.
- **Honest reporting.** Rather than chasing a single accuracy number, this project documents where each layer succeeds and where it doesn't — which is what a real security engineering writeup should do.

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity) — labeled network flow data covering benign traffic and multiple real attack types (DoS, DDoS, PortScan, Brute Force, Web Attacks, Bot, and rare classes like Infiltration and Heartbleed).

The raw 15 labels were grouped into 7 categories (BENIGN, DoS, DDoS, PortScan, Brute Force, Web Attack, Bot) to produce statistically meaningful class sizes. Two classes (Infiltration, Heartbleed) had fewer than 50 samples total and were excluded from training as a deliberate, documented choice rather than force-fit into an unreliable train/test split.

## Architecture

```
Raw traffic flow
      │
      ├──► XGBoost classifier ──► Attack category + confidence
      │           │
      │           └──► SHAP TreeExplainer ──► Per-flow feature explanation
      │
      └──► Autoencoder (trained on benign only) ──► Reconstruction error
                  │
                  └──► Anomaly flag (threshold = 95th percentile of benign error)
```

## Results

**XGBoost classifier** — 99.88% accuracy, weighted F1 ≈ 1.00 across all 7 categories on held-out test data. Class imbalance handled via balanced sample weighting so minority classes (Bot, Web Attack) aren't ignored in favor of majority BENIGN traffic.

**SHAP explainability** — Global feature importance highlights `Destination Port`, `Init_Win_bytes_backward/forward`, and packet timing features as the strongest predictors, confirming the model leans on genuine behavioral signal rather than a single shortcut feature. Per-flow waterfall plots let you trace exactly why any individual flow was flagged.

**Autoencoder anomaly detection** — Trained exclusively on benign traffic. At a 95th-percentile threshold, it flags:
- DoS: 68.6% of instances
- DDoS: 62.6% of instances
- Bot / PortScan / Brute Force / Web Attack: under 5% (comparable to the benign false-positive rate)

This is expected, not a flaw: volumetric attacks (DoS/DDoS) produce statistically extreme traffic that reconstructs poorly, while low-and-slow attacks (port scans, brute force) look similar to normal traffic at the level of a single flow. This finding is the core argument for the two-layer design — signature-based detection for known attacks, anomaly-based detection for statistically extreme or novel ones.

## Tech stack

| Layer | Tools |
|---|---|
| Classification | XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| Anomaly detection | Autoencoder (TensorFlow / Keras) |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |

## Running locally

```bash
git clone https://github.com/shravya-codes/ai-ids-shap-explainability.git
cd ai-ids-shap-explainability
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.11 (TensorFlow does not yet support 3.14+).

## Project structure

```
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── xgboost_model.pkl            # Trained classifier
├── label_encoder.pkl
├── scaler.pkl                   # Feature scaler for autoencoder input
├── autoencoder.keras            # Trained anomaly detector
├── threshold.json               # Anomaly threshold (95th percentile of benign error)
├── feature_columns.json
└── demo_traffic_sample.csv      # Sample flows for the live dashboard demo
```

## Author

Shravya — [GitHub](https://github.com/shravya-codes) · [LinkedIn](https://linkedin.com/in/shravya-none-22093b397)

Part of a broader cybersecurity ML portfolio, alongside [PhishRadar](https://github.com/shravya-codes/PhishRadar) (phishing URL detection) and NetWatchAI (network intrusion detection).
