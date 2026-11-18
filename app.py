"""Streamlit triage app: score a claim and explain the score.

Run:  streamlit run app.py   (after `python src/fraud/train.py`)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fraud.data import CATEGORICAL, NUMERIC, load_dataset  # noqa: E402

st.set_page_config(page_title="Claims Fraud Triage", layout="wide")
bundle = joblib.load(Path(__file__).parent / "models" / "fraud_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]
X, y = load_dataset()

st.title("Claims fraud triage")
st.caption(f"Model: {type(model.named_steps['clf']).__name__} · alert threshold {threshold:.2f} "
           "(chosen to minimise expected investigation + leakage cost)")

idx = st.number_input("Pick a claim from the dataset (row index)", 0, len(X) - 1, 0)
claim = X.iloc[[idx]].copy()

with st.expander("Edit claim details"):
    c1, c2, c3 = st.columns(3)
    claim["incident_severity"] = c1.selectbox(
        "Incident severity", sorted(X["incident_severity"].unique()),
        index=sorted(X["incident_severity"].unique()).index(claim["incident_severity"].iat[0]))
    claim["total_claim_amount"] = c2.number_input("Total claim amount", value=float(claim["total_claim_amount"].iat[0]))
    claim["witnesses"] = c3.number_input("Witnesses", 0, 5, int(claim["witnesses"].iat[0]))
    claim["police_report_available"] = c1.selectbox("Police report", ["YES", "NO", "UNKNOWN"],
        index=["YES", "NO", "UNKNOWN"].index(claim["police_report_available"].iat[0]))

score = float(model.predict_proba(claim)[0, 1])
flag = score >= threshold
m1, m2, m3 = st.columns(3)
m1.metric("Fraud score", f"{score:.2f}")
m2.metric("Decision", "Refer to SIU" if flag else "Fast-track")
m3.metric("Actual label", "Fraud" if y.iat[idx] else "Legit")

prep, clf = model.named_steps["prep"], model.named_steps["clf"]
xt = prep.transform(claim)
xt = xt.toarray() if hasattr(xt, "toarray") else xt
names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
sv = shap.TreeExplainer(clf)(xt)
contrib = pd.Series(sv.values[0], index=names)
top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)

st.subheader("Why this score")
st.bar_chart(top.rename("SHAP contribution (log-odds)"))
st.caption("Positive bars push the claim towards fraud; negative bars push it towards legitimate.")
