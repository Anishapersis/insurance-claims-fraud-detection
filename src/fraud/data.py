"""Data loading, cleaning and feature engineering for the claims fraud model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "insurance_claims.csv"
DATA_URL = "https://raw.githubusercontent.com/mwitiderrick/insurancedata/master/insurance_claims.csv"
TARGET = "fraud_reported"

# Identifiers and free-text fields carry no generalisable signal (or leak row identity).
DROP_COLS = ["policy_number", "insured_zip", "incident_location", "_c39"]

CATEGORICAL = [
    "policy_state", "policy_csl", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "insured_relationship",
    "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "incident_state", "incident_city",
    "property_damage", "police_report_available", "auto_make",
]

NUMERIC = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "auto_year",
    # engineered
    "policy_age_days", "vehicle_age", "injury_share", "property_share",
    "vehicle_share", "claim_to_premium", "incident_weekday",
]


def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV (downloading the public dataset on first run) and normalise the
    '?' placeholder to a proper missing value."""
    path = Path(path)
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path)
    return df.replace("?", np.nan)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features an investigator would reason about."""
    out = df.copy()
    bind = pd.to_datetime(out["policy_bind_date"], errors="coerce")
    incident = pd.to_datetime(out["incident_date"], errors="coerce")

    out["policy_age_days"] = (incident - bind).dt.days
    out["vehicle_age"] = incident.dt.year - out["auto_year"]
    out["incident_weekday"] = incident.dt.weekday

    total = out["total_claim_amount"].replace(0, np.nan)
    out["injury_share"] = out["injury_claim"] / total
    out["property_share"] = out["property_claim"] / total
    out["vehicle_share"] = out["vehicle_claim"] / total
    out["claim_to_premium"] = out["total_claim_amount"] / out["policy_annual_premium"]

    # Missing categorical answers are informative ("unknown" police report etc.).
    for col in CATEGORICAL:
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    return out


def load_dataset(path: str | Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return model-ready features X and binary target y (1 = fraud)."""
    df = engineer(load_raw(path))
    y = (df[TARGET] == "Y").astype(int)
    X = df[CATEGORICAL + NUMERIC]
    return X, y
