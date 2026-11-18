"""Train, compare and explain fraud models.

Usage:
    python -m fraud.train            # from the src/ directory, or
    python src/fraud/train.py

Writes the chosen model to models/, metrics to reports/metrics.json and
figures to reports/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraud.data import CATEGORICAL, NUMERIC, load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
SEED = 42

# Business assumption used to choose the alert threshold:
# a missed fraud costs far more than a false alarm sent to investigation.
COST_FALSE_NEGATIVE = 10_000  # average fraudulent payout leaked (CAD)
COST_FALSE_POSITIVE = 500     # investigator time for a legitimate claim (CAD)


def preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL),
        ("num", Pipeline(num_steps), NUMERIC),
    ])


def candidates(pos_weight: float) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                           class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("prep", preprocessor(scale=False)),
            ("clf", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=pos_weight, eval_metric="aucpr",
                                  random_state=SEED, n_jobs=2)),
        ]),
    }


def best_cost_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in grid:
        pred = p >= t
        fn = int(((~pred) & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        costs.append(fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE)
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def main() -> None:
    for d in (MODELS, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1) Compare models with 5-fold CV on the training set (PR-AUC is the headline metric
    #    because only ~25% of claims are fraud and alerts are what investigators act on).
    results, oof = {}, {}
    for name, pipe in candidates(pos_weight).items():
        p = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof[name] = p
        results[name] = {"cv_pr_auc": round(average_precision_score(y_tr, p), 4),
                         "cv_roc_auc": round(roc_auc_score(y_tr, p), 4)}
        print(f"{name:22s} PR-AUC {results[name]['cv_pr_auc']:.3f}  ROC-AUC {results[name]['cv_roc_auc']:.3f}")

    best = max(results, key=lambda k: results[k]["cv_pr_auc"])
    threshold, _ = best_cost_threshold(y_tr.to_numpy(), oof[best])
    print(f"Selected: {best}; cost-optimal threshold (from CV) = {threshold:.2f}")

    # 2) Refit on all training data, evaluate once on the untouched test set.
    model = candidates(pos_weight)[best].fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    pred = (p_te >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    base_cost = int(y_te.sum()) * COST_FALSE_NEGATIVE  # no model: every fraud is paid
    model_cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    test = {
        "model": best, "threshold": round(threshold, 2),
        "test_pr_auc": round(average_precision_score(y_te, p_te), 4),
        "test_roc_auc": round(roc_auc_score(y_te, p_te), 4),
        "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_claims": int(len(y_te)), "alerts_raised": int(tp + fp),
        "estimated_cost_no_model_cad": base_cost, "estimated_cost_with_model_cad": int(model_cost),
        "estimated_saving_pct": round(100 * (1 - model_cost / base_cost), 1),
    }
    print(json.dumps(test, indent=2))

    joblib.dump({"model": model, "threshold": threshold}, MODELS / "fraud_model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps({"cv": results, "test": test}, indent=2))

    # 3) Figures: PR curves, confusion matrix, SHAP global importance.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, p in oof.items():
        pr, rc, _ = precision_recall_curve(y_tr, p)
        ax.plot(rc, pr, label=f"{name} (AP={results[name]['cv_pr_auc']:.2f})")
    ax.axhline(y_tr.mean(), ls="--", c="grey", lw=1, label="random baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall, 5-fold CV")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "pr_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
           xlabel="Predicted", ylabel="Actual", title=f"Test set @ threshold {threshold:.2f}")
    fig.tight_layout(); fig.savefig(FIGS / "confusion_matrix.png", dpi=150); plt.close(fig)

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    X_te_t = prep.transform(X_te)
    X_te_t = X_te_t.toarray() if hasattr(X_te_t, "toarray") else X_te_t
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    explainer = shap.Explainer(clf, X_te_t, feature_names=names) if best == "logistic_regression" \
        else shap.TreeExplainer(clf, feature_names=names)
    sv = explainer(X_te_t)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("What drives the fraud score (SHAP, test set)")
    plt.tight_layout(); plt.savefig(FIGS / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    imp = pd.Series(np.abs(sv.values).mean(0), index=names).sort_values(ascending=False).head(10)
    (REPORTS / "top_features.json").write_text(imp.round(4).to_json(indent=2))
    print(imp)


if __name__ == "__main__":
    main()
