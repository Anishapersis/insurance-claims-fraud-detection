"""Ablation: how much of the model's skill depends on the suspicious hobby feature?"""
import sys
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fraud.data as D  # noqa: E402
import fraud.train as T  # noqa: E402

X, y = D.load_dataset()
cv = StratifiedKFold(5, shuffle=True, random_state=T.SEED)
pw = float((y == 0).sum() / (y == 1).sum())
for label, drop in [("all features", []), ("without insured_hobbies", ["insured_hobbies"])]:
    cats = [c for c in D.CATEGORICAL if c not in drop]
    T.CATEGORICAL[:] = cats  # preprocessor reads this list
    pipe = T.candidates(pw)["xgboost"]
    p = cross_val_predict(pipe, X.drop(columns=drop), y, cv=cv, method="predict_proba")[:, 1]
    print(f"{label:26s} PR-AUC {average_precision_score(y, p):.3f}  ROC-AUC {roc_auc_score(y, p):.3f}")
