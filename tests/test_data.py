import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud.data import CATEGORICAL, NUMERIC, load_dataset, load_raw  # noqa: E402


def test_shapes_and_target():
    X, y = load_dataset()
    assert len(X) == len(y) == 1000
    assert set(y.unique()) == {0, 1}
    assert list(X.columns) == CATEGORICAL + NUMERIC


def test_no_placeholders_left():
    X, _ = load_dataset()
    assert not (X[CATEGORICAL] == "?").any().any()


def test_identifiers_excluded():
    X, _ = load_dataset()
    for col in ("policy_number", "insured_zip", "incident_location"):
        assert col not in X.columns


def test_claim_shares_sum_to_one():
    X, _ = load_dataset()
    total = X["injury_share"] + X["property_share"] + X["vehicle_share"]
    assert ((total - 1).abs() < 1e-6).all()


def test_raw_has_question_marks_replaced():
    assert load_raw()["collision_type"].isna().sum() > 0
