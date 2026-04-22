
from app.predictor import normalize_features, prepare_dataframe


def test_normalize_features_converts_age_years():
    features = {"AGE_YEARS": 40, "AMT_CREDIT": 100000.0}
    result = normalize_features(features)

    assert "AGE_YEARS" not in result
    assert result["DAYS_BIRTH"] == -14600
    assert result["AMT_CREDIT"] == 100000.0


def test_prepare_dataframe_keeps_all_default_columns():
    df = prepare_dataframe({"AGE_YEARS": 35, "CODE_GENDER": "F"})

    assert df.shape[0] == 1
    assert "CODE_GENDER" in df.columns
    assert "DAYS_BIRTH" in df.columns
    assert "AMT_CREDIT" in df.columns
