from sklearn.metrics import average_precision_score

from fraud_monitor.baselines import fit_baselines
from fraud_monitor.features import add_stateless_features
from tests.factories import make_ieee_cis_tables


def test_baselines_fit_and_score_unseen_future_categories() -> None:
    tables = make_ieee_cis_tables(rows=240)
    features = add_stateless_features(tables.train_transaction)
    development = features.iloc[:160].copy()
    validation = features.iloc[160:].copy()
    validation.loc[validation.index[0], "ProductCD"] = "UNSEEN"

    models = fit_baselines(
        development,
        tables.train_transaction["isFraud"].iloc[:160].to_numpy(),
    )
    predictions = models.predict_probabilities(validation)

    assert set(predictions) == {"dummy", "logistic"}
    assert all(len(values) == len(validation) for values in predictions.values())
    assert all(((values >= 0) & (values <= 1)).all() for values in predictions.values())
    assert average_precision_score(
        tables.train_transaction["isFraud"].iloc[160:], predictions["logistic"]
    ) >= 0.0

