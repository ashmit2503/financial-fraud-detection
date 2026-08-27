"""Deterministic IEEE-CIS-like tables used by tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticTables:
    train_transaction: pd.DataFrame
    train_identity: pd.DataFrame
    test_transaction: pd.DataFrame
    test_identity: pd.DataFrame

    def write_csvs(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.train_transaction.to_csv(directory / "train_transaction.csv", index=False)
        self.train_identity.to_csv(directory / "train_identity.csv", index=False)
        self.test_transaction.to_csv(directory / "test_transaction.csv", index=False)
        self.test_identity.to_csv(directory / "test_identity.csv", index=False)


def make_ieee_cis_tables(rows: int = 240, seed: int = 42) -> SyntheticTables:
    """Return small chronologically ordered tables with controlled fraud signal."""

    if rows < 40:
        raise ValueError("At least 40 rows are required for temporal test partitions.")

    rng = np.random.default_rng(seed)
    train_id = np.arange(2_987_000, 2_987_000 + rows)
    transaction_dt = 86_400 + np.arange(rows) * 21_600
    product = rng.choice(["W", "H", "C", "S", "R"], size=rows, p=[0.55, 0.15, 0.15, 0.1, 0.05])
    amount = rng.lognormal(mean=4.2, sigma=0.7, size=rows).round(2)
    card1 = rng.integers(1000, 1030, size=rows)
    addr1 = rng.choice([100, 200, 300, np.nan], size=rows, p=[0.35, 0.3, 0.25, 0.1])
    fraud_logit = (
        -3.6
        + 1.4 * (product == "C")
        + 0.8 * (amount > 150)
        + 0.5 * (transaction_dt > np.quantile(transaction_dt, 0.7))
    )
    fraud_probability = 1.0 / (1.0 + np.exp(-fraud_logit))
    target = rng.binomial(1, fraud_probability)

    train_transaction = pd.DataFrame(
        {
            "TransactionID": train_id,
            "isFraud": target,
            "TransactionDT": transaction_dt,
            "TransactionAmt": amount,
            "ProductCD": product,
            "card1": card1,
            "card2": rng.choice([100, 200, 300, np.nan], size=rows),
            "card3": 150,
            "card4": rng.choice(["visa", "mastercard", np.nan], size=rows),
            "card5": rng.choice([100, 200, np.nan], size=rows),
            "card6": rng.choice(["credit", "debit"], size=rows),
            "addr1": addr1,
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], size=rows),
            "C1": rng.poisson(2.0, size=rows),
            "D1": rng.choice([1.0, 5.0, np.nan], size=rows),
            "M1": rng.choice(["T", "F", np.nan], size=rows),
            "V1": rng.normal(size=rows),
        }
    )
    identity_mask = rng.random(rows) < 0.45
    train_identity = pd.DataFrame(
        {
            "TransactionID": train_id[identity_mask],
            "DeviceType": rng.choice(["desktop", "mobile"], size=identity_mask.sum()),
            "DeviceInfo": rng.choice(
                ["Windows", "iOS Device", "SM-G960"], size=identity_mask.sum()
            ),
            "id_12": rng.choice(["Found", "NotFound"], size=identity_mask.sum()),
        }
    )

    test_rows = max(40, rows // 3)
    test_id = np.arange(3_660_000, 3_660_000 + test_rows)
    test_transaction = train_transaction.drop(columns="isFraud").iloc[:test_rows].copy()
    test_transaction["TransactionID"] = test_id
    test_transaction["TransactionDT"] = (
        train_transaction["TransactionDT"].max() + 86_400 + np.arange(test_rows) * 21_600
    )
    test_identity = train_identity.drop(columns="TransactionID").iloc[: test_rows // 2].copy()
    test_identity.insert(0, "TransactionID", test_id[: len(test_identity)])
    test_identity = test_identity.rename(columns={"id_12": "id-12"})

    return SyntheticTables(
        train_transaction=train_transaction,
        train_identity=train_identity,
        test_transaction=test_transaction,
        test_identity=test_identity,
    )
