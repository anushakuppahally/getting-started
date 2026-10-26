
import inspect
import json
from pathlib import Path

import joblib
from datasets import load_dataset
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from aeon.transformations.collection.convolution_based import MiniRocket
from ts_model import extract_features_and_series


DATA_DIR = Path("data/train")
OUT_DIR = Path("solutions/ts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_TIMESTEPS = 128


def load_training_data():
    ds = load_dataset("aimo-interp/train-main-v2", split="train")
    ds = ds.filter(lambda row: row["is_robust"] is not None)

    cases = [
        {
            "model_id": row["model_id"],
            "reasoning_effort": row["reasoning_effort"],
            "original_problem": row["problem"],
            "problem_id": row.get("problem_id", row["problem"]),
        }
        for row in ds
    ]
    labels = [int(row["is_robust"]) for row in ds]

    return cases, labels


def load_validation_data():
    ds = load_dataset("aimo-interp/val-sample", split="train")

    cases = [
        {
            "model_id": row["model_id"],
            "reasoning_effort": row["reasoning_effort"],
            "original_problem": row["original_problem"],
            "problem_id": row.get("problem_id", row["original_problem"]),
        }
        for row in ds
    ]

    # Validation labels may use a different schema or be unavailable.
    labels = [
        row.get("model_is_robust")
        for row in ds
    ]

    return cases, labels

def evaluate(name, model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    print(f"\n{name}")
    print(f"Accuracy:          {accuracy_score(y_val, pred):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_val, pred):.4f}"
    )
    print(
        f"F1:                "
        f"{f1_score(y_val, pred, zero_division=0):.4f}"
    )

    return model

def main():
    cases, labels = load_training_data()

    if not cases:
        raise ValueError("The training dataset is empty.")

    if len(cases) != len(labels):
        raise ValueError(
            f"Input/label count mismatch: {len(cases)} cases, "
            f"{len(labels)} labels."
        )

        
    X_features, X_series, y, groups = [], [], [], []

    for i, (case, label) in enumerate(zip(cases, labels)):
        model_id = case["model_id"]
        problem = case["original_problem"]
        reasoning_effort = case["reasoning_effort"]

        # Extract activation features for this example.
        features, series = extract_features_and_series(
            model_id,
            problem,
        )

        features = np.asarray(features, dtype=np.float32).reshape(-1)
        series = np.asarray(series, dtype=np.float32)

        if series.ndim != 2:
            raise ValueError(f"Unexpected series shape: {series.shape}")

        if not np.isfinite(features).all() or not np.isfinite(series).all():
            raise ValueError(f"Non-finite values at row {i}")

        X_features.append(features)
        X_series.append(series)
        y.append(int(label))
        groups.append(str(case.get("problem_id", problem)))

        if (i + 1) % 5 == 0 or i + 1 == len(cases):
            print(f"Extracted {i + 1}/{len(cases)} training examples")

    X_features = np.stack(X_features)
    X_series = np.stack(X_series)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)


    if len(np.unique(y)) < 2:
        raise ValueError("Training requires both classes.")

    if len(np.unique(groups)) < 2:
        raise ValueError("Need at least two distinct problem groups.")

    print("\nDataset summary")
    print("Handcrafted features:", X_features.shape)
    print("Activation sequences:", X_series.shape)
    print("Class counts:", np.bincount(y, minlength=2))
    print("Distinct groups:", len(np.unique(groups)))

    # Cache expensive feature extraction so you can rerun experiments.
    np.save(OUT_DIR / "features.npy", X_features)
    np.save(OUT_DIR / "series.npy", X_series)
    np.save(OUT_DIR / "labels.npy", y)
    np.save(OUT_DIR / "groups.npy", groups)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(X_features, y, groups=groups)
    )

    Xf_train, Xf_val = X_features[train_idx], X_features[val_idx]
    Xs_train, Xs_val = X_series[train_idx], X_series[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "The grouped training split has only one class. "
            "Try a different split seed or more labeled groups."
        )

    print("\nSplit sizes")
    print("Train:", len(train_idx), "Validation:", len(val_idx))
    print("Train classes:", np.bincount(y_train, minlength=2))
    print("Validation classes:", np.bincount(y_val, minlength=2))

    # Experiment A: handcrafted ts features + logistic regression.
    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    lr_model = evaluate(
        "ts + Logistic Regression",
        lr_model,
        Xf_train,
        Xf_val,
        y_train,
        y_val,
    )

    joblib.dump(lr_model, OUT_DIR / "ts_logistic.joblib")

    # Experiment B: MiniRocket features + Ridge classifier.
    # Inspect the installed aeon API and only pass supported options.
    parameters = inspect.signature(MiniRocket).parameters
    rocket_kwargs = {}

    # Different aeon versions expose different constructor arguments.
    # n_jobs is included only if supported.
    if "n_jobs" in parameters:
        rocket_kwargs["n_jobs"] = 1

    rocket = MiniRocket(**rocket_kwargs)

    Z_train = rocket.fit_transform(Xs_train)
    Z_val = rocket.transform(Xs_val)

    Z_train = np.asarray(Z_train, dtype=np.float32)
    Z_val = np.asarray(Z_val, dtype=np.float32)

    if not np.isfinite(Z_train).all() or not np.isfinite(Z_val).all():
        raise ValueError("MiniRocket produced non-finite features.")

    rocket_model = make_pipeline(
        StandardScaler(),
        RidgeClassifier(class_weight="balanced"),
    )

    rocket_model = evaluate(
        "ts + MiniRocket + Ridge",
        rocket_model,
        Z_train,
        Z_val,
        y_train,
        y_val,
    )

    joblib.dump(
        {
            "transformer": rocket,
            "classifier": rocket_model,
            "n_timesteps": N_TIMESTEPS,
        },
        OUT_DIR / "ts_minirocket.joblib",
    )

    print("\nSaved:")
    print(OUT_DIR / "ts_logistic.joblib")
    print(OUT_DIR / "ts_minirocket.joblib")


if __name__ == "__main__":
    main()
