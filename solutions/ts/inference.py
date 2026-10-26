
from pathlib import Path
import joblib

from trr_model import extract_features

_ARTIFACT_PATH = Path(__file__).parent / "trr_classifier.joblib"
_CLASSIFIER = None


def predict_robustness(model_id, reasoning_effort, problems):
    global _CLASSIFIER

    if _CLASSIFIER is None:
        _CLASSIFIER = joblib.load(_ARTIFACT_PATH)

    predictions = []

    for problem in problems:
        features = extract_features(
            model_id=model_id,
            problem=problem,
            reasoning_effort=reasoning_effort,
        )

        prediction = _CLASSIFIER.predict(
            features.reshape(1, -1)
        )[0]

        predictions.append(bool(prediction))

    return predictions
