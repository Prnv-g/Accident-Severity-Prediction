"""Inference utilities copied from the training notebook's prediction path.

The serialized bundle contains the fitted ColumnTransformer and classifier. Raw
timestamp and street values still need the deterministic feature engineering
defined in the notebook before the bundle is called.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "accident_severity_model.joblib"
MODEL_DETAILS_PATH = BASE_DIR / "model_details.json"

NUMERIC_WEATHER = [
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
]
ROAD_FLAGS = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
]
LOCATION = ["Start_Lat", "Start_Lng"]
RECORDED_CATEGORIES = [
    "State",
    "Weather_Condition",
    "Sunrise_Sunset",
    "Civil_Twilight",
]
RAW_FEATURES = (
    ["Start_Time", "Street"]
    + LOCATION
    + NUMERIC_WEATHER
    + ROAD_FLAGS
    + RECORDED_CATEGORIES
)
NUMERIC_FEATURES = (
    LOCATION
    + NUMERIC_WEATHER
    + ROAD_FLAGS
    + ["Hour", "Day_Of_Week", "Month", "Year"]
)
CATEGORY_FEATURES = RECORDED_CATEGORIES + ["Road_Type"]
FEATURES = NUMERIC_FEATURES + CATEGORY_FEATURES


class ModelContractError(RuntimeError):
    """Raised when a saved artifact does not match the notebook contract."""


def road_type(street: pd.Series) -> np.ndarray:
    """Group street labels exactly as in the training notebook."""

    street = street.astype("string").str.upper().str.strip().fillna("")
    return np.select(
        [
            street.str.match(r"^(?:I[- ]?\d|INTERSTATE\b)").to_numpy(dtype=bool),
            street.str.match(r"^(?:US[- ]?\d|U\.S\.\s*\d|[A-Z]{2}-\d)").to_numpy(
                dtype=bool
            ),
            street.str.contains(
                r"\b(?:HWY|HIGHWAY|RTE|ROUTE|FWY|FREEWAY)\b"
            ).to_numpy(dtype=bool),
            street.str.contains(r"\b(?:RAMP|EXIT)\b").to_numpy(dtype=bool),
            street.str.contains(r"\b(?:RD|ROAD|DR|DRIVE|LN|LANE)\b").to_numpy(
                dtype=bool
            ),
            street.str.contains(
                r"\b(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD)\b"
            ).to_numpy(dtype=bool),
            street.eq("").to_numpy(dtype=bool),
        ],
        [
            "Interstate",
            "Numbered route",
            "Highway/route",
            "Ramp",
            "Road/drive/lane",
            "Street/avenue",
            "Unknown",
        ],
        default="Other",
    )


def make_features(records: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's raw-to-model feature engineering in exact order."""

    data = pd.DataFrame(index=records.index)
    timestamp = pd.to_datetime(records["Start_Time"], format="mixed", errors="coerce")
    data["Hour"] = timestamp.dt.hour.astype("float32")
    data["Day_Of_Week"] = timestamp.dt.day_of_week.astype("float32")
    data["Month"] = timestamp.dt.month.astype("float32")
    data["Year"] = timestamp.dt.year.astype("float32")
    data["Road_Type"] = road_type(records["Street"])

    for column in LOCATION + NUMERIC_WEATHER:
        data[column] = pd.to_numeric(records[column], errors="coerce").astype("float32")
    for column in ROAD_FLAGS:
        values = records[column].astype("string").str.lower().str.strip()
        data[column] = values.map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).astype("float32")
    for column in RECORDED_CATEGORIES:
        values = records[column].astype("string").str.strip().replace("", pd.NA)
        data[column] = values.fillna("Unknown").astype(object)
    return data[FEATURES]


def _json_safe_records(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        return []
    frame = pd.read_csv(csv_path)
    frame = frame.astype(object).where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


class AccidentSeverityModel:
    """Loads and serves the existing model without changing its behavior."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.bundle = self._load_and_validate_bundle()
        self.model = self.bundle["model"]
        self.model_name = str(self.bundle["model_name"])
        self.raw_input_columns = list(self.bundle["raw_input_columns"])
        self.class_labels = [int(value) for value in self.bundle["class_labels"]]
        self.score_multipliers = np.asarray(
            self.bundle["score_multipliers"], dtype=float
        )
        self.probabilities_supported = callable(
            getattr(self.model, "predict_proba", None)
        )
        self.thread_limit = min(4, os.cpu_count() or 1)
        self.metrics = {
            "validation_metrics": _json_safe_records(
                BASE_DIR / "validation_metrics.csv"
            ),
            "reporting_shift_metrics": _json_safe_records(
                BASE_DIR / "reporting_shift_metrics.csv"
            ),
        }

    def _load_and_validate_bundle(self) -> dict[str, Any]:
        if not self.model_path.is_file():
            raise ModelContractError(f"Model file not found: {self.model_path}")

        bundle = joblib.load(self.model_path)
        if not isinstance(bundle, dict):
            raise ModelContractError("The model artifact must contain a dictionary bundle.")

        required_keys = {
            "model",
            "model_name",
            "raw_input_columns",
            "class_labels",
            "score_multipliers",
        }
        missing_keys = sorted(required_keys - set(bundle))
        if missing_keys:
            raise ModelContractError(
                f"Model bundle is missing keys: {', '.join(missing_keys)}"
            )

        raw_columns = list(bundle["raw_input_columns"])
        if raw_columns != RAW_FEATURES:
            raise ModelContractError(
                "Saved raw input columns do not match the notebook feature order."
            )

        with MODEL_DETAILS_PATH.open(encoding="utf-8") as details_file:
            details = json.load(details_file)
        if list(details.get("input_columns", [])) != raw_columns:
            raise ModelContractError(
                "model_details.json and the model bundle disagree on input columns."
            )

        labels = np.asarray(bundle["class_labels"], dtype=int)
        multipliers = np.asarray(bundle["score_multipliers"], dtype=float)
        model = bundle["model"]
        model_classes = np.asarray(getattr(model, "classes_", []), dtype=int)
        if labels.tolist() != [1, 2, 3, 4]:
            raise ModelContractError("The saved class label mapping is not [1, 2, 3, 4].")
        if len(multipliers) != len(labels):
            raise ModelContractError("Score multipliers do not match the class labels.")
        if not np.array_equal(model_classes + 1, labels):
            raise ModelContractError(
                "Classifier classes do not match the saved severity label mapping."
            )
        feature_names = list(getattr(model, "feature_names_in_", []))
        if feature_names != FEATURES:
            raise ModelContractError(
                "Classifier input order does not match the notebook's engineered features."
            )
        if not callable(getattr(model, "predict_proba", None)):
            raise ModelContractError(
                "The notebook prediction rule requires classifier predict_proba support."
            )
        return bundle

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            raise ValueError("At least one input record is required.")

        frame = pd.DataFrame.from_records(records)
        missing = sorted(set(self.raw_input_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing input columns: {', '.join(missing)}")
        frame = frame.loc[:, self.raw_input_columns]
        features = make_features(frame)

        with threadpool_limits(limits=self.thread_limit):
            scores = np.asarray(self.model.predict_proba(features), dtype=float)
        if scores.shape != (len(frame), len(self.class_labels)):
            raise ModelContractError("Classifier returned an unexpected score shape.")

        chosen_positions = (scores * self.score_multipliers).argmax(axis=1)
        results: list[dict[str, Any]] = []
        for position, row_scores in zip(chosen_positions, scores, strict=True):
            results.append(
                {
                    "predicted_severity": self.class_labels[int(position)],
                    "model_scores": {
                        str(label): float(score)
                        for label, score in zip(
                            self.class_labels, row_scores, strict=True
                        )
                    },
                }
            )
        return results

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "required_inputs": self.raw_input_columns,
            "supported_severity_levels": self.class_labels,
            "model_scores_available": self.probabilities_supported,
            "model_scores_calibrated": False,
            "score_multipliers": self.score_multipliers.tolist(),
            "evaluation_metrics": self.metrics,
        }
