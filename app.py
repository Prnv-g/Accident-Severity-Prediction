"""FastAPI application for the saved US accident severity model."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from model_inference import AccidentSeverityModel, ModelContractError


class AccidentRecord(BaseModel):
    """The raw model inputs, using the original dataset column names as aliases."""

    model_config = ConfigDict(
        extra="forbid", populate_by_name=False, allow_inf_nan=False
    )

    start_time: str | None = Field(alias="Start_Time")
    street: str | None = Field(alias="Street")
    start_lat: float | None = Field(alias="Start_Lat")
    start_lng: float | None = Field(alias="Start_Lng")
    temperature_f: float | None = Field(alias="Temperature(F)")
    humidity_percent: float | None = Field(alias="Humidity(%)")
    pressure_in: float | None = Field(alias="Pressure(in)")
    visibility_mi: float | None = Field(alias="Visibility(mi)")
    wind_speed_mph: float | None = Field(alias="Wind_Speed(mph)")
    precipitation_in: float | None = Field(alias="Precipitation(in)")
    amenity: bool | None = Field(alias="Amenity")
    bump: bool | None = Field(alias="Bump")
    crossing: bool | None = Field(alias="Crossing")
    give_way: bool | None = Field(alias="Give_Way")
    junction: bool | None = Field(alias="Junction")
    no_exit: bool | None = Field(alias="No_Exit")
    railway: bool | None = Field(alias="Railway")
    roundabout: bool | None = Field(alias="Roundabout")
    station: bool | None = Field(alias="Station")
    stop: bool | None = Field(alias="Stop")
    traffic_calming: bool | None = Field(alias="Traffic_Calming")
    traffic_signal: bool | None = Field(alias="Traffic_Signal")
    state: str | None = Field(alias="State")
    weather_condition: str | None = Field(alias="Weather_Condition")
    sunrise_sunset: str | None = Field(alias="Sunrise_Sunset")
    civil_twilight: str | None = Field(alias="Civil_Twilight")

    @field_validator("start_time")
    @classmethod
    def validate_start_time(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = pd.to_datetime(value, format="mixed", errors="coerce")
        if pd.isna(parsed):
            raise ValueError("Start_Time must be a valid date-time string or null.")
        return value

    @field_validator(
        "amenity",
        "bump",
        "crossing",
        "give_way",
        "junction",
        "no_exit",
        "railway",
        "roundabout",
        "station",
        "stop",
        "traffic_calming",
        "traffic_signal",
        mode="before",
    )
    @classmethod
    def validate_road_flag(cls, value: Any) -> bool | None:
        if value is None or isinstance(value, bool):
            return value
        if type(value) is int and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0"}:
                return False
        raise ValueError("Road flags must be true, false, 1, 0, or null.")


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[AccidentRecord] = Field(min_length=1, max_length=1000)


class PredictionResult(BaseModel):
    predicted_severity: Annotated[int, Field(ge=1, le=4)]
    model_scores: dict[str, float]


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResult]


severity_model = AccidentSeverityModel()


app = FastAPI(
    title="US Accident Severity Prediction API",
    description=(
        "Serves the existing trained model. Model scores come from predict_proba "
        "but are not calibrated real-world probabilities."
    ),
    version="1.0.0",
)
app.state.severity_model = severity_model


def _model(request: Request) -> AccidentSeverityModel:
    return request.app.state.severity_model


@app.get("/health", tags=["system"])
def health(request: Request) -> dict[str, Any]:
    model = _model(request)
    return {
        "status": "ok",
        "model_ready": True,
        "model_name": model.model_name,
    }


@app.get("/model/info", tags=["model"])
def model_info(request: Request) -> dict[str, Any]:
    return _model(request).model_info()


@app.post("/predict", response_model=PredictionResult, tags=["prediction"])
def predict(record: AccidentRecord, request: Request) -> dict[str, Any]:
    try:
        raw_record = record.model_dump(by_alias=True)
        return _model(request).predict([raw_record])[0]
    except (ValueError, ModelContractError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"]
)
def predict_batch(
    payload: BatchPredictionRequest, request: Request
) -> dict[str, Any]:
    try:
        raw_records = [record.model_dump(by_alias=True) for record in payload.records]
        predictions = _model(request).predict(raw_records)
        return {"count": len(predictions), "predictions": predictions}
    except (ValueError, ModelContractError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
