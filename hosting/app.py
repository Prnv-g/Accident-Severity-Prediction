"""FastAPI application for the saved US accident severity model."""

from __future__ import annotations

import json
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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


# Lazily load or instantiate model
_model_instance = None


def get_model() -> AccidentSeverityModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = AccidentSeverityModel()
    return _model_instance


app = FastAPI(
    title="US Accident Severity Prediction API",
    description=(
        "Serves the trained US Accident Severity machine learning model. "
        "Predicts accident severity from 1 (least severe) to 4 (most severe)."
    ),
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse, tags=["ui"])
def root_ui():
    """Interactive HTML web tester for browser testing on Vercel."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Accident Severity Model API</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --accent: #38bdf8;
      --text: #f8fafc;
      --muted: #94a3b8;
      --border: #334155;
    }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
      line-height: 1.5;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    h1 { margin: 0; font-size: 1.6rem; color: #fff; }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #064e3b;
      color: #34d399;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 500;
    }
    .links a {
      color: var(--accent);
      text-decoration: none;
      margin-right: 14px;
      font-weight: 500;
    }
    .links a:hover { text-decoration: underline; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
    }
    label {
      display: block;
      font-size: 0.8rem;
      color: var(--muted);
      margin-bottom: 4px;
    }
    input, select {
      width: 100%;
      box-sizing: border-box;
      padding: 8px 10px;
      background: #0f172a;
      border: 1px solid var(--border);
      color: #fff;
      border-radius: 6px;
      font-size: 0.9rem;
    }
    button {
      background: #0284c7;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.95rem;
      transition: background 0.15s;
    }
    button:hover { background: #0369a1; }
    .result-box {
      margin-top: 16px;
      padding: 16px;
      border-radius: 8px;
      background: #0f172a;
      border: 1px solid var(--border);
      display: none;
    }
    .sev-badge {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 1.1rem;
      font-weight: 700;
      margin-top: 8px;
    }
    .sev-1 { background: #15803d; color: #fff; }
    .sev-2 { background: #0284c7; color: #fff; }
    .sev-3 { background: #d97706; color: #fff; }
    .sev-4 { background: #dc2626; color: #fff; }
    pre {
      background: #090d16;
      padding: 10px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>US Accident Severity Prediction API</h1>
        <div style="font-size:0.85rem; color: var(--muted); margin-top: 4px;">
          Histogram Gradient Boosting (Weighted) &bull; Hosted on Vercel
        </div>
      </div>
      <div>
        <span class="status-badge">&bull; Model Ready</span>
      </div>
    </header>

    <div style="margin-bottom: 16px;" class="links">
      <a href="/docs" target="_blank">Interactive Swagger Docs (/docs)</a>
      <a href="/model/info" target="_blank">Model Info (/model/info)</a>
      <a href="/health" target="_blank">Health Check (/health)</a>
    </div>

    <div class="card">
      <h3 style="margin-top: 0;">Quick Test Prediction</h3>
      <form id="predForm">
        <div class="grid">
          <div>
            <label>Start Time</label>
            <input type="text" id="Start_Time" value="2022-08-15 17:30:00">
          </div>
          <div>
            <label>Street</label>
            <input type="text" id="Street" value="I-95 N">
          </div>
          <div>
            <label>Start Latitude</label>
            <input type="number" step="0.0001" id="Start_Lat" value="40.7128">
          </div>
          <div>
            <label>Start Longitude</label>
            <input type="number" step="0.0001" id="Start_Lng" value="-74.0060">
          </div>
          <div>
            <label>Temperature (&deg;F)</label>
            <input type="number" step="0.1" id="Temperature(F)" value="78.0">
          </div>
          <div>
            <label>Humidity (%)</label>
            <input type="number" step="0.1" id="Humidity(%)" value="68.0">
          </div>
          <div>
            <label>Pressure (in)</label>
            <input type="number" step="0.01" id="Pressure(in)" value="29.85">
          </div>
          <div>
            <label>Visibility (mi)</label>
            <input type="number" step="0.1" id="Visibility(mi)" value="10.0">
          </div>
          <div>
            <label>Wind Speed (mph)</label>
            <input type="number" step="0.1" id="Wind_Speed(mph)" value="8.0">
          </div>
          <div>
            <label>Precipitation (in)</label>
            <input type="number" step="0.01" id="Precipitation(in)" value="0.0">
          </div>
          <div>
            <label>State</label>
            <input type="text" id="State" value="NY">
          </div>
          <div>
            <label>Weather Condition</label>
            <input type="text" id="Weather_Condition" value="Fair">
          </div>
          <div>
            <label>Sunrise / Sunset</label>
            <select id="Sunrise_Sunset">
              <option value="Day">Day</option>
              <option value="Night">Night</option>
            </select>
          </div>
          <div>
            <label>Civil Twilight</label>
            <select id="Civil_Twilight">
              <option value="Day">Day</option>
              <option value="Night">Night</option>
            </select>
          </div>
          <div>
            <label>Junction</label>
            <select id="Junction">
              <option value="true">True</option>
              <option value="false">False</option>
            </select>
          </div>
          <div>
            <label>Traffic Signal</label>
            <select id="Traffic_Signal">
              <option value="false">False</option>
              <option value="true">True</option>
            </select>
          </div>
        </div>
        <div style="margin-top: 18px; display:flex; gap:10px;">
          <button type="submit">Predict Severity</button>
        </div>
      </form>

      <div id="resultBox" class="result-box">
        <h4 style="margin: 0 0 8px 0;">Prediction Result:</h4>
        <div id="badgeContainer"></div>
        <pre id="jsonResult" style="margin-top: 12px;"></pre>
      </div>
    </div>
  </div>

  <script>
    const flags = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming'];
    document.getElementById('predForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const payload = {
        "Start_Time": document.getElementById('Start_Time').value,
        "Street": document.getElementById('Street').value,
        "Start_Lat": parseFloat(document.getElementById('Start_Lat').value) || null,
        "Start_Lng": parseFloat(document.getElementById('Start_Lng').value) || null,
        "Temperature(F)": parseFloat(document.getElementById('Temperature(F)').value) || null,
        "Humidity(%)": parseFloat(document.getElementById('Humidity(%)').value) || null,
        "Pressure(in)": parseFloat(document.getElementById('Pressure(in)').value) || null,
        "Visibility(mi)": parseFloat(document.getElementById('Visibility(mi)').value) || null,
        "Wind_Speed(mph)": parseFloat(document.getElementById('Wind_Speed(mph)').value) || null,
        "Precipitation(in)": parseFloat(document.getElementById('Precipitation(in)').value) || null,
        "State": document.getElementById('State').value,
        "Weather_Condition": document.getElementById('Weather_Condition').value,
        "Sunrise_Sunset": document.getElementById('Sunrise_Sunset').value,
        "Civil_Twilight": document.getElementById('Civil_Twilight').value,
        "Junction": document.getElementById('Junction').value === 'true',
        "Traffic_Signal": document.getElementById('Traffic_Signal').value === 'true'
      };
      flags.forEach(f => payload[f] = false);

      const resBox = document.getElementById('resultBox');
      const badgeBox = document.getElementById('badgeContainer');
      const jsonRes = document.getElementById('jsonResult');
      resBox.style.display = 'block';
      badgeBox.innerHTML = 'Computing...';

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (response.ok) {
          const sev = data.predicted_severity;
          badgeBox.innerHTML = `<span class="sev-badge sev-${sev}">Predicted Severity: Level ${sev}</span>`;
          jsonRes.textContent = JSON.stringify(data, null, 2);
        } else {
          badgeBox.innerHTML = '<span style="color:#f87171;">Prediction Error</span>';
          jsonRes.textContent = JSON.stringify(data, null, 2);
        }
      } catch (err) {
        badgeBox.innerHTML = '<span style="color:#f87171;">Network Error</span>';
        jsonRes.textContent = err.toString();
      }
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    model = get_model()
    return {
        "status": "ok",
        "model_ready": True,
        "model_name": model.model_name,
    }


@app.get("/model/info", tags=["model"])
def model_info() -> dict[str, Any]:
    return get_model().model_info()


@app.post("/predict", response_model=PredictionResult, tags=["prediction"])
def predict(record: AccidentRecord) -> dict[str, Any]:
    try:
        raw_record = record.model_dump(by_alias=True)
        return get_model().predict([raw_record])[0]
    except (ValueError, ModelContractError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"]
)
def predict_batch(payload: BatchPredictionRequest) -> dict[str, Any]:
    try:
        raw_records = [record.model_dump(by_alias=True) for record in payload.records]
        predictions = get_model().predict(raw_records)
        return {"count": len(predictions), "predictions": predictions}
    except (ValueError, ModelContractError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
