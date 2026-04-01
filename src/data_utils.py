from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/ankushnarwade/indian-climate-dataset-20242025"

REQUIRED_COLUMNS = {
    "Date": ["date"],
    "City": ["city"],
    "State": ["state"],
    "Temperature_Max_C": ["temperature_max_c", "temperature_max"],
    "Temperature_Min_C": ["temperature_min_c", "temperature_min"],
    "Temperature_Avg_C": ["temperature_avg_c", "temperature_avg"],
    "Humidity_pct": ["humidity_pct", "humidity"],
    "Rainfall_mm": ["rainfall_mm", "rainfall"],
    "Wind_Speed_kmh": ["wind_speed_km_h", "wind_speed_kmh", "wind_speed"],
    "AQI": ["aqi"],
    "AQI_Category": ["aqi_category"],
    "Pressure_hPa": ["pressure_hpa", "pressure"],
    "Cloud_Cover_pct": ["cloud_cover_pct", "cloud_cover"],
}

NUMERIC_COLUMNS = [
    "Temperature_Max_C",
    "Temperature_Min_C",
    "Temperature_Avg_C",
    "Humidity_pct",
    "Rainfall_mm",
    "Wind_Speed_kmh",
    "AQI",
    "Pressure_hPa",
    "Cloud_Cover_pct",
]


def _normalize_column(name: str) -> str:
    text = name.strip().lower()
    text = text.replace("°", "")
    text = text.replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def ensure_dataset(data_path: str | Path) -> Path:
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {path}. Download Indian_Climate_Dataset_2024_2025.csv "
            f"from {KAGGLE_DATASET_URL} and place it in the data folder."
        )

    original = pd.read_csv(path)
    normalized_lookup = {_normalize_column(column): column for column in original.columns}

    rename_map: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in REQUIRED_COLUMNS.items():
        source = None
        for alias in aliases:
            if alias in normalized_lookup:
                source = normalized_lookup[alias]
                break
        if source is None:
            missing.append(canonical)
        else:
            rename_map[source] = canonical

    if missing:
        raise ValueError(
            "The dataset does not match the expected schema. Missing columns: "
            + ", ".join(missing)
        )

    df = original.rename(columns=rename_map)[list(REQUIRED_COLUMNS.keys())].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "City", "Temperature_Avg_C"]).sort_values(["City", "Date"])

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Temperature_Avg_C"])
    df.to_csv(path, index=False)
    return path


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    ensure_dataset(data_path)
    df = pd.read_csv(data_path, parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)
