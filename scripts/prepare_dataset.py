"""Export labeled, windowed CSV files from Sensor Logger recordings."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset"
CLEAN_DIR = ROOT / "dataset" / "cleaned"
WINDOW_DIR = ROOT / "dataset" / "windows"

ACTIVITIES = ("still", "standing", "walking", "jumping")
TRIM_START_SEC = 1.0  # skip phone placement / record-button tap
TRIM_END_SEC = 10.0   # stop before pause-button tap at 11 s
WINDOW_SAMPLES = 100  # 1 s at 100 Hz
HOP_SAMPLES = 50


def find_recording_dirs() -> dict[str, Path]:
    recordings: dict[str, Path] = {}
    for activity in ACTIVITIES:
        matches = sorted(RAW_DIR.glob(f"{activity}-*"))
        if not matches:
            raise FileNotFoundError(f"No recording found for activity '{activity}'")
        recordings[activity] = matches[0]
    return recordings


def load_metadata(recording_dir: Path) -> dict[str, str]:
    meta = pd.read_csv(recording_dir / "Metadata.csv")
    row = meta.iloc[0]
    rates = str(row["sampleRateMs"]).split("|")
    sensors = str(row["sensors"]).split("|")
    sample_rate_ms = {
        sensor: int(rate) if rate else None for sensor, rate in zip(sensors, rates)
    }
    return {
        "device": row["device name"],
        "platform": row["platform"],
        "recording_time": row["recording time"],
        "sample_rate_hz": 1000 / sample_rate_ms["Accelerometer"],
    }


def load_merged_sensors(recording_dir: Path) -> pd.DataFrame:
    accel = pd.read_csv(recording_dir / "Accelerometer.csv")
    gyro = pd.read_csv(recording_dir / "Gyroscope.csv")

    merged = accel.merge(
        gyro,
        on=["time", "seconds_elapsed"],
        suffixes=("_accel", "_gyro"),
    )
    merged = merged.sort_values("seconds_elapsed").reset_index(drop=True)
    return merged


def trim_activity_segment(
    df: pd.DataFrame,
    start_sec: float = TRIM_START_SEC,
    end_sec: float = TRIM_END_SEC,
) -> pd.DataFrame:
    """Keep only the clean activity window, excluding setup and pause artifacts."""
    mask = (df["seconds_elapsed"] >= start_sec) & (df["seconds_elapsed"] <= end_sec)
    return df.loc[mask].reset_index(drop=True)


def resample_to_target_hz(df: pd.DataFrame, target_hz: float = 100.0) -> pd.DataFrame:
    duration = df["seconds_elapsed"].iloc[-1] - df["seconds_elapsed"].iloc[0]
    n_samples = int(duration * target_hz) + 1
    target_times = np.linspace(df["seconds_elapsed"].iloc[0], df["seconds_elapsed"].iloc[-1], n_samples)

    columns = [c for c in df.columns if c not in {"time", "seconds_elapsed"}]
    resampled = {"seconds_elapsed": target_times}
    for col in columns:
        resampled[col] = np.interp(target_times, df["seconds_elapsed"], df[col])
    return pd.DataFrame(resampled)


def export_full_recordings() -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for activity, recording_dir in find_recording_dirs().items():
        meta = load_metadata(recording_dir)
        merged = load_merged_sensors(recording_dir)
        merged = resample_to_target_hz(merged, target_hz=meta["sample_rate_hz"])
        merged = trim_activity_segment(merged)
        merged.insert(0, "activity", activity)
        merged.insert(1, "timestamp", merged["seconds_elapsed"])

        out_path = CLEAN_DIR / f"{activity}_full_recording.csv"
        merged.to_csv(out_path, index=False)

        summary_rows.append(
            {
                "activity": activity,
                "device": meta["device"],
                "recording_time": meta["recording_time"],
                "sample_rate_hz": meta["sample_rate_hz"],
                "duration_s": round(merged["seconds_elapsed"].iloc[-1], 2),
                "n_samples": len(merged),
                "file": out_path.name,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(CLEAN_DIR / "collection_summary.csv", index=False)
    return summary


def export_windowed_samples() -> pd.DataFrame:
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for activity, recording_dir in find_recording_dirs().items():
        activity_dir = WINDOW_DIR / activity
        activity_dir.mkdir(parents=True, exist_ok=True)

        merged = load_merged_sensors(recording_dir)
        merged = resample_to_target_hz(merged, target_hz=100.0)
        merged = trim_activity_segment(merged)
        sensor_cols = [c for c in merged.columns if c != "seconds_elapsed"]

        sample_idx = 1
        for start in range(0, len(merged) - WINDOW_SAMPLES + 1, HOP_SAMPLES):
            window = merged.iloc[start : start + WINDOW_SAMPLES].copy()
            window.insert(0, "activity", activity)
            window.insert(1, "window_id", sample_idx)
            window.insert(2, "timestamp", window["seconds_elapsed"])

            filename = f"{activity}_sample_{sample_idx:02d}.csv"
            window.to_csv(activity_dir / filename, index=False)

            manifest_rows.append(
                {
                    "activity": activity,
                    "window_id": sample_idx,
                    "start_s": round(window["seconds_elapsed"].iloc[0], 3),
                    "end_s": round(window["seconds_elapsed"].iloc[-1], 3),
                    "n_samples": len(window),
                    "file": f"{activity}/{filename}",
                }
            )
            sample_idx += 1

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(WINDOW_DIR / "window_manifest.csv", index=False)
    return manifest


if __name__ == "__main__":
    summary = export_full_recordings()
    manifest = export_windowed_samples()
    print(f"Exported {len(summary)} full recordings to {CLEAN_DIR}")
    print(f"Exported {len(manifest)} windowed samples to {WINDOW_DIR}")
