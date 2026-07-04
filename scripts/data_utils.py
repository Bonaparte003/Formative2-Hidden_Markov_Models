"""Shared dataset discovery and preprocessing utilities."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
CLEAN_DIR = DATASET_DIR / "cleaned"
WINDOW_DIR = DATASET_DIR / "windows"

ACTIVITIES = ("still", "standing", "walking", "jumping")
ACTIVITY_PATTERN = re.compile(r"^(still|standing|walking|jumping)\d*-", re.IGNORECASE)

TRIM_START_SEC = 1.0
TRIM_END_SEC = 10.0
TARGET_HZ = 100.0
WINDOW_SAMPLES = 100
HOP_SAMPLES = 50

UNSEEN_SESSION_PREFIXES = (
    "still5-2026-07-01",
    "standing5-2026-07-01",
    "walking5-2026-07-01",
    "jumping5-2026-07-01",
)


def parse_activity(name: str) -> str | None:
    match = ACTIVITY_PATTERN.match(name)
    return match.group(1).lower() if match else None


def normalize_session_name(name: str) -> str:
    """Lowercase the activity prefix so exports match still1-, walking5-, etc."""
    match = ACTIVITY_PATTERN.match(name)
    if not match:
        return name
    return name[: match.end()].lower() + name[match.end() :]


def unzip_archives(dataset_dir: Path = DATASET_DIR) -> int:
    """Extract any zip in dataset/ that does not yet have a matching folder."""
    extracted = 0
    for archive in sorted(dataset_dir.glob("*.zip")):
        target = dataset_dir / archive.stem
        if target.is_dir():
            continue
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        extracted += 1
    return extracted


def find_all_recording_dirs(dataset_dir: Path = DATASET_DIR) -> list[dict]:
    """Return metadata for every extracted Sensor Logger session folder."""
    sessions: list[dict] = []
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir() or folder.name in {"cleaned", "windows"}:
            continue
        activity = parse_activity(folder.name)
        if activity is None:
            continue
        if not (folder / "Accelerometer.csv").exists():
            continue
        meta = load_metadata(folder)
        session = normalize_session_name(folder.name)
        sessions.append(
            {
                "activity": activity,
                "session": session,
                "path": folder,
                "device": meta["device"],
                "platform": meta["platform"],
                "recording_time": meta["recording_time"],
                "sample_rate_hz": meta["sample_rate_hz"],
                "is_unseen": is_unseen_session(folder.name),
            }
        )
    if not sessions:
        raise FileNotFoundError(f"No Sensor Logger folders found in {dataset_dir}")
    return sessions


def is_unseen_session(session_name: str) -> bool:
    lower = session_name.lower()
    return any(lower.startswith(prefix) for prefix in UNSEEN_SESSION_PREFIXES)


def load_metadata(recording_dir: Path) -> dict:
    meta = pd.read_csv(recording_dir / "Metadata.csv").iloc[0]
    rates = str(meta["sampleRateMs"]).split("|")
    sensors = str(meta["sensors"]).split("|")
    rate_map = {s: int(r) if r else None for s, r in zip(sensors, rates)}
    return {
        "device": meta["device name"],
        "platform": meta["platform"],
        "recording_time": meta["recording time"],
        "sample_rate_hz": 1000 / rate_map["Accelerometer"],
    }


def load_merged_sensors(recording_dir: Path) -> pd.DataFrame:
    accel = pd.read_csv(recording_dir / "Accelerometer.csv")
    gyro = pd.read_csv(recording_dir / "Gyroscope.csv")
    merged = accel.merge(
        gyro,
        on=["time", "seconds_elapsed"],
        suffixes=("_accel", "_gyro"),
    )
    return merged.sort_values("seconds_elapsed").reset_index(drop=True)


def trim_activity_segment(
    df: pd.DataFrame,
    start_sec: float = TRIM_START_SEC,
    end_sec: float = TRIM_END_SEC,
) -> pd.DataFrame:
    mask = (df["seconds_elapsed"] >= start_sec) & (df["seconds_elapsed"] <= end_sec)
    return df.loc[mask].reset_index(drop=True)


def resample_to_hz(df: pd.DataFrame, target_hz: float = TARGET_HZ) -> pd.DataFrame:
    t0, t1 = df["seconds_elapsed"].iloc[0], df["seconds_elapsed"].iloc[-1]
    n = int((t1 - t0) * target_hz) + 1
    new_t = np.linspace(t0, t1, n)
    cols = [c for c in df.columns if c not in {"time", "seconds_elapsed"}]
    out: dict[str, np.ndarray] = {"seconds_elapsed": new_t}
    for col in cols:
        out[col] = np.interp(new_t, df["seconds_elapsed"], df[col])
    return pd.DataFrame(out)


def preprocess_recording(recording_dir: Path, target_hz: float = TARGET_HZ) -> pd.DataFrame:
    meta = load_metadata(recording_dir)
    merged = load_merged_sensors(recording_dir)
    merged = resample_to_hz(merged, target_hz=meta["sample_rate_hz"])
    return trim_activity_segment(merged)


def extract_windows_from_df(df: pd.DataFrame, activity: str, session: str) -> list[dict]:
    windows = []
    for start in range(0, len(df) - WINDOW_SAMPLES + 1, HOP_SAMPLES):
        segment = df.iloc[start : start + WINDOW_SAMPLES]
        windows.append(
            {
                "activity": activity,
                "session": session,
                "start_s": float(segment["seconds_elapsed"].iloc[0]),
                "data": segment,
            }
        )
    return windows


def load_all_windows(dataset_dir: Path = DATASET_DIR) -> tuple[list[dict], pd.DataFrame]:
    """Load every session and return window dicts plus a collection summary table."""
    unzip_archives(dataset_dir)
    sessions = find_all_recording_dirs(dataset_dir)
    summary_rows = []
    all_windows: list[dict] = []

    for sess in sessions:
        df = preprocess_recording(sess["path"])
        windows = extract_windows_from_df(df, sess["activity"], sess["session"])
        all_windows.extend(windows)
        summary_rows.append(
            {
                "Activity": sess["activity"],
                "Session": sess["session"],
                "Device": sess["device"],
                "Platform": sess["platform"],
                "Sample Rate (Hz)": sess["sample_rate_hz"],
                "Trimmed Duration (s)": round(
                    df["seconds_elapsed"].iloc[-1] - df["seconds_elapsed"].iloc[0], 2
                ),
                "Windows": len(windows),
                "Split": "unseen" if sess["is_unseen"] else "train",
            }
        )

    return all_windows, pd.DataFrame(summary_rows)
