"""Export labeled, windowed CSV files from all Sensor Logger recordings."""

from pathlib import Path

import pandas as pd

from data_utils import (
    CLEAN_DIR,
    HOP_SAMPLES,
    WINDOW_DIR,
    WINDOW_SAMPLES,
    find_all_recording_dirs,
    preprocess_recording,
    unzip_archives,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"


def export_full_recordings() -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for sess in find_all_recording_dirs():
        df = preprocess_recording(sess["path"])
        df.insert(0, "activity", sess["activity"])
        df.insert(1, "session", sess["session"])
        df.insert(2, "timestamp", df["seconds_elapsed"])

        out = CLEAN_DIR / f"{sess['session']}.csv"
        df.to_csv(out, index=False)
        rows.append(
            {
                "activity": sess["activity"],
                "session": sess["session"],
                "device": sess["device"],
                "sample_rate_hz": sess["sample_rate_hz"],
                "duration_s": round(df["seconds_elapsed"].iloc[-1], 2),
                "n_samples": len(df),
                "split": "unseen" if sess["is_unseen"] else "train",
                "file": out.name,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(CLEAN_DIR / "collection_summary.csv", index=False)
    return summary


def export_windowed_samples() -> pd.DataFrame:
    WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for sess in find_all_recording_dirs():
        activity_dir = WINDOW_DIR / sess["activity"]
        activity_dir.mkdir(parents=True, exist_ok=True)

        df = preprocess_recording(sess["path"])
        session_slug = sess["session"].split("-")[0]  # e.g. still5

        for sample_idx, start in enumerate(
            range(0, len(df) - WINDOW_SAMPLES + 1, HOP_SAMPLES), start=1
        ):
            window = df.iloc[start : start + WINDOW_SAMPLES].copy()
            window.insert(0, "activity", sess["activity"])
            window.insert(1, "session", sess["session"])
            window.insert(2, "window_id", sample_idx)
            window.insert(3, "timestamp", window["seconds_elapsed"])

            filename = f"{session_slug}_sample_{sample_idx:02d}.csv"
            window.to_csv(activity_dir / filename, index=False)

            manifest_rows.append(
                {
                    "activity": sess["activity"],
                    "session": sess["session"],
                    "window_id": sample_idx,
                    "start_s": round(window["seconds_elapsed"].iloc[0], 3),
                    "end_s": round(window["seconds_elapsed"].iloc[-1], 3),
                    "n_samples": len(window),
                    "split": "unseen" if sess["is_unseen"] else "train",
                    "file": f"{sess['activity']}/{filename}",
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(WINDOW_DIR / "window_manifest.csv", index=False)
    return manifest


if __name__ == "__main__":
    n = unzip_archives(DATASET_DIR)
    if n:
        print(f"Extracted {n} new zip archive(s)")
    summary = export_full_recordings()
    manifest = export_windowed_samples()
    print(f"Exported {len(summary)} trimmed recordings to {CLEAN_DIR}")
    print(f"Exported {len(manifest)} labelled window CSVs to {WINDOW_DIR}")
    print(summary.groupby("activity")["session"].count().rename("sessions"))
    print(manifest.groupby("activity").size().rename("windows"))
