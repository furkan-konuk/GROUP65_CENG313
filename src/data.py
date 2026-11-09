"""Data loading helpers for the "Fake or Real: The Impostor Hunt in Texts" competition.

Expected layout (as downloaded from Kaggle):

    data/
    ├── train.csv              # columns: id, real_text_id (1 or 2)
    ├── train/
    │   ├── article_0000/
    │   │   ├── file_1.txt
    │   │   └── file_2.txt
    │   └── ...
    └── test/
        └── article_XXXX/ ...
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def _read_text(path: Path) -> str:
    """Read a text file safely (some fake texts contain unusual unicode)."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def find_data_dir(start: str | os.PathLike = ".") -> Path:
    """Locate the folder that contains `train/` and `train.csv`.

    Kaggle zips sometimes add an extra nested folder, so we search a few levels.
    """
    start = Path(start)
    for candidate in [start, *start.rglob("*")]:
        if candidate.is_dir() and (candidate / "train").is_dir() and (candidate / "train.csv").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a folder with train/ and train.csv under {start.resolve()}"
    )


def load_split(data_dir: str | os.PathLike, split: str = "train") -> pd.DataFrame:
    """Load one split as a DataFrame with columns: id, text_1, text_2 (+ real_text_id for train)."""
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    rows = []
    for article in sorted(split_dir.iterdir()):
        if not article.is_dir():
            continue
        match = re.search(r"(\d+)$", article.name)
        if match is None:
            continue
        rows.append(
            {
                "id": int(match.group(1)),
                "text_1": _read_text(article / "file_1.txt"),
                "text_2": _read_text(article / "file_2.txt"),
            }
        )
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

    if split == "train":
        labels = pd.read_csv(data_dir / "train.csv")
        df = df.merge(labels[["id", "real_text_id"]], on="id", how="left")
    return df


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """One row per text (two rows per article). Handy for EDA.

    Columns: id, file (1/2), text, is_real (only if labels exist).
    """
    parts = []
    for k in (1, 2):
        part = df[["id", f"text_{k}"]].rename(columns={f"text_{k}": "text"})
        part["file"] = k
        if "real_text_id" in df:
            part["is_real"] = (df["real_text_id"] == k).astype(int)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["id", "file"]).reset_index(drop=True)


def get_target(df: pd.DataFrame):
    """Binary target for models: 1 if file_1 is the real one, else 0."""
    return (df["real_text_id"] == 1).astype(int).values
