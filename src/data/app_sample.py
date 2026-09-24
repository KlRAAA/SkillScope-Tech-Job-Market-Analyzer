"""Build the small dataset the Streamlit app loads (committed to the repo).

The app never needs full descriptions, so it gets every tech posting with
the columns the dashboard uses, the role cluster, and the 0/1 skill flags.
Size is a few MB, well under the 50 MB budget.

Usage (after clean, features and clustering have run):
    python -m src.data.app_sample
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.skills import SKILL_NAMES, skill_column

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
APP_SAMPLE_PATH = PROCESSED / "app_jobs.parquet"

COLUMNS = [
    "job_id", "title", "company_name", "seniority", "employment_type", "work_type",
    "city", "state", "location_level", "salary_yearly", "listed_date",
]  # fmt: skip


def build() -> pd.DataFrame:
    jobs = pd.read_parquet(PROCESSED / "jobs.parquet")
    clusters = pd.read_parquet(PROCESSED / "clusters.parquet")
    skills = pd.read_parquet(PROCESSED / "skills.parquet")
    assert (clusters["job_id"].to_numpy() == jobs["job_id"].to_numpy()).all()
    assert len(skills) == len(jobs)

    app = jobs[COLUMNS].copy()
    app["cluster"] = clusters["cluster_name"].astype("category").to_numpy()
    app["skill_count"] = skills["skill_count"].to_numpy()
    flags = skills.drop(columns="skill_count").rename(
        columns={skill_column(n): n for n in SKILL_NAMES}
    )
    app = pd.concat([app, flags.astype(np.uint8).set_index(app.index)], axis=1)
    return app


def main() -> None:
    app = build()
    app.to_parquet(APP_SAMPLE_PATH, index=False, compression="zstd")
    size_mb = APP_SAMPLE_PATH.stat().st_size / 1e6
    print(f"Saved {len(app):,} rows x {app.shape[1]} columns to "
          f"{APP_SAMPLE_PATH.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")  # fmt: skip


if __name__ == "__main__":
    main()
