# SkillScope: Tech Job Market Analyzer

Portfolio project for software development internship applications. An end-to-end data analysis + ML project on LinkedIn tech job postings (2023–2024). It must stay clean, reproducible, and easy to explain in an interview.

Repo: https://github.com/KlRAAA/SkillScope-Tech-Job-Market-Analyzer

## Goal

Find which skills are in demand, how requirements differ by seniority and location, and what salary patterns exist. Then build:
1. **Seniority classifier**: Entry / Mid / Senior from title + description (NLP).
2. **Role clustering**: group postings into role types (Frontend, Data/ML, DevOps, ...).
3. **Streamlit dashboard** on Streamlit Community Cloud: EDA charts plus a "paste a job description" analyzer.

## Data

- Kaggle `arshkon/linkedin-job-postings` (~124k postings). Download: `python -m src.data.download`.
- Credentials live in `~/.kaggle/kaggle.json` only. Never put API keys in the repo or chat.
- `data/raw/` and `data/processed/jobs.parquet` are git-ignored (rebuild with `python -m src.data.clean`). A small (<50 MB) app sample will be committed.
- Never assume column names. Inspect the files first (see `notebooks/01_data_overview.ipynb`).

## Structure

```
data/raw/            git-ignored Kaggle files
data/processed/      jobs.parquet + small app sample
notebooks/           01_data_overview, 02_cleaning, 03_eda, 04_seniority_model, 05_role_clustering
src/data/            download.py, clean.py
src/features/        text.py, skills.py, pipeline.py
src/viz.py           shared chart style
src/models/          train_classifier.py, train_clusters.py, predict.py
models/              saved .joblib pipelines
reports/figures/     exported charts
reports/metrics.json
app/streamlit_app.py
tests/
```

## Conventions

- Reusable logic goes in `src/`. The notebooks import from `src/` and are only for exploration and explanation.
- Run modules from the repo root: `python -m src.data.download`.
- Environment: `.venv` on Python 3.14 (3.11 wasn't available as a normal Windows install). Versions are pinned in `requirements.txt`.
- Limited RAM: use Parquet, compact dtypes (category, float32), and filtered subsets.
- Format with `black`, lint with `ruff`, test with `pytest`.
- Charts use `src/viz.py`: `set_style()`, `barh()`, `save()`. Seniority colors are fixed (Entry blue, Mid orange, Senior aqua). Work type uses its own trio (violet/pink/green). Both are colorblind-validated.
- Work one phase at a time. At the end of each phase: run everything, summarize (what was built, key numbers, surprises, next step), make a small commit, and wait for the OK.
- For real trade-offs, present the options instead of picking silently.
- Git: the author is `Xyrus <xyrusdimacali@gmail.com>` only. No AI co-author trailers. Short imperative commit subjects.

## Phase checklist

- [x] **Phase 1: Setup and data overview.** Repo, venv, requirements, .gitignore, `download.py`, `01_data_overview.ipynb`
- [x] **Phase 2: Cleaning.** Joins, tech-title filter, dedupe, text cleaning, yearly salary + IQR, location/work type, seniority labels, `jobs.parquet`, tests
- [x] **Phase 3: EDA.** 10–15 charts with takeaways, key findings
- [x] **Phase 4: Features.** Skills extractor, years/keyword flags, TF-IDF, company-name removal, `src/features/pipeline.py` (build_features + ColumnTransformer + full pipeline)
- [ ] **Phase 5: Seniority classifier.** Baseline, LR/SVM/XGB CV, tuning, with/without title, explainability, error analysis
- [ ] **Phase 6: Role clustering.** TF-IDF → SVD → KMeans, choose k, name clusters, 2D plot, per-cluster profiles
- [ ] **Phase 7: Streamlit dashboard.** 4 pages, caching, committed sample, deployment steps
- [ ] **Phase 8: Polish.** README, black/ruff, CI with pytest, interview talking points

## Notes / decisions log

- `mappings/skills.csv` in the dataset is tiny (~680 bytes). These are LinkedIn's broad skill *categories* (e.g. "IT", "Engineering"), not specific tech skills. This is why Phase 4 builds its own regex skill extractor.
- Postings are almost all from April 2024 (listed ~2024-03-24 → 2024-04-20). Time trends are limited to ~4 weeks.
- `formatted_experience_level` is missing for 23.7%. With the spec mapping, Senior (Director + Executive) is only ~5% of labeled rows.
- `remote_allowed` is `1` or NaN only. `normalized_salary` exists but has absurd outliers, so we normalize ourselves.
- Phase 2 funnel: 123,849 raw → 10,820 tech titles → 10,040 after dedupe. Salary is known for 2,899 (28.9%). IQR fences are $6.6k–$260k (k=1.5).
- The tech filter favors precision. It drops ambiguous titles (test/application engineer, technical lead, business analyst). It keeps IT support (help desk, desktop support).
- Seniority mapping **C** (chosen): Intern + Entry + Associate → Entry (2,316), Mid-Senior → Mid (4,601), Director + Executive → Senior (191). Options A–D are compared in `02_cleaning.ipynb`. Salary outliers use IQR with k=1.5 (chosen).
- `work_type`: "on-site" means "not stated as remote or hybrid". There is no explicit on-site flag.
- `src/features/skills.py` (166 regex skills) and `text.py` (years + keyword flags) were built in Phase 3 because the EDA needed them. The skill matrix takes ~80s on 4 cores, so it is cached in `data/processed/skills.parquet` (git-ignored; delete it after changing patterns).
- EDA: 55% of postings were listed on Apr 18–19, 2024, so there are no trend claims. Entry descriptions are full of IT-support terms and staffing-agency names (Dice, TEKsystems). Watch for boilerplate leakage in Phase 5.
- Features have two stages. `build_features()` is stateless (text, skills, years, flags, length) and cached in `data/processed/features.parquet` with/without-title versions (`python -m src.features.pipeline`, ~3 min). `make_preprocessor()` holds the fitted parts (TF-IDF, imputer, scaler) and is fitted on train only. `make_full_pipeline()` makes the saved model accept raw title/description rows.
- Company-name leakage: 54% of descriptions contain their own company name, and some companies label almost all postings one way (TEKsystems 83% Entry, Motion Recruitment 94% Mid). Each posting's company name is removed from its text.
