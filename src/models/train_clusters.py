"""Group tech postings into role types.

Pipeline: text (title x3 + description, company name removed)
-> TF-IDF (English + HR-boilerplate stop words) -> TruncatedSVD (100 dims)
-> L2 normalize -> KMeans.

TF-IDF + SVD is latent semantic analysis (LSA). It compresses 20k sparse word
features into 100 dense "topics", so that KMeans' Euclidean distances behave
like cosine similarity between documents. k is chosen from a 4-15 scan
(elbow + silhouette) and from how interpretable the clusters are.

Usage:
    python -m src.models.train_clusters
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer

from src.features.pipeline import cluster_texts
from src.features.text import BOILERPLATE_STOP_WORDS, TOKEN_PATTERN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS_PATH = PROJECT_ROOT / "data" / "processed" / "jobs.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "clusters.joblib"
ASSIGNMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "clusters.parquet"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

SEED = 42
K_RANGE = range(4, 16)
K = 10  # see 05_role_clustering.ipynb for why

# Cluster id -> (name, a term that must be among the cluster's top terms).
# KMeans is seeded, so ids are stable for the same data. The check catches a
# silent relabeling if the data or settings change.
CLUSTER_NAMES = {
    0: ("Data Science & ML", "machine learning"),
    1: ("Application Development (Java/.NET/Web)", "developer"),
    2: ("Software Engineering (Product)", "software engineer"),
    3: ("IT Support & Help Desk", "help desk"),
    4: ("Network & Systems Administration", "network"),
    5: ("Data & Business Analysis", "data analyst"),
    6: ("Software, Cloud & QA Engineering", "devops"),
    7: ("Data Engineering & BI", "data engineer"),
    8: ("Cybersecurity", "cybersecurity"),
    9: ("Solutions & Cloud Architecture", "architect"),
}


def make_lsa(k: int = K) -> Pipeline:
    return Pipeline(
        [
            ("text", FunctionTransformer(cluster_texts)),
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words=list(ENGLISH_STOP_WORDS.union(BOILERPLATE_STOP_WORDS)),
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.5,
                    max_features=20_000,
                    sublinear_tf=True,
                    token_pattern=TOKEN_PATTERN,
                    dtype=np.float32,
                ),
            ),
            ("svd", TruncatedSVD(n_components=100, random_state=SEED)),
            ("normalize", Normalizer(copy=False)),
            ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=SEED)),
        ]
    )


def scan_k(Z: np.ndarray, k_range=K_RANGE, sample: int = 4000) -> pd.DataFrame:
    """Inertia (for the elbow) and silhouette (on a fixed sample, for speed) per k."""
    idx = np.random.default_rng(SEED).choice(len(Z), min(sample, len(Z)), replace=False)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(Z[idx], km.labels_[idx]),
            }
        )
    return pd.DataFrame(rows)


def top_terms(model: Pipeline, n: int = 15) -> dict[int, list[str]]:
    """Top TF-IDF terms per cluster, from the centroids mapped back to word space."""
    terms = np.array(model.named_steps["tfidf"].get_feature_names_out())
    centroids = model.named_steps["svd"].inverse_transform(
        model.named_steps["kmeans"].cluster_centers_
    )
    return {
        c: terms[np.argsort(row)[::-1][:n]].tolist() for c, row in enumerate(centroids)
    }


def check_names(terms: dict[int, list[str]]) -> None:
    for cluster, (name, signature) in CLUSTER_NAMES.items():
        if not any(signature in term for term in terms[cluster]):
            raise ValueError(
                f"Cluster {cluster} no longer looks like '{name}' (top terms: {terms[cluster][:8]}). "
                "Re-inspect the clusters and update CLUSTER_NAMES."
            )


def main() -> None:
    jobs = pd.read_parquet(JOBS_PATH)
    X = jobs[["title", "description_clean", "company_name"]]

    model = make_lsa(K)
    labels = model.fit_predict(X)
    Z = model[:-1].transform(X)  # normalized LSA vectors
    print(
        f"LSA explained variance: {model.named_steps['svd'].explained_variance_ratio_.sum():.2f}"
    )

    print("Scanning k ...")
    scan = scan_k(Z)
    print(scan.round(4).to_string(index=False))

    terms = top_terms(model)
    check_names(terms)
    names = {c: name for c, (name, _) in CLUSTER_NAMES.items()}

    print("Running t-SNE for the 2D map ...")
    xy = TSNE(
        n_components=2, init="pca", perplexity=40, random_state=SEED
    ).fit_transform(Z)

    assignments = pd.DataFrame(
        {
            "job_id": jobs["job_id"].to_numpy(),
            "cluster": labels,
            "cluster_name": [names[c] for c in labels],
            "tsne_x": xy[:, 0].astype(np.float32),
            "tsne_y": xy[:, 1].astype(np.float32),
        }
    )
    assignments.to_parquet(ASSIGNMENTS_PATH, index=False)
    joblib.dump({"pipeline": model, "names": names, "k": K}, MODEL_PATH, compress=3)

    sizes = assignments["cluster"].value_counts().sort_index()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics["clusters"] = {
        "k": K,
        "method": "title x3 + description -> TF-IDF -> SVD(100) -> L2 -> KMeans",
        "explained_variance": float(
            model.named_steps["svd"].explained_variance_ratio_.sum()
        ),
        "k_scan": scan.to_dict(orient="records"),
        "clusters": [
            {
                "id": c,
                "name": names[c],
                "size": int(sizes[c]),
                "top_terms": terms[c][:10],
            }
            for c in range(K)
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    for c in sizes.sort_values(ascending=False).index:
        print(f"{c:2d} {sizes[c]:5d}  {names[c]:<42} {', '.join(terms[c][:8])}")
    print(
        f"Saved {MODEL_PATH.relative_to(PROJECT_ROOT)} and {ASSIGNMENTS_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
