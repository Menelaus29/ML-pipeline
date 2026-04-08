import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def get_clustering_metrics(X, labels, model=None) -> dict:
    # Compute silhouette, Davies-Bouldin, cluster count, noise points, and inertia from labels.
    labels = np.array(labels)

    n_clusters_found = len(set(labels) - {-1})
    noise_points = int(np.sum(labels == -1))

    inertia = None
    if model is not None and hasattr(model, "inertia_"):
        inertia = float(model.inertia_)

    if n_clusters_found < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "n_clusters_found": n_clusters_found,
            "noise_points": noise_points,
            "inertia": inertia,
            "warning": "n_clusters_found < 2; silhouette and Davies-Bouldin scores are undefined.",
        }

    # Exclude noise points (-1) before computing distance-based metrics
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    return {
        "silhouette_score": float(silhouette_score(X_clean, labels_clean)),
        "davies_bouldin_score": float(davies_bouldin_score(X_clean, labels_clean)),
        "n_clusters_found": n_clusters_found,
        "noise_points": noise_points,
        "inertia": inertia,
    }


def build_elbow_data(X, k_range=range(2, 11)) -> dict:
    # Fit KMeans for each k in k_range and return k values with their inertias.
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(float(km.inertia_))
    return {"k": list(k_range), "inertia": inertias}