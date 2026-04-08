import numpy as np
import pytest
from backend.services.clustering import get_clustering_metrics, build_elbow_data


def make_two_cluster_data():
    # Return a well-separated 2-cluster dataset and matching integer labels.
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=0.0, scale=0.3, size=(50, 2))
    c2 = rng.normal(loc=5.0, scale=0.3, size=(50, 2))
    X = np.vstack([c1, c2])
    labels = np.array([0] * 50 + [1] * 50)
    return X, labels


def test_silhouette_positive_for_two_clean_clusters():
    X, labels = make_two_cluster_data()
    result = get_clustering_metrics(X, labels)
    assert result["silhouette_score"] > 0


def test_noise_points_counted_correctly():
    X, labels = make_two_cluster_data()
    labels_with_noise = labels.copy()
    labels_with_noise[:5] = -1
    result = get_clustering_metrics(X, labels_with_noise)
    assert result["noise_points"] == 5


def test_fewer_than_two_clusters_returns_none_scores():
    X, _ = make_two_cluster_data()
    labels = np.zeros(100, dtype=int)
    result = get_clustering_metrics(X, labels)
    assert result["silhouette_score"] is None
    assert result["davies_bouldin_score"] is None
    assert "warning" in result


def test_build_elbow_data_matching_lengths():
    X, _ = make_two_cluster_data()
    k_range = range(2, 6)
    result = build_elbow_data(X, k_range=k_range)
    assert len(result["k"]) == len(result["inertia"])
    assert result["k"] == list(k_range)