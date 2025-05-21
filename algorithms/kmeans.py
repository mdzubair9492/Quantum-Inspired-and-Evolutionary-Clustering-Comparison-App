import numpy as np
from typing import Tuple, List, Any, Dict
from utils.metrics import compute_metrics

class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.inertia_history: List[float] = []

    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        idx = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        return data[idx].copy()

    def _assign_clusters(self, data: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(data[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def _update_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        new_centers = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            pts = data[labels == k]
            new_centers[k] = pts.mean(axis=0) if len(pts) > 0 else self.centroids[k]
        return new_centers

    def _calculate_sse(self, data: np.ndarray, labels: np.ndarray) -> float:
        sse = 0.0
        for k in range(self.n_clusters):
            pts = data[labels == k]
            sse += np.sum((pts - self.centroids[k])**2)
        return sse

    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], List[float]]:
        
        self.centroids = self._initialize_centroids(data)
        self.inertia_history = []

        for _ in range(self.max_iter):
            labels = self._assign_clusters(data)
            sse = self._calculate_sse(data, labels)
            self.inertia_history.append(sse)

            new_centers = self._update_centroids(data, labels)
            if np.allclose(new_centers, self.centroids):
                break
            self.centroids = new_centers

        
        labels = self._assign_clusters(data)
        final_sse = self.inertia_history[-1] if self.inertia_history else self._calculate_sse(data, labels)
        metrics = compute_metrics(data, labels, final_sse)
        return labels, metrics, self.inertia_history


def fit_predict(data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], List[float]]:
    k = params.get('n_clusters', 3)
    max_iter = params.get('max_iter', 100)
    km = KMeans(n_clusters=k, max_iter=max_iter)
    return km.fit(data)
