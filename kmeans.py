import numpy as np
from typing import Optional, Tuple

class KMeans:

    def __init__(
            self,
            n_clusters: int,
            max_iter: int = 300,
            tol: float = 1e-4,
            init: str = "k-means++",
            n_init: int = 5,
            random_state: Optional[int] = None,
            standardize: bool = True
    ):
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.standardize = standardize

         # Fitted attributes
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.labels_: Optional[np.ndarray] = None

        # For standardization
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        # RNG
        self._rng = np.random.default_rng(random_state)


    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Compute k-means clustering on X.
        We standardize by default so one feature (e.g., volatility) does not drown out another (e.g., momentum).
        References: scikit-learn user guide scikit-learn developers. (n.d.),
        """
        X = self._check_array(X)
        Xs = self._standardize_fit(X) if self.standardize else X

        best_inertia = np.inf
        best = {}

        for _ in range(self.n_init):
            centers = self._init_centroids(Xs)
            labels, centers, inertia, n_iter = self._lloyd_loop(Xs, centers)

            if inertia < best_inertia:
                best_inertia = inertia
                best = {
                    "labels": labels,
                    "centers": centers,
                    "inertia": inertia,
                    "n_iter": n_iter
                }

        self.labels_ = best["labels"]
        self.cluster_centers_ = self._destandardize_centers(best["centers"]) if self.standardize else best["centers"]
        self.inertia = float(best["inertia"])
        self.n_iter_ = int(best["n_iter"])
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit to X and return labels."""
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample in X to the nearest centroid."""
        self._check_is_fitted()
        X = self._check_array(X)
        Xs = self._standardize_transform(X) if self.standardize else X
        centers = self._standardize_transform(self.cluster_centers_) if self.standardize else self.cluster_centers_
        return self._nearest_labels(Xs, centers)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return distances (L2) from each sample to each centroid."""
        self._check_is_fitted()
        X = self._check_array(X)
        Xs = self._standardize_transform(X) if self.standardize else X
        centers = self._standardize_transform(self.cluster_centers_) if self.standardize else self.cluster_centers_
        return self._pairwise_sq_dists(Xs, centers) ** 0.5

    # ---------- Core internals ----------

    def _lloyd_loop(
            self, Xs: np.ndarray,
            centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Main k-means loop: assign -> update until convergence or max_iter.
        This is the classic Lloyd/Forgy routine (Lloyd, 1982): regroup points around the nearest center,
        move centers to the middle of their group, repeat. Simple and reliable for small feature sets.
        References: scikit-learn developers. (n.d.)
        (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
        """
        for it in range(1, self.max_iter):
            labels = self._nearest_labels(Xs, centers)
            new_centers = self._recompute_centers(Xs, labels, centers)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= self.tol:
                inertia = self._inertia(Xs, centers, labels)
                return labels, centers, inertia, it
            
        # Max iterations reached
        inertia = self._inertia(Xs, centers, labels)
        return labels, centers, inertia, self.max_iter
        
    def _recompute_centers(self, Xs: np.ndarray, labels: np.ndarray, old_centers: np.ndarray) -> np.ndarray:
        """
        Recompute centroids; handle empty clusters by re-seeding to the farthest point
        (matches scikit-learn KMeans fallback: Reference: scikit-learn developers. (n.d.).
        If a cluster ends up with zero members, pick the farthest point and make it the new center.
        """
        K = self.n_clusters
        n_features = Xs.shape[1]
        centers = np.zeros((K, n_features), dtype=Xs.dtype)
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                centers[k] = Xs[mask].mean(axis=0)
            else:
                # Empty cluster: pick the farthest point from its nearest centroid
                d2 = self._pairwise_sq_dists(Xs, old_centers).min(axis=1)
                idx = int(np.argmax(d2))
                centers[k] = Xs[idx]
        return centers

    def _inertia(self, Xs: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
        """Sum of squared distances to closest centroid."""
        d2 = ((Xs - centers[labels]) ** 2).sum(axis=1)
        return float(d2.sum())

    # ---------- Initialization ----------

    def _init_centroids(self, Xs: np.ndarray) -> np.ndarray:
        if self.init == "random":
            idx = self._rng.choice(Xs.shape[0], size=self.n_clusters, replace=False)
            return Xs[idx].astype(float, copy=True)
        elif self.init == "k-means++":
            return self._kpp_init(Xs)
        else:
            raise ValueError('init must be "random" or "k-means++"')

    def _kpp_init(self, Xs: np.ndarray) -> np.ndarray:
        """
        k-means++ initialization.
        References: scikit-learn developers. (n.d.),
        Smart seeding to avoid bad first guesses; steadier results.
        """
        n_samples = Xs.shape[0]
        centers = np.empty((self.n_clusters, Xs.shape[1]), dtype=float)

        # Pick first center uniformly
        i0 = int(self._rng.integers(0, n_samples))
        centers[0] = Xs[i0]

        # Distances to nearest chosen center
        closest_d2 = self._pairwise_sq_dists(Xs, centers[0:1]).ravel()

        for k in range(1, self.n_clusters):
            # Prob ∝ distance^2
            probs = closest_d2 / closest_d2.sum()
            next_idx = int(self._rng.choice(n_samples, p=probs))
            centers[k] = Xs[next_idx]
            # Update closest distances
            d2_new = self._pairwise_sq_dists(Xs, centers[k:k+1]).ravel()
            closest_d2 = np.minimum(closest_d2, d2_new)
        return centers


    # ---------- Helpers ----------


    def _nearest_labels(self, Xs: np.ndarray, centers: np.ndarray) -> np.ndarray:
        d2 = self._pairwise_sq_dists(Xs, centers)
        return np.argmin(d2, axis=1)

    @staticmethod
    def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Efficient squared Euclidean distances between rows of A and rows of B.
        Returns shape (A.shape[0], B.shape[0]).
        """
        # (a - b)^2 = a^2 + b^2 - 2ab
        A2 = np.sum(A * A, axis=1, keepdims=True)
        B2 = np.sum(B * B, axis=1, keepdims=True).T
        AB = A @ B.T
        d2 = np.maximum(A2 + B2 - 2 * AB, 0.0)
        return d2

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        # Put every feature on the same scale so one number can't overwhelm the others.
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0, ddof=0)
        self._std[self._std == 0.0] = 1.0
        return (X - self._mean) / self._std

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            return X
        return (X - self._mean) / self._std

    def _destandardize_centers(self, centers: np.ndarray) -> np.ndarray:
        return centers * self._std + self._mean

    @staticmethod
    def _check_array(x: np.ndarray) -> np.ndarray:
        X = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2d array of [n_samples, n_features].")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        return X
    
    def _check_is_fitted(self):
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X) first.")
        

# Documentation references:        
# scikit-learn developers. (n.d.). sklearn.cluster.KMeans — scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# SciPy developers. (n.d.). scipy.cluster.vq.kmeans — SciPy documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html
# GeeksforGeeks. (n.d.). K-means++ Algorithm – ML. https://www.geeksforgeeks.org/machine-learning/ml-k-means-algorithm/
# Real Python. (n.d.). K-means clustering in Python. https://realpython.com/k-means-clustering-python/