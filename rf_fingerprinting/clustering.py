"""RF Signal Clustering Module.

Clusters RF signals based on extracted features using K-Means, DBSCAN, and GMM.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Tuple, Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class RFClusterer:
    """Cluster RF signals into device types/sources."""

    def __init__(self, method: Literal['kmeans', 'dbscan', 'gmm'] = 'kmeans',
                 n_clusters: int = 5, **kwargs):
        """Initialize RF clusterer.
        
        Args:
            method: Clustering method ('kmeans', 'dbscan', 'gmm')
            n_clusters: Number of clusters (for kmeans and gmm)
            **kwargs: Additional parameters for the clustering method
        """
        self.method = method
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.clusterer = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.labels = None
        self.n_clusters_found = None

    def fit(self, X: np.ndarray) -> 'RFClusterer':
        """Fit clustering model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting {self.method} clustering")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit clusterer
        if self.method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                **self.kwargs
            )
            self.labels = self.clusterer.fit_predict(X_scaled)
            self.n_clusters_found = self.n_clusters
            
        elif self.method == 'dbscan':
            eps = self.kwargs.get('eps', 0.5)
            min_samples = self.kwargs.get('min_samples', 5)
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.labels = self.clusterer.fit_predict(X_scaled)
            self.n_clusters_found = len(set(self.labels)) - (1 if -1 in self.labels else 0)
            logger.info(f"Found {self.n_clusters_found} clusters, noise points: {np.sum(self.labels == -1)}")
            
        elif self.method == 'gmm':
            self.clusterer = GaussianMixture(
                n_components=self.n_clusters,
                random_state=42,
                **self.kwargs
            )
            self.clusterer.fit(X_scaled)
            self.labels = self.clusterer.predict(X_scaled)
            self.n_clusters_found = self.n_clusters
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        logger.info(f"Clustering complete. Labels shape: {self.labels.shape}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'kmeans':
            return self.clusterer.predict(X_scaled)
        elif self.method == 'gmm':
            return self.clusterer.predict(X_scaled)
        elif self.method == 'dbscan':
            # DBSCAN doesn't have predict for new data
            raise NotImplementedError("DBSCAN doesn't support prediction on new data")

    def get_labels(self) -> np.ndarray:
        """Get cluster labels for training data.
        
        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")
        return self.labels

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        labels = self.labels
        
        metrics = {}
        
        # Only compute for valid labels (not noise in DBSCAN)
        valid_mask = labels != -1
        if np.sum(valid_mask) == 0:
            logger.warning("No valid clusters found")
            return metrics
        
        X_valid = X_scaled[valid_mask]
        labels_valid = labels[valid_mask]
        
        # Silhouette score (higher is better, range: -1 to 1)
        if len(set(labels_valid)) > 1 and len(labels_valid) > 1:
            metrics['silhouette'] = silhouette_score(X_valid, labels_valid)
        
        # Davies-Bouldin Index (lower is better)
        if len(set(labels_valid)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
        
        # Calinski-Harabasz Index (higher is better)
        if len(set(labels_valid)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_valid, labels_valid)
        
        # Inertia (for kmeans)
        if self.method == 'kmeans' and hasattr(self.clusterer, 'inertia_'):
            metrics['inertia'] = self.clusterer.inertia_
        
        # Log results
        logger.info(f"Clustering evaluation metrics: {metrics}")
        return metrics

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers (if applicable).
        
        Returns:
            Cluster centers in scaled space, or None if not applicable
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")
        
        if self.method == 'kmeans':
            return self.clusterer.cluster_centers_
        elif self.method == 'gmm':
            return self.clusterer.means_
        return None

    def get_cluster_sizes(self) -> Dict[int, int]:
        """Get size of each cluster.
        
        Returns:
            Dictionary mapping cluster ID to size
        """
        if not self.is_fitted:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")
        
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))
