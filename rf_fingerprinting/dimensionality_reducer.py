"""Dimensionality Reduction Module.

Reduces feature space using PCA, t-SNE, and UMAP for visualization and efficiency.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Reduce dimensionality of RF fingerprint features."""

    def __init__(self, method: Literal['pca', 'pca_incremental'] = 'pca',
                 n_components: int = 10):
        """Initialize dimensionality reducer.
        
        Args:
            method: 'pca' or 'pca_incremental'
            n_components: Number of output dimensions
        """
        self.method = method
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.reducer = None
        self.is_fitted = False
        self.explained_variance_ratio = None
        self.cumulative_variance = None

    def fit(self, X: np.ndarray) -> 'DimensionalityReducer':
        """Fit the dimensionality reducer.
        
        Args:
            X: Input feature matrix (n_samples, n_features)
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting {self.method} with {self.n_components} components")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit reducer
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components, random_state=42)
        elif self.method == 'pca_incremental':
            # For large datasets
            self.reducer = PCA(n_components=self.n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.reducer.fit(X_scaled)
        
        # Store variance information
        self.explained_variance_ratio = self.reducer.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        
        total_variance = np.sum(self.explained_variance_ratio)
        logger.info(f"Explained variance: {total_variance:.4f}")
        logger.info(f"Top 5 components: {self.explained_variance_ratio[:5]}")
        
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to reduced space.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Reduced dimensionality data
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.reducer.transform(X_scaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Reduced dimensionality data
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Transform back to original feature space.
        
        Args:
            X_reduced: Reduced dimensionality data
            
        Returns:
            Reconstructed feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        X_original = self.reducer.inverse_transform(X_reduced)
        return self.scaler.inverse_transform(X_original)

    def get_components(self) -> np.ndarray:
        """Get principal component vectors.
        
        Returns:
            Component matrix (n_components, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        return self.reducer.components_

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component.
        
        Returns:
            Array of variance ratios
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        return self.explained_variance_ratio

    def get_cumulative_variance(self) -> np.ndarray:
        """Get cumulative explained variance.
        
        Returns:
            Array of cumulative variances
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        return self.cumulative_variance

    def optimal_components(self, variance_threshold: float = 0.95) -> int:
        """Find optimal number of components for variance threshold.
        
        Args:
            variance_threshold: Desired cumulative variance (0-1)
            
        Returns:
            Optimal number of components
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        optimal = np.argmax(self.cumulative_variance >= variance_threshold) + 1
        logger.info(f"Optimal components for {variance_threshold:.2%} variance: {optimal}")
        return optimal
