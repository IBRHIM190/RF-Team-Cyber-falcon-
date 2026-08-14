"""RF Fingerprint Classifier Module.

Classifies RF signals into known device types using supervised learning.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Tuple, Dict, Optional, Literal, List
import logging
import pickle

logger = logging.getLogger(__name__)


class RFFingerprinter:
    """Classify RF signals and fingerprint device types."""

    def __init__(self, method: Literal['rf', 'gb', 'svm'] = 'rf', **kwargs):
        """Initialize RF fingerprinter.
        
        Args:
            method: Classification method ('rf'=RandomForest, 'gb'=GradientBoosting, 'svm'=SVM)
            **kwargs: Additional parameters for the classifier
        """
        self.method = method
        self.kwargs = kwargs
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.classes = None
        self.feature_importances = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RFFingerprinter':
        """Fit classifier on labeled RF signals.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (device types/sources)
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting {self.method} classifier")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_
        
        logger.info(f"Classes: {self.classes}")
        logger.info(f"Class distribution: {np.bincount(y_encoded)}")
        
        # Create and fit classifier
        if self.method == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', 15),
                random_state=42,
                n_jobs=-1,
                **{k: v for k, v in self.kwargs.items() if k not in ['n_estimators', 'max_depth']}
            )
        elif self.method == 'gb':
            self.classifier = GradientBoostingClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', 5),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=42,
                **{k: v for k, v in self.kwargs.items() if k not in ['n_estimators', 'max_depth', 'learning_rate']}
            )
        elif self.method == 'svm':
            self.classifier = SVC(
                kernel=self.kwargs.get('kernel', 'rbf'),
                C=self.kwargs.get('C', 1.0),
                probability=True,
                random_state=42,
                **{k: v for k, v in self.kwargs.items() if k not in ['kernel', 'C']}
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.classifier.fit(X_scaled, y_encoded)
        
        # Store feature importances if available
        if hasattr(self.classifier, 'feature_importances_'):
            self.feature_importances = self.classifier.feature_importances_
        
        self.is_fitted = True
        logger.info("Classifier fitting complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict device type for RF signals.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        y_encoded = self.classifier.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (class_names, probabilities)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        if not hasattr(self.classifier, 'predict_proba'):
            raise NotImplementedError(f"{self.method} doesn't support probability prediction")
        
        X_scaled = self.scaler.transform(X)
        proba = self.classifier.predict_proba(X_scaled)
        return self.classes, proba

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        y_pred = self.predict(X)
        y_encoded = self.label_encoder.transform(y)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC for binary classification
        if len(self.classes) == 2:
            try:
                _, proba = self.predict_proba(X)
                metrics['roc_auc'] = roc_auc_score(y_encoded, proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def get_classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        """Get detailed classification report.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Classification report string
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        y_pred = self.predict(X)
        return classification_report(y, y_pred)

    def get_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get confusion matrix.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Confusion matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred, labels=self.classes)

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Get feature importances (for tree-based methods).
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importances is None:
            return None
        
        return self.feature_importances

    def save(self, filepath: str) -> None:
        """Save classifier to disk.
        
        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted classifier")
        
        state = {
            'method': self.method,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classes': self.classes,
            'feature_importances': self.feature_importances
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Classifier saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'RFFingerprinter':
        """Load classifier from disk.
        
        Args:
            filepath: Path to saved classifier
            
        Returns:
            Loaded RFFingerprinter instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        instance = cls(method=state['method'])
        instance.classifier = state['classifier']
        instance.scaler = state['scaler']
        instance.label_encoder = state['label_encoder']
        instance.classes = state['classes']
        instance.feature_importances = state['feature_importances']
        instance.is_fitted = True
        
        logger.info(f"Classifier loaded from {filepath}")
        return instance
