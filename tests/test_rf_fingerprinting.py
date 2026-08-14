"""Comprehensive test suite for RF Fingerprinting Framework."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rf_fingerprinting.feature_extractor import RFSignalFeatureExtractor
from rf_fingerprinting.dimensionality_reducer import DimensionalityReducer
from rf_fingerprinting.clustering import RFClusterer
from rf_fingerprinting.classifier import RFFingerprinter


class TestRFSignalFeatureExtractor:
    """Test feature extraction module."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return RFSignalFeatureExtractor(sample_rate=2e6, fft_size=1024)

    @pytest.fixture
    def sample_signal(self):
        """Generate sample RF signal."""
        np.random.seed(42)
        t = np.linspace(0, 1, 2000)
        # QPSK modulated signal
        freq = 100e3
        phase = 2 * np.pi * freq * t
        amplitude = np.random.choice([1, -1], size=2000) + 1j * np.random.choice([1, -1], size=2000)
        signal = amplitude * np.exp(1j * phase) * np.exp(1j * np.random.randn(2000) * 0.1)
        return signal

    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.sample_rate == 2e6
        assert extractor.fft_size == 1024
        assert len(extractor.feature_names) > 0

    def test_feature_extraction(self, extractor, sample_signal):
        """Test feature extraction."""
        features = extractor.extract_features(sample_signal)
        
        assert isinstance(features, dict)
        assert len(features) == extractor.n_features
        assert all(np.isfinite(v) for v in features.values())
        
        # Check specific features exist
        assert 'mean_amplitude' in features
        assert 'spectral_centroid' in features
        assert 'phase_std' in features

    def test_feature_extraction_array(self, extractor, sample_signal):
        """Test feature extraction as array."""
        features_array = extractor.extract_features_array(sample_signal)
        
        assert isinstance(features_array, np.ndarray)
        assert features_array.shape == (extractor.n_features,)
        assert np.all(np.isfinite(features_array))

    def test_time_domain_features(self, extractor, sample_signal):
        """Test time-domain feature extraction."""
        features = extractor._extract_time_features(sample_signal)
        
        assert 'mean_amplitude' in features
        assert 'rms' in features
        assert 'crest_factor' in features
        assert features['rms'] >= 0
        assert features['crest_factor'] >= 1.0

    def test_frequency_domain_features(self, extractor, sample_signal):
        """Test frequency-domain feature extraction."""
        features = extractor._extract_frequency_features(sample_signal)
        
        assert 'spectral_centroid' in features
        assert 'spectral_entropy' in features
        assert 'spectral_power' in features
        assert features['spectral_power'] >= 0

    def test_phase_features(self, extractor, sample_signal):
        """Test phase feature extraction."""
        features = extractor._extract_phase_features(sample_signal)
        
        assert 'phase_std' in features
        assert 'instantaneous_frequency_mean' in features
        assert np.isfinite(features['phase_std'])

    def test_consistency(self, extractor, sample_signal):
        """Test that feature extraction is consistent."""
        features1 = extractor.extract_features_array(sample_signal)
        features2 = extractor.extract_features_array(sample_signal)
        
        np.testing.assert_array_almost_equal(features1, features2)


class TestDimensionalityReducer:
    """Test dimensionality reduction module."""

    @pytest.fixture
    def reducer(self):
        """Create reducer instance."""
        return DimensionalityReducer(method='pca', n_components=5)

    @pytest.fixture
    def sample_data(self):
        """Generate sample feature data."""
        np.random.seed(42)
        return np.random.randn(100, 20)

    def test_reducer_initialization(self, reducer):
        """Test reducer initialization."""
        assert reducer.n_components == 5
        assert not reducer.is_fitted

    def test_fit(self, reducer, sample_data):
        """Test fitting the reducer."""
        reducer.fit(sample_data)
        
        assert reducer.is_fitted
        assert reducer.explained_variance_ratio is not None
        assert len(reducer.explained_variance_ratio) == 5
        assert np.sum(reducer.explained_variance_ratio) <= 1.0

    def test_transform(self, reducer, sample_data):
        """Test transforming data."""
        reducer.fit(sample_data)
        X_reduced = reducer.transform(sample_data)
        
        assert X_reduced.shape == (100, 5)
        assert np.all(np.isfinite(X_reduced))

    def test_fit_transform(self, reducer, sample_data):
        """Test fit_transform."""
        X_reduced = reducer.fit_transform(sample_data)
        
        assert X_reduced.shape == (100, 5)
        assert reducer.is_fitted

    def test_inverse_transform(self, reducer, sample_data):
        """Test inverse transform."""
        X_reduced = reducer.fit_transform(sample_data)
        X_reconstructed = reducer.inverse_transform(X_reduced)
        
        assert X_reconstructed.shape == sample_data.shape
        # Reconstruction won't be perfect
        assert np.allclose(X_reconstructed, sample_data, atol=1.0)

    def test_get_components(self, reducer, sample_data):
        """Test getting principal components."""
        reducer.fit(sample_data)
        components = reducer.get_components()
        
        assert components.shape == (5, 20)

    def test_optimal_components(self, reducer, sample_data):
        """Test optimal component selection."""
        reducer.fit(sample_data)
        optimal = reducer.optimal_components(variance_threshold=0.9)
        
        assert 1 <= optimal <= 5

    def test_unfitted_error(self, reducer, sample_data):
        """Test error when using unfitted reducer."""
        with pytest.raises(RuntimeError):
            reducer.transform(sample_data)


class TestRFClusterer:
    """Test clustering module."""

    @pytest.fixture(params=['kmeans', 'gmm'])
    def clusterer(self, request):
        """Create clusterer instances."""
        return RFClusterer(method=request.param, n_clusters=3)

    @pytest.fixture
    def sample_data(self):
        """Generate sample clustered data."""
        np.random.seed(42)
        # Create three clusters
        c1 = np.random.randn(30, 10) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        c2 = np.random.randn(30, 10) + np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        c3 = np.random.randn(30, 10) + np.array([-5, -5, -5, -5, -5, -5, -5, -5, -5, -5])
        return np.vstack([c1, c2, c3])

    def test_clustering_initialization(self, clusterer):
        """Test clusterer initialization."""
        assert clusterer.n_clusters == 3
        assert not clusterer.is_fitted

    def test_fit(self, clusterer, sample_data):
        """Test fitting clusterer."""
        clusterer.fit(sample_data)
        
        assert clusterer.is_fitted
        assert len(clusterer.labels) == len(sample_data)
        assert clusterer.n_clusters_found > 0

    def test_predict(self, clusterer, sample_data):
        """Test prediction (for kmeans and gmm)."""
        if clusterer.method == 'dbscan':
            pytest.skip("DBSCAN doesn't support prediction")
        
        clusterer.fit(sample_data)
        labels = clusterer.predict(sample_data[:10])
        
        assert len(labels) == 10
        assert np.all(np.isin(labels, clusterer.labels))

    def test_get_labels(self, clusterer, sample_data):
        """Test getting labels."""
        clusterer.fit(sample_data)
        labels = clusterer.get_labels()
        
        assert len(labels) == len(sample_data)
        assert np.all(np.isin(labels, range(3)))

    def test_evaluate(self, clusterer, sample_data):
        """Test clustering evaluation."""
        clusterer.fit(sample_data)
        metrics = clusterer.evaluate(sample_data)
        
        assert isinstance(metrics, dict)
        # Should have at least one metric
        assert len(metrics) > 0

    def test_get_cluster_sizes(self, clusterer, sample_data):
        """Test getting cluster sizes."""
        clusterer.fit(sample_data)
        sizes = clusterer.get_cluster_sizes()
        
        assert isinstance(sizes, dict)
        assert sum(sizes.values()) == len(sample_data)

    def test_get_cluster_centers(self, clusterer, sample_data):
        """Test getting cluster centers."""
        clusterer.fit(sample_data)
        centers = clusterer.get_cluster_centers()
        
        if centers is not None:
            assert centers.shape[0] == 3  # n_clusters


class TestRFFingerprinter:
    """Test classification module."""

    @pytest.fixture(params=['rf', 'gb'])
    def fingerprinter(self, request):
        """Create fingerprinter instances."""
        return RFFingerprinter(method=request.param)

    @pytest.fixture
    def sample_data(self):
        """Generate sample labeled data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        # Create features
        X = np.random.randn(n_samples, n_features)
        
        # Create labels
        y = np.array(['device_a'] * 40 + ['device_b'] * 30 + ['device_c'] * 30)
        
        return X, y

    def test_fingerprinter_initialization(self, fingerprinter):
        """Test fingerprinter initialization."""
        assert fingerprinter.method in ['rf', 'gb', 'svm']
        assert not fingerprinter.is_fitted

    def test_fit(self, fingerprinter, sample_data):
        """Test fitting fingerprinter."""
        X, y = sample_data
        fingerprinter.fit(X, y)
        
        assert fingerprinter.is_fitted
        assert len(fingerprinter.classes) == 3
        assert np.array_equal(fingerprinter.classes, np.array(['device_a', 'device_b', 'device_c']))

    def test_predict(self, fingerprinter, sample_data):
        """Test prediction."""
        X, y = sample_data
        fingerprinter.fit(X, y)
        
        y_pred = fingerprinter.predict(X[:10])
        
        assert len(y_pred) == 10
        assert np.all(np.isin(y_pred, fingerprinter.classes))

    def test_predict_proba(self, fingerprinter, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        fingerprinter.fit(X, y)
        
        classes, proba = fingerprinter.predict_proba(X[:10])
        
        assert len(classes) == 3
        assert proba.shape == (10, 3)
        assert np.allclose(np.sum(proba, axis=1), 1.0)

    def test_evaluate(self, fingerprinter, sample_data):
        """Test evaluation."""
        X, y = sample_data
        fingerprinter.fit(X, y)
        
        metrics = fingerprinter.evaluate(X, y)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_get_classification_report(self, fingerprinter, sample_data):
        """Test classification report."""
        X, y = sample_data
        fingerprinter.fit(X, y)
        
        report = fingerprinter.get_classification_report(X, y)
        
        assert isinstance(report, str)
        assert 'precision' in report.lower()
        assert 'recall' in report.lower()

    def test_get_confusion_matrix(self, fingerprinter, sample_data):
        """Test confusion matrix."""
        X, y = sample_data
        fingerprinter.fit(X, y)
        
        cm = fingerprinter.get_confusion_matrix(X, y)
        
        assert cm.shape == (3, 3)
        assert np.sum(cm) == len(y)

    def test_get_feature_importances(self, fingerprinter, sample_data):
        """Test feature importances (for tree-based methods)."""
        if fingerprinter.method == 'svm':
            pytest.skip("SVM doesn't support feature importances")
        
        X, y = sample_data
        fingerprinter.fit(X, y)
        
        importances = fingerprinter.get_feature_importances()
        
        if importances is not None:
            assert len(importances) == X.shape[1]
            assert np.all(importances >= 0)


class TestIntegration:
    """Integration tests for complete pipeline."""

    @pytest.fixture
    def rf_signals(self):
        """Generate sample RF signals from different devices."""
        np.random.seed(42)
        signals = {}
        
        for device in ['device_a', 'device_b', 'device_c']:
            device_signals = []
            for _ in range(5):
                t = np.linspace(0, 1, 2000)
                freq = np.random.uniform(50e3, 200e3)
                phase = 2 * np.pi * freq * t
                
                # Different modulation for each device
                if device == 'device_a':
                    amp = np.sin(2 * np.pi * 10 * t)
                elif device == 'device_b':
                    amp = np.cos(2 * np.pi * 20 * t)
                else:
                    amp = np.exp(-t)
                
                signal = amp * np.exp(1j * phase)
                signal += 0.1 * np.random.randn(2000)
                device_signals.append(signal)
            
            signals[device] = device_signals
        
        return signals

    def test_complete_pipeline(self, rf_signals):
        """Test complete fingerprinting pipeline."""
        # Step 1: Extract features
        extractor = RFSignalFeatureExtractor()
        features_list = []
        labels_list = []
        
        for device, signals in rf_signals.items():
            for signal in signals:
                features = extractor.extract_features_array(signal)
                features_list.append(features)
                labels_list.append(device)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        assert X.shape[0] == 15
        assert len(y) == 15
        
        # Step 2: Reduce dimensionality
        reducer = DimensionalityReducer(n_components=10)
        X_reduced = reducer.fit_transform(X)
        
        assert X_reduced.shape == (15, 10)
        
        # Step 3: Clustering
        clusterer = RFClusterer(method='kmeans', n_clusters=3)
        clusterer.fit(X_reduced)
        
        clusters = clusterer.get_labels()
        assert len(clusters) == 15
        
        # Step 4: Classification
        fingerprinter = RFFingerprinter(method='rf')
        fingerprinter.fit(X, y)
        
        y_pred = fingerprinter.predict(X)
        accuracy = np.mean(y_pred == y)
        
        assert accuracy > 0.5  # Should have some accuracy
        assert len(y_pred) == 15

    def test_pipeline_with_save_load(self, rf_signals, tmp_path):
        """Test pipeline with model saving and loading."""
        # Extract features and train
        extractor = RFSignalFeatureExtractor()
        features_list = []
        labels_list = []
        
        for device, signals in rf_signals.items():
            for signal in signals:
                features = extractor.extract_features_array(signal)
                features_list.append(features)
                labels_list.append(device)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Train and save
        fingerprinter = RFFingerprinter(method='rf')
        fingerprinter.fit(X, y)
        
        filepath = tmp_path / 'fingerprinter.pkl'
        fingerprinter.save(str(filepath))
        
        # Load and test
        loaded_fingerprinter = RFFingerprinter.load(str(filepath))
        y_pred_orig = fingerprinter.predict(X)
        y_pred_loaded = loaded_fingerprinter.predict(X)
        
        np.testing.assert_array_equal(y_pred_orig, y_pred_loaded)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
