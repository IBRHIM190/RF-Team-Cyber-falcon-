"""Example usage of RF Fingerprinting Framework."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rf_fingerprinting.feature_extractor import RFSignalFeatureExtractor
from rf_fingerprinting.dimensionality_reducer import DimensionalityReducer
from rf_fingerprinting.clustering import RFClusterer
from rf_fingerprinting.classifier import RFFingerprinter


def generate_device_signals(device_type: str, n_samples: int = 10) -> list:
    """Generate RF signals for different device types.
    
    Args:
        device_type: Type of device ('wifi', '5g', 'bluetooth', 'radar')
        n_samples: Number of signal samples to generate
        
    Returns:
        List of complex signal arrays
    """
    np.random.seed(42)
    signals = []
    
    for _ in range(n_samples):
        t = np.linspace(0, 0.1, 2000)
        
        if device_type == 'wifi':
            # WiFi: OFDM-like modulation around 2.4 GHz
            freq = 2.4e9
            bw = 20e6  # 20 MHz bandwidth
            ofdm_signal = np.zeros(2000, dtype=complex)
            for k in range(52):
                subcarrier_freq = freq + (k - 26) * (bw / 52)
                ofdm_signal += np.exp(2j * np.pi * subcarrier_freq * t)
            signal = ofdm_signal / 52
            
        elif device_type == '5g':
            # 5G: Higher frequency, wider bandwidth
            freq = 28e9  # mmWave frequency
            bw = 100e6
            num_subcarriers = 200
            ofdm_signal = np.zeros(2000, dtype=complex)
            for k in range(num_subcarriers):
                subcarrier_freq = freq + (k - num_subcarriers/2) * (bw / num_subcarriers)
                ofdm_signal += np.exp(2j * np.pi * subcarrier_freq * t)
            signal = ofdm_signal / num_subcarriers
            
        elif device_type == 'bluetooth':
            # Bluetooth: FSK modulation, lower frequency
            freq_low = 2.4e9
            freq_high = freq_low + 1e6
            bits = np.random.choice([0, 1], size=100)
            signal = np.zeros(2000, dtype=complex)
            for i, bit in enumerate(bits):
                freq = freq_high if bit else freq_low
                idx_start = int(i * 20)
                idx_end = min(idx_start + 20, 2000)
                signal[idx_start:idx_end] = np.exp(2j * np.pi * freq * t[idx_start:idx_end])
            
        elif device_type == 'radar':
            # Radar: Chirp modulation (linear frequency modulation)
            freq_start = 76e9
            freq_end = 81e9
            chirp_signal = np.exp(2j * np.pi * np.linspace(freq_start, freq_end, 2000) * t)
            signal = chirp_signal
        
        else:
            raise ValueError(f"Unknown device type: {device_type}")
        
        # Add noise
        signal += 0.05 * (np.random.randn(2000) + 1j * np.random.randn(2000))
        signals.append(signal)
    
    return signals


def example_complete_pipeline():
    """Example: Complete RF fingerprinting pipeline."""
    print("="*70)
    print("RF Fingerprinting Framework - Complete Pipeline Example")
    print("="*70)
    
    # Step 1: Generate RF signals from different devices
    print("\n[1] Generating RF signals from different device types...")
    device_types = ['wifi', '5g', 'bluetooth', 'radar']
    all_signals = {}
    all_labels = []
    
    for device_type in device_types:
        signals = generate_device_signals(device_type, n_samples=10)
        all_signals[device_type] = signals
        all_labels.extend([device_type] * len(signals))
        print(f"   Generated {len(signals)} signals for {device_type}")
    
    print(f"   Total signals: {sum(len(s) for s in all_signals.values())}")
    
    # Step 2: Extract features
    print("\n[2] Extracting features from signals...")
    extractor = RFSignalFeatureExtractor(sample_rate=2e6, fft_size=1024)
    print(f"   Feature extractor initialized with {extractor.n_features} features")
    print(f"   Features: {', '.join(extractor.feature_names[:5])}...")
    
    features_list = []
    for device_type, signals in all_signals.items():
        for signal in signals:
            features = extractor.extract_features_array(signal)
            features_list.append(features)
    
    X = np.array(features_list)
    y = np.array(all_labels)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Feature statistics:")
    print(f"     Mean: {X.mean():.6f}")
    print(f"     Std:  {X.std():.6f}")
    print(f"     Min:  {X.min():.6f}")
    print(f"     Max:  {X.max():.6f}")
    
    # Step 3: Dimensionality reduction
    print("\n[3] Reducing dimensionality with PCA...")
    reducer = DimensionalityReducer(method='pca', n_components=10)
    X_reduced = reducer.fit_transform(X)
    
    print(f"   Reduced shape: {X_reduced.shape}")
    print(f"   Explained variance ratio: {reducer.get_explained_variance_ratio()[:5]}")
    print(f"   Cumulative variance: {reducer.get_cumulative_variance()[-1]:.4f}")
    
    optimal = reducer.optimal_components(variance_threshold=0.95)
    print(f"   Optimal components for 95% variance: {optimal}")
    
    # Step 4: Clustering analysis
    print("\n[4] Clustering analysis...")
    clusterer = RFClusterer(method='kmeans', n_clusters=4)
    clusterer.fit(X_reduced)
    
    clusters = clusterer.get_labels()
    cluster_sizes = clusterer.get_cluster_sizes()
    print(f"   Cluster sizes: {cluster_sizes}")
    
    metrics = clusterer.evaluate(X_reduced)
    print(f"   Clustering metrics:")
    for metric, value in metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # Step 5: Classification with RF Fingerprinter
    print("\n[5] Training RF Fingerprinter...")
    fingerprinter = RFFingerprinter(method='rf', n_estimators=100, max_depth=15)
    fingerprinter.fit(X, y)
    
    print(f"   Classes: {fingerprinter.classes}")
    print(f"   Feature importances (top 5):")
    
    importances = fingerprinter.get_feature_importances()
    if importances is not None:
        top_indices = np.argsort(importances)[-5:][::-1]
        for idx in top_indices:
            print(f"     Feature {idx}: {importances[idx]:.6f}")
    
    # Step 6: Evaluation
    print("\n[6] Evaluating classifier...")
    y_pred = fingerprinter.predict(X)
    metrics = fingerprinter.evaluate(X, y)
    
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    
    # Confusion matrix
    cm = fingerprinter.get_confusion_matrix(X, y)
    print(f"\n   Confusion Matrix:")
    print(f"   {cm}")
    
    # Step 7: Save model
    print("\n[7] Saving trained model...")
    model_path = 'fingerprinter_model.pkl'
    fingerprinter.save(model_path)
    print(f"   Model saved to: {model_path}")
    
    # Step 8: Load and test model
    print("\n[8] Loading and testing saved model...")
    loaded_fingerprinter = RFFingerprinter.load(model_path)
    y_pred_loaded = loaded_fingerprinter.predict(X[:10])
    print(f"   Predictions on 10 samples: {y_pred_loaded}")
    print(f"   True labels: {y[:10]}")
    
    print("\n" + "="*70)
    print("Pipeline complete!")
    print("="*70)


def example_single_signal_analysis():
    """Example: Analyze a single RF signal."""
    print("\n" + "="*70)
    print("Single Signal Analysis Example")
    print("="*70)
    
    # Generate a WiFi signal
    print("\nGenerating WiFi signal...")
    signals = generate_device_signals('wifi', n_samples=1)
    signal = signals[0]
    
    # Extract features
    print("Extracting features...")
    extractor = RFSignalFeatureExtractor()
    features = extractor.extract_features(signal)
    
    print("\nExtracted features:")
    for feature_name, value in list(features.items())[:10]:
        print(f"  {feature_name:40s}: {value:12.6f}")
    print(f"  ... ({len(features)} total features)")
    
    # Visualize signal
    print("\nVisualizing signal (if matplotlib available)...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time domain
        axes[0, 0].plot(np.abs(signal)[:500])
        axes[0, 0].set_title('Signal Magnitude (Time Domain)')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Magnitude')
        
        # Phase
        axes[0, 1].plot(np.angle(signal)[:500])
        axes[0, 1].set_title('Signal Phase (Time Domain)')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Phase (rad)')
        
        # Spectrum
        spectrum = np.abs(np.fft.fft(signal))
        freq = np.fft.fftfreq(len(signal))
        axes[1, 0].semilogy(freq[:len(freq)//2], spectrum[:len(spectrum)//2])
        axes[1, 0].set_title('Frequency Spectrum')
        axes[1, 0].set_xlabel('Normalized Frequency')
        axes[1, 0].set_ylabel('Magnitude')
        
        # Spectrogram
        from scipy import signal as scipy_signal
        f, t, Sxx = scipy_signal.spectrogram(signal[:500], fs=2e6, nperseg=128)
        axes[1, 1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10))
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_title('Spectrogram')
        
        plt.tight_layout()
        plt.savefig('rf_signal_analysis.png', dpi=100)
        print("Saved visualization to: rf_signal_analysis.png")
        plt.close()
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    # Run examples
    example_complete_pipeline()
    example_single_signal_analysis()
