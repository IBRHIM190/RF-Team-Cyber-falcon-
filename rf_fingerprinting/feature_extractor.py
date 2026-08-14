"""RF Signal Feature Extraction Module.

Extracts statistical, spectral, and temporal features from RF signals.
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class RFSignalFeatureExtractor:
    """Extract comprehensive features from RF signals for fingerprinting."""

    def __init__(self, sample_rate: float = 2e6, fft_size: int = 1024):
        """Initialize feature extractor.
        
        Args:
            sample_rate: Sampling rate in Hz (default: 2 MHz)
            fft_size: FFT size for spectral analysis
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.feature_names = []
        self._build_feature_list()

    def _build_feature_list(self):
        """Build list of feature names."""
        # Time-domain features
        time_features = [
            'mean_amplitude', 'std_amplitude', 'max_amplitude',
            'min_amplitude', 'rms', 'peak_to_average_power',
            'crest_factor', 'skewness', 'kurtosis'
        ]
        
        # Frequency-domain features
        freq_features = [
            'spectral_centroid', 'spectral_spread', 'spectral_flatness',
            'spectral_entropy', 'spectral_power', 'peak_frequency'
        ]
        
        # Phase features
        phase_features = [
            'phase_std', 'phase_mean', 'instantaneous_frequency_mean',
            'instantaneous_frequency_std'
        ]
        
        # Statistical moments
        moment_features = [
            'moment_2', 'moment_3', 'moment_4',
            'normalized_moment_2', 'normalized_moment_3', 'normalized_moment_4'
        ]
        
        # Autocorrelation features
        autocorr_features = [
            'autocorr_peak', 'autocorr_slope',
            'autocorr_periodicity_strength'
        ]
        
        self.feature_names = (
            time_features + freq_features + phase_features +
            moment_features + autocorr_features
        )

    @property
    def n_features(self) -> int:
        """Return number of features extracted."""
        return len(self.feature_names)

    def extract_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract all features from RF signal.
        
        Args:
            signal_data: Complex IQ signal array
            
        Returns:
            Dictionary mapping feature names to values
        """
        features = {}
        
        # Time-domain features
        features.update(self._extract_time_features(signal_data))
        
        # Frequency-domain features
        features.update(self._extract_frequency_features(signal_data))
        
        # Phase features
        features.update(self._extract_phase_features(signal_data))
        
        # Statistical moments
        features.update(self._extract_moment_features(signal_data))
        
        # Autocorrelation features
        features.update(self._extract_autocorrelation_features(signal_data))
        
        return features

    def extract_features_array(self, signal_data: np.ndarray) -> np.ndarray:
        """Extract features as numpy array (ordered by feature_names).
        
        Args:
            signal_data: Complex IQ signal array
            
        Returns:
            Feature vector (1D array)
        """
        features_dict = self.extract_features(signal_data)
        return np.array([features_dict[name] for name in self.feature_names])

    def _extract_time_features(self, sig: np.ndarray) -> Dict[str, float]:
        """Extract time-domain features."""
        # Convert complex signal to amplitude
        amplitude = np.abs(sig)
        
        # Basic statistics
        mean_amp = np.mean(amplitude)
        std_amp = np.std(amplitude)
        max_amp = np.max(amplitude)
        min_amp = np.min(amplitude)
        
        # Power metrics
        rms = np.sqrt(np.mean(amplitude ** 2))
        power = amplitude ** 2
        avg_power = np.mean(power)
        peak_power = np.max(power)
        pap = peak_power / avg_power if avg_power > 0 else 1.0
        
        # Crest factor (peak to RMS)
        crest = max_amp / rms if rms > 0 else 1.0
        
        # Higher-order statistics
        normalized_amp = (amplitude - mean_amp) / (std_amp + 1e-10)
        skew = stats.skew(amplitude)
        kurt = stats.kurtosis(amplitude)
        
        return {
            'mean_amplitude': float(mean_amp),
            'std_amplitude': float(std_amp),
            'max_amplitude': float(max_amp),
            'min_amplitude': float(min_amp),
            'rms': float(rms),
            'peak_to_average_power': float(pap),
            'crest_factor': float(crest),
            'skewness': float(skew),
            'kurtosis': float(kurt)
        }

    def _extract_frequency_features(self, sig: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features."""
        # FFT computation
        fft_result = fft(sig, n=self.fft_size)
        magnitude = np.abs(fft_result)
        power_spectrum = magnitude ** 2
        freq = fftfreq(self.fft_size, 1 / self.sample_rate)
        
        # Normalize
        power_spectrum = power_spectrum / (np.max(power_spectrum) + 1e-10)
        
        # Spectral centroid
        centroid = np.sum(freq[:len(freq)//2] * power_spectrum[:len(power_spectrum)//2]) / (
            np.sum(power_spectrum[:len(power_spectrum)//2]) + 1e-10
        )
        
        # Spectral spread
        spread = np.sqrt(
            np.sum(((freq[:len(freq)//2] - centroid) ** 2) * power_spectrum[:len(power_spectrum)//2]) /
            (np.sum(power_spectrum[:len(power_spectrum)//2]) + 1e-10)
        )
        
        # Spectral flatness (Wiener entropy)
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arithmetic_mean = np.mean(power_spectrum)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Spectral entropy
        prob = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        
        # Total spectral power
        total_power = np.sum(power_spectrum)
        
        # Peak frequency
        peak_freq = freq[np.argmax(magnitude)]
        
        return {
            'spectral_centroid': float(centroid),
            'spectral_spread': float(spread),
            'spectral_flatness': float(flatness),
            'spectral_entropy': float(entropy),
            'spectral_power': float(total_power),
            'peak_frequency': float(peak_freq)
        }

    def _extract_phase_features(self, sig: np.ndarray) -> Dict[str, float]:
        """Extract phase and instantaneous frequency features."""
        # Phase information
        phase = np.angle(sig)
        phase_unwrapped = np.unwrap(phase)
        phase_std = np.std(phase)
        phase_mean = np.mean(phase)
        
        # Instantaneous frequency
        phase_diff = np.diff(phase_unwrapped)
        inst_freq = phase_diff * self.sample_rate / (2 * np.pi)
        inst_freq_mean = np.mean(inst_freq)
        inst_freq_std = np.std(inst_freq)
        
        return {
            'phase_std': float(phase_std),
            'phase_mean': float(phase_mean),
            'instantaneous_frequency_mean': float(inst_freq_mean),
            'instantaneous_frequency_std': float(inst_freq_std)
        }

    def _extract_moment_features(self, sig: np.ndarray) -> Dict[str, float]:
        """Extract statistical moment features."""
        amplitude = np.abs(sig)
        mean_amp = np.mean(amplitude)
        std_amp = np.std(amplitude)
        
        # Raw moments
        m2 = np.mean(amplitude ** 2)
        m3 = np.mean(amplitude ** 3)
        m4 = np.mean(amplitude ** 4)
        
        # Normalized moments
        nm2 = m2 / (mean_amp ** 2 + 1e-10)
        nm3 = m3 / (mean_amp ** 3 + 1e-10)
        nm4 = m4 / (mean_amp ** 4 + 1e-10)
        
        return {
            'moment_2': float(m2),
            'moment_3': float(m3),
            'moment_4': float(m4),
            'normalized_moment_2': float(nm2),
            'normalized_moment_3': float(nm3),
            'normalized_moment_4': float(nm4)
        }

    def _extract_autocorrelation_features(self, sig: np.ndarray) -> Dict[str, float]:
        """Extract autocorrelation-based features."""
        amplitude = np.abs(sig)
        
        # Autocorrelation
        autocorr = signal.correlate(amplitude, amplitude, mode='full')
        autocorr = autocorr / np.max(autocorr)
        mid = len(autocorr) // 2
        autocorr = autocorr[mid:]
        
        # Peak of autocorrelation (should be 1 at lag 0)
        autocorr_peak = autocorr[0]
        
        # Slope of autocorrelation decay
        if len(autocorr) > 10:
            slope = (autocorr[1] - autocorr[10]) / 9
        else:
            slope = 0
        
        # Periodicity strength (magnitude of secondary peaks)
        if len(autocorr) > 1:
            secondary_peaks = np.max(autocorr[1:min(100, len(autocorr))])
        else:
            secondary_peaks = 0
        
        return {
            'autocorr_peak': float(autocorr_peak),
            'autocorr_slope': float(slope),
            'autocorr_periodicity_strength': float(secondary_peaks)
        }
