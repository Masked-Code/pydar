"""
Scan result container and analysis tools.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .utils.conversions import linear_to_db, time_to_range


class ScanResult:
    """Container for radar scan results with analysis methods."""
    
    def __init__(self, tx_signal: np.ndarray, rx_signal: np.ndarray,
                 waveform: 'Waveform', radar_system: 'RadarSystem',
                 environment: 'Environment', target_info: List[Dict[str, Any]],
                 noise_power: float):
        """
        Initialize scan result.
        
        Args:
            tx_signal: Transmitted signal
            rx_signal: Received signal
            waveform: Waveform used
            radar_system: Radar system used
            environment: Environment scanned
            target_info: List of target information dictionaries
            noise_power: Receiver noise power
        """
        self.tx_signal = tx_signal
        self.rx_signal = rx_signal
        self.waveform = waveform
        self.radar_system = radar_system
        self.environment = environment
        self.target_info = target_info
        self.noise_power = noise_power
        self.sample_rate = waveform.sample_rate
        self.duration = waveform.duration
        
        # Computed results cache
        self._matched_filter_output = None
        self._range_profile = None
        self._doppler_profile = None
    
    def summary(self) -> str:
        """Generate summary of scan results."""
        from .utils.conversions import linear_to_db
        
        lines = [
            "Radar Scan Summary",
            "=" * 50,
            f"Waveform: {self.waveform.__class__.__name__}",
            f"Duration: {self.duration*1e6:.1f} μs",
            f"Bandwidth: {self.waveform.bandwidth/1e6:.1f} MHz",
            f"Center Frequency: {self.radar_system.frequency/1e9:.1f} GHz",
            f"Number of targets: {len(self.target_info)}",
            ""
        ]
        
        if self.target_info:
            lines.append("Target Details:")
            lines.append("-" * 50)
            for i, info in enumerate(self.target_info):
                target = info['target']
                lines.extend([
                    f"Target {i+1}:",
                    f"  Range: {target.range:.1f} m",
                    f"  Velocity: {target.velocity:.1f} m/s",
                    f"  RCS: {linear_to_db(target.rcs):.1f} dBsm",
                    f"  SNR: {info['snr']:.1f} dB",
                    ""
                ])
        
        return "\n".join(lines)
    
    def matched_filter(self) -> np.ndarray:
        """
        Apply matched filter to received signal.
        
        Returns:
            Matched filter output
        """
        if self._matched_filter_output is None:
            # Matched filter is time-reversed complex conjugate of transmitted signal
            matched_filter = np.conj(self.tx_signal[::-1])
            
            # Apply filter via convolution
            from scipy import signal
            self._matched_filter_output = signal.convolve(
                self.rx_signal, matched_filter, mode='same'
            )
        
        return self._matched_filter_output
    
    def range_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute range profile from matched filter output.
        
        Returns:
            Tuple of (ranges, amplitudes)
        """
        if self._range_profile is None:
            # Get matched filter output
            mf_output = self.matched_filter()
            
            # Convert time axis to range
            time_axis = np.arange(len(mf_output)) / self.sample_rate
            ranges = time_to_range(time_axis)
            
            # Compute amplitude
            amplitude = np.abs(mf_output)
            
            self._range_profile = (ranges, amplitude)
        
        return self._range_profile
    
    def plot_signals(self, figsize: Tuple[float, float] = (12, 8)):
        """
        Plot transmitted and received signals.
        
        Args:
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        time_axis = np.arange(len(self.tx_signal)) / self.sample_rate * 1e6  # μs
        
        # Transmitted signal
        axes[0, 0].plot(time_axis, np.real(self.tx_signal))
        axes[0, 0].set_title('Transmitted Signal (Real)')
        axes[0, 0].set_xlabel('Time (μs)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time_axis, np.imag(self.tx_signal))
        axes[0, 1].set_title('Transmitted Signal (Imaginary)')
        axes[0, 1].set_xlabel('Time (μs)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True)
        
        # Received signal
        axes[1, 0].plot(time_axis, np.real(self.rx_signal))
        axes[1, 0].set_title('Received Signal (Real)')
        axes[1, 0].set_xlabel('Time (μs)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(time_axis, np.imag(self.rx_signal))
        axes[1, 1].set_title('Received Signal (Imaginary)')
        axes[1, 1].set_xlabel('Time (μs)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_range_profile(self, max_range: Optional[float] = None,
                           db_scale: bool = True, figsize: Tuple[float, float] = (10, 6)):
        """
        Plot range profile.
        
        Args:
            max_range: Maximum range to display
            db_scale: Plot in dB scale
            figsize: Figure size
        """
        ranges, amplitude = self.range_profile()
        
        # Limit range if specified
        if max_range is not None:
            mask = ranges <= max_range
            ranges = ranges[mask]
            amplitude = amplitude[mask]
        
        plt.figure(figsize=figsize)
        
        if db_scale:
            amplitude_db = linear_to_db(amplitude, min_value=1e-10)
            plt.plot(ranges / 1000, amplitude_db)
            plt.ylabel('Amplitude (dB)')
        else:
            plt.plot(ranges / 1000, amplitude)
            plt.ylabel('Amplitude')
        
        plt.xlabel('Range (km)')
        plt.title('Range Profile')
        plt.grid(True)
        
        # Mark detected targets
        for info in self.target_info:
            target = info['target']
            plt.axvline(x=target.range / 1000, color='r', linestyle='--', 
                        alpha=0.7, label=f'Target @ {target.range/1000:.1f} km')
        
        if self.target_info:
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_spectrogram(self, window_size: int = 256, overlap: float = 0.5,
                         figsize: Tuple[float, float] = (10, 6)):
        """
        Plot spectrogram of received signal.
        
        Args:
            window_size: FFT window size
            overlap: Window overlap fraction
            figsize: Figure size
        """
        from scipy import signal
        
        # Compute spectrogram
        nperseg = window_size
        noverlap = int(overlap * nperseg)
        
        f, t, Sxx = signal.spectrogram(
            self.rx_signal,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False
        )
        
        # Shift frequencies to center at 0
        f_shifted = np.fft.fftshift(f)
        Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
        
        plt.figure(figsize=figsize)
        
        # Convert to dB
        Sxx_db = 10 * np.log10(np.abs(Sxx_shifted) + 1e-10)
        
        plt.pcolormesh(t * 1e6, f_shifted / 1e3, Sxx_db, 
                       shading='auto', cmap='viridis')
        plt.colorbar(label='Power (dB)')
        
        plt.xlabel('Time (μs)')
        plt.ylabel('Frequency (kHz)')
        plt.title('Spectrogram of Received Signal')
        
        # Mark Doppler frequencies of targets
        for info in self.target_info:
            doppler = info['doppler_shift']
            plt.axhline(y=doppler / 1e3, color='r', linestyle='--', 
                        alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        plt.show()
    
    def save_to_hdf5(self, filename: str):
        """
        Save scan results to HDF5 file.
        
        Args:
            filename: Output filename
        """
        import h5py
        
        with h5py.File(filename, 'w') as f:
            # Save signals
            f.create_dataset('tx_signal', data=self.tx_signal)
            f.create_dataset('rx_signal', data=self.rx_signal)
            
            # Save metadata
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['duration'] = self.duration
            f.attrs['noise_power'] = self.noise_power
            f.attrs['radar_frequency'] = self.radar_system.frequency
            f.attrs['radar_power'] = self.radar_system.power
            
            # Save waveform info
            waveform_group = f.create_group('waveform')
            waveform_group.attrs['type'] = self.waveform.__class__.__name__
            waveform_group.attrs['bandwidth'] = self.waveform.bandwidth
            
            # Save target info
            if self.target_info:
                targets_group = f.create_group('targets')
                for i, info in enumerate(self.target_info):
                    target = info['target']
                    target_group = targets_group.create_group(f'target_{i}')
                    target_group.attrs['range'] = target.range
                    target_group.attrs['velocity'] = target.velocity
                    target_group.attrs['rcs'] = target.rcs
                    target_group.attrs['azimuth'] = target.azimuth
                    target_group.attrs['elevation'] = target.elevation
                    target_group.attrs['snr'] = info['snr']
    
    @classmethod
    def load_from_hdf5(cls, filename: str) -> 'ScanResult':
        """
        Load scan results from HDF5 file.
        
        Args:
            filename: Input filename
            
        Returns:
            ScanResult object
        """
        import h5py
        
        with h5py.File(filename, 'r') as f:
            # Load signals
            tx_signal = f['tx_signal'][:]
            rx_signal = f['rx_signal'][:]
            
            # Load metadata
            sample_rate = f.attrs['sample_rate']
            duration = f.attrs['duration']
            noise_power = f.attrs['noise_power']
            
            # Create minimal waveform object for compatibility
            from .waveforms import CustomWaveform
            waveform = CustomWaveform(
                duration=duration,
                sample_rate=sample_rate,
                bandwidth=f['waveform'].attrs['bandwidth'],
                samples=tx_signal
            )
            
            # Note: Full reconstruction would require saving complete objects
            # This is a simplified version for basic analysis
            
            return cls(
                tx_signal=tx_signal,
                rx_signal=rx_signal,
                waveform=waveform,
                radar_system=None,  # Would need to save/load full object
                environment=None,   # Would need to save/load full object
                target_info=[],     # Would need to reconstruct
                noise_power=noise_power
            )
