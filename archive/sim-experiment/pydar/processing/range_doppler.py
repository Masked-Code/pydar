"""
Range-Doppler processing for radar signals.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift, ifft
import matplotlib.pyplot as plt

from ..utils.conversions import linear_to_db, doppler_to_velocity, time_to_range


class RangeDopplerProcessor:
    """Range-Doppler processing for pulse-Doppler radar."""
    
    def __init__(self, num_pulses: int, prf: float, range_bins: int,
                 window: str = 'hann'):
        """
        Initialize Range-Doppler processor.
        
        Args:
            num_pulses: Number of pulses to process
            prf: Pulse repetition frequency in Hz
            range_bins: Number of range bins
            window: Window function for Doppler processing
        """
        self.num_pulses = num_pulses
        self.prf = prf
        self.range_bins = range_bins
        self.window = window
        
        # Precompute window function
        self.doppler_window = signal.get_window(window, num_pulses)
    
    def process(self, data: np.ndarray, scan_result: Optional['ScanResult'] = None) -> 'RangeDopplerMap':
        """
        Process radar data to create range-Doppler map.
        
        Args:
            data: Input data (can be 1D for single pulse or 2D for multiple)
            scan_result: Optional ScanResult object for metadata
            
        Returns:
            RangeDopplerMap object
        """
        # Reshape data if needed
        if data.ndim == 1:
            # Single pulse - need to collect multiple pulses
            raise ValueError("Range-Doppler processing requires multiple pulses")
        
        # Ensure correct shape (range_bins x num_pulses)
        if data.shape[0] != self.range_bins:
            data = data.T
        
        # Apply matched filter in range dimension if scan_result provided
        if scan_result is not None:
            # Use matched filter from scan result
            matched_filter = np.conj(scan_result.tx_signal[::-1])
            range_compressed = np.zeros_like(data)
            
            for i in range(self.num_pulses):
                range_compressed[:, i] = signal.convolve(
                    data[:, i], matched_filter, mode='same'
                )
        else:
            range_compressed = data
        
        # Apply window in Doppler dimension
        windowed_data = range_compressed * self.doppler_window[np.newaxis, :]
        
        # Doppler FFT
        doppler_processed = fftshift(fft(windowed_data, axis=1), axes=1)
        
        # Create range and Doppler axes
        if scan_result is not None:
            sample_rate = scan_result.sample_rate
            time_axis = np.arange(self.range_bins) / sample_rate
            range_axis = time_to_range(time_axis)
            frequency = scan_result.radar_system.frequency
        else:
            # Default values
            range_axis = np.arange(self.range_bins)
            frequency = 10e9  # Default 10 GHz
        
        # Doppler axis
        doppler_bins = self.num_pulses
        doppler_res = self.prf / doppler_bins
        doppler_axis = np.arange(-doppler_bins//2, doppler_bins//2) * doppler_res
        
        # Convert to velocity
        velocity_axis = doppler_to_velocity(doppler_axis, frequency)
        
        return RangeDopplerMap(
            data=doppler_processed,
            range_axis=range_axis,
            doppler_axis=doppler_axis,
            velocity_axis=velocity_axis,
            prf=self.prf,
            frequency=frequency
        )


class RangeDopplerMap:
    """Container for range-Doppler processing results."""
    
    def __init__(self, data: np.ndarray, range_axis: np.ndarray,
                 doppler_axis: np.ndarray, velocity_axis: np.ndarray,
                 prf: float, frequency: float):
        """
        Initialize range-Doppler map.
        
        Args:
            data: Complex range-Doppler data
            range_axis: Range values in meters
            doppler_axis: Doppler frequencies in Hz
            velocity_axis: Velocity values in m/s
            prf: Pulse repetition frequency
            frequency: Radar frequency
        """
        self.data = data
        self.range_axis = range_axis
        self.doppler_axis = doppler_axis
        self.velocity_axis = velocity_axis
        self.prf = prf
        self.frequency = frequency
        
        # Compute magnitude
        self.magnitude = np.abs(data)
        self.magnitude_db = linear_to_db(self.magnitude, min_value=1e-10)
    
    def plot(self, db_scale: bool = True, velocity_scale: bool = True,
             clim: Optional[Tuple[float, float]] = None,
             figsize: Tuple[float, float] = (10, 8)):
        """
        Plot range-Doppler map.
        
        Args:
            db_scale: Plot in dB scale
            velocity_scale: Use velocity axis instead of Doppler
            clim: Color limits (min, max)
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Choose data to plot
        if db_scale:
            plot_data = self.magnitude_db
            cbar_label = 'Magnitude (dB)'
        else:
            plot_data = self.magnitude
            cbar_label = 'Magnitude'
        
        # Choose vertical axis
        if velocity_scale:
            v_axis = self.velocity_axis
            v_label = 'Velocity (m/s)'
        else:
            v_axis = self.doppler_axis / 1e3  # Convert to kHz
            v_label = 'Doppler (kHz)'
        
        # Create mesh
        extent = [
            self.range_axis[0] / 1000,  # Convert to km
            self.range_axis[-1] / 1000,
            v_axis[0],
            v_axis[-1]
        ]
        
        im = plt.imshow(plot_data.T, aspect='auto', origin='lower',
                        extent=extent, interpolation='nearest')
        
        if clim is not None:
            im.set_clim(clim)
        
        plt.colorbar(im, label=cbar_label)
        plt.xlabel('Range (km)')
        plt.ylabel(v_label)
        plt.title('Range-Doppler Map')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def detect_targets(self, threshold_db: float = 20.0) -> np.ndarray:
        """
        Simple threshold detection.
        
        Args:
            threshold_db: Detection threshold above noise floor
            
        Returns:
            Array of detected target locations (range_idx, doppler_idx)
        """
        # Estimate noise floor (simple method)
        noise_floor = np.median(self.magnitude_db)
        
        # Threshold
        threshold = noise_floor + threshold_db
        
        # Find peaks above threshold
        detections = np.argwhere(self.magnitude_db > threshold)
        
        return detections
    
    def extract_target_info(self, detections: np.ndarray) -> list:
        """
        Extract target information from detections.
        
        Args:
            detections: Array of detection indices
            
        Returns:
            List of target dictionaries
        """
        targets = []
        
        for detection in detections:
            range_idx, doppler_idx = detection
            
            target_info = {
                'range': self.range_axis[range_idx],
                'doppler': self.doppler_axis[doppler_idx],
                'velocity': self.velocity_axis[doppler_idx],
                'magnitude': self.magnitude[range_idx, doppler_idx],
                'magnitude_db': self.magnitude_db[range_idx, doppler_idx],
                'range_idx': range_idx,
                'doppler_idx': doppler_idx
            }
            
            targets.append(target_info)
        
        return targets
    
    def apply_cfar(self, guard_cells: int = 2, training_cells: int = 8,
                   pfa: float = 1e-6) -> np.ndarray:
        """
        Apply 2D CFAR detection.
        
        Args:
            guard_cells: Number of guard cells
            training_cells: Number of training cells
            pfa: Probability of false alarm
            
        Returns:
            Binary detection map
        """
        from .cfar import cfar_2d
        
        # Use magnitude data for CFAR
        detections = cfar_2d(
            self.magnitude,
            guard_cells=guard_cells,
            training_cells=training_cells,
            pfa=pfa
        )
        
        return detections
