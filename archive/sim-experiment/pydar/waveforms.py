"""
Radar waveform generation module.

This module provides various radar waveform types including
chirps, pulses, and custom waveforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy import signal


class Waveform(ABC):
    """Abstract base class for radar waveforms."""
    
    def __init__(self, duration: float, sample_rate: float):
        """
        Initialize waveform parameters.
        
        Args:
            duration: Waveform duration in seconds
            sample_rate: Sample rate in Hz
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        self.time_vector = np.linspace(0, duration, self.num_samples, endpoint=False)
    
    @abstractmethod
    def generate(self, t: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate waveform samples.
        
        Args:
            t: Optional time vector, uses internal time vector if None
            
        Returns:
            Complex waveform samples
        """
        pass
    
    @property
    @abstractmethod
    def bandwidth(self) -> float:
        """Get waveform bandwidth in Hz."""
        pass
    
    def autocorrelation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute waveform autocorrelation function.
        
        Returns:
            Tuple of (lags, autocorrelation)
        """
        waveform = self.generate()
        autocorr = signal.correlate(waveform, waveform, mode='full')
        lags = signal.correlation_lags(len(waveform), len(waveform), mode='full')
        time_lags = lags / self.sample_rate
        
        return time_lags, autocorr / np.max(np.abs(autocorr))
    
    def ambiguity_function(self, max_delay: float, max_doppler: float,
                           num_delay: int = 100, num_doppler: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute narrowband ambiguity function.
        
        Args:
            max_delay: Maximum delay in seconds
            max_doppler: Maximum Doppler frequency in Hz
            num_delay: Number of delay samples
            num_doppler: Number of Doppler samples
            
        Returns:
            Tuple of (delays, dopplers, ambiguity_function)
        """
        waveform = self.generate()
        
        # Ensure 0 is included in the arrays for even number of points
        if num_delay % 2 == 0:
            delays = np.concatenate([np.linspace(-max_delay, 0, num_delay//2, endpoint=False),
                                   np.linspace(0, max_delay, num_delay//2)])
        else:
            delays = np.linspace(-max_delay, max_delay, num_delay)
            
        if num_doppler % 2 == 0:
            dopplers = np.concatenate([np.linspace(-max_doppler, 0, num_doppler//2, endpoint=False),
                                     np.linspace(0, max_doppler, num_doppler//2)])
        else:
            dopplers = np.linspace(-max_doppler, max_doppler, num_doppler)
        
        ambiguity = np.zeros((num_doppler, num_delay), dtype=complex)
        
        # Reference waveform (no shift)
        waveform_ref = waveform.copy()
        
        for i, doppler in enumerate(dopplers):
            # Apply Doppler shift to the waveform
            doppler_shift = np.exp(1j * 2 * np.pi * doppler * self.time_vector)
            waveform_shifted = waveform * doppler_shift
            
            # Compute cross-correlation
            correlation = signal.correlate(waveform_ref, waveform_shifted, mode='full')
            
            # Get correlation lags in time
            lags = signal.correlation_lags(len(waveform_ref), len(waveform_shifted), mode='full')
            time_lags = lags / self.sample_rate
            
            # Interpolate to desired delay grid
            # Only interpolate within valid range
            valid_mask = (time_lags >= -max_delay) & (time_lags <= max_delay)
            if np.any(valid_mask):
                ambiguity[i, :] = np.interp(delays, time_lags[valid_mask], correlation[valid_mask])
            else:
                ambiguity[i, :] = 0
        
        # Normalize
        ambiguity = np.abs(ambiguity)
        max_val = np.max(ambiguity)
        if max_val > 0:
            ambiguity = ambiguity / max_val
        
        return delays, dopplers, ambiguity


@dataclass
class LinearFMChirp(Waveform):
    """Linear frequency modulated (LFM) chirp waveform."""
    
    def __init__(self, duration: float, sample_rate: float, 
                 bandwidth: float, center_frequency: float = 0.0):
        """
        Initialize LFM chirp.
        
        Args:
            duration: Chirp duration in seconds
            sample_rate: Sample rate in Hz
            bandwidth: Chirp bandwidth in Hz
            center_frequency: Center frequency in Hz (for complex baseband, use 0)
        """
        super().__init__(duration, sample_rate)
        self._bandwidth = bandwidth
        self.center_frequency = center_frequency
        self.chirp_rate = bandwidth / duration
    
    def generate(self, t: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate LFM chirp waveform."""
        if t is None:
            t = self.time_vector
        
        # Instantaneous frequency: f(t) = f0 + kt
        # Phase: phi(t) = 2*pi * (f0*t + 0.5*k*t^2)
        phase = 2 * np.pi * (self.center_frequency * t + 
                             0.5 * self.chirp_rate * t**2)
        
        return np.exp(1j * phase)
    
    @property
    def bandwidth(self) -> float:
        """Get chirp bandwidth."""
        return self._bandwidth
    
    def time_bandwidth_product(self) -> float:
        """Calculate time-bandwidth product."""
        return self.duration * self.bandwidth


@dataclass
class PulseTrain(Waveform):
    """Coherent pulse train waveform."""
    
    def __init__(self, pulse_width: float, prf: float, num_pulses: int,
                 sample_rate: float, pulse_type: str = 'rect'):
        """
        Initialize pulse train.
        
        Args:
            pulse_width: Individual pulse width in seconds
            prf: Pulse repetition frequency in Hz
            num_pulses: Number of pulses in train
            sample_rate: Sample rate in Hz
            pulse_type: Type of pulse ('rect', 'gaussian', 'sinc')
        """
        pri = 1.0 / prf
        duration = num_pulses * pri
        super().__init__(duration, sample_rate)
        
        self.pulse_width = pulse_width
        self.prf = prf
        self.pri = pri
        self.num_pulses = num_pulses
        self.pulse_type = pulse_type
        self._bandwidth = 1.0 / pulse_width  # Approximate bandwidth
    
    def generate(self, t: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate pulse train waveform."""
        if t is None:
            t = self.time_vector
        
        waveform = np.zeros_like(t, dtype=complex)
        
        for i in range(self.num_pulses):
            pulse_start = i * self.pri
            pulse_end = pulse_start + self.pulse_width
            
            # Find samples within pulse
            pulse_mask = (t >= pulse_start) & (t < pulse_end)
            
            if self.pulse_type == 'rect':
                waveform[pulse_mask] = 1.0
            
            elif self.pulse_type == 'gaussian':
                # Gaussian pulse centered in pulse window
                pulse_center = pulse_start + self.pulse_width / 2
                sigma = self.pulse_width / 6  # 99.7% within pulse width
                pulse_t = t[pulse_mask]
                waveform[pulse_mask] = np.exp(-0.5 * ((pulse_t - pulse_center) / sigma)**2)
            
            elif self.pulse_type == 'sinc':
                # Sinc pulse
                pulse_center = pulse_start + self.pulse_width / 2
                pulse_t = t[pulse_mask]
                x = (pulse_t - pulse_center) / self.pulse_width * 2 * np.pi
                waveform[pulse_mask] = np.sinc(x / np.pi)
        
        return waveform
    
    @property
    def bandwidth(self) -> float:
        """Get pulse bandwidth."""
        return self._bandwidth
    
    def duty_cycle(self) -> float:
        """Calculate duty cycle."""
        return self.pulse_width * self.prf


@dataclass
class BarkerCode(Waveform):
    """Barker code phase-coded waveform."""
    
    # Barker codes by length
    BARKER_CODES = {
        2: [1, -1],
        3: [1, 1, -1],
        4: [1, 1, -1, 1],
        5: [1, 1, 1, -1, 1],
        7: [1, 1, 1, -1, -1, 1, -1],
        11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
    }
    
    def __init__(self, code_length: int, chip_width: float, sample_rate: float):
        """
        Initialize Barker code waveform.
        
        Args:
            code_length: Length of Barker code (2, 3, 4, 5, 7, 11, or 13)
            chip_width: Duration of each code chip in seconds
            sample_rate: Sample rate in Hz
        """
        if code_length not in self.BARKER_CODES:
            raise ValueError(f"Invalid Barker code length: {code_length}")
        
        duration = code_length * chip_width
        super().__init__(duration, sample_rate)
        
        self.code_length = code_length
        self.chip_width = chip_width
        self.code = np.array(self.BARKER_CODES[code_length])
        self._bandwidth = 1.0 / chip_width
    
    def generate(self, t: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate Barker code waveform."""
        if t is None:
            t = self.time_vector
        
        waveform = np.zeros_like(t, dtype=complex)
        
        for i, chip in enumerate(self.code):
            chip_start = i * self.chip_width
            chip_end = (i + 1) * self.chip_width
            chip_mask = (t >= chip_start) & (t < chip_end)
            waveform[chip_mask] = chip
        
        return waveform
    
    @property
    def bandwidth(self) -> float:
        """Get code bandwidth."""
        return self._bandwidth


@dataclass
class SteppedFrequency(Waveform):
    """Stepped frequency waveform."""
    
    def __init__(self, num_steps: int, step_duration: float, 
                 start_frequency: float, frequency_step: float,
                 sample_rate: float):
        """
        Initialize stepped frequency waveform.
        
        Args:
            num_steps: Number of frequency steps
            step_duration: Duration of each frequency step in seconds
            start_frequency: Starting frequency in Hz
            frequency_step: Frequency increment between steps in Hz
            sample_rate: Sample rate in Hz
        """
        duration = num_steps * step_duration
        super().__init__(duration, sample_rate)
        
        self.num_steps = num_steps
        self.step_duration = step_duration
        self.start_frequency = start_frequency
        self.frequency_step = frequency_step
        self._bandwidth = num_steps * frequency_step
    
    def generate(self, t: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate stepped frequency waveform."""
        if t is None:
            t = self.time_vector
        
        waveform = np.zeros_like(t, dtype=complex)
        
        for i in range(self.num_steps):
            step_start = i * self.step_duration
            step_end = (i + 1) * self.step_duration
            step_mask = (t >= step_start) & (t < step_end)
            
            frequency = self.start_frequency + i * self.frequency_step
            step_t = t[step_mask]
            waveform[step_mask] = np.exp(1j * 2 * np.pi * frequency * (step_t - step_start))
        
        return waveform
    
    @property
    def bandwidth(self) -> float:
        """Get total bandwidth."""
        return self._bandwidth
    
    def synthetic_bandwidth(self) -> float:
        """Calculate synthetic bandwidth after processing."""
        return self._bandwidth


class CustomWaveform(Waveform):
    """Custom waveform from user-provided samples or function."""
    
    def __init__(self, duration: float, sample_rate: float, 
                 bandwidth: float, generator_func=None, samples=None):
        """
        Initialize custom waveform.
        
        Args:
            duration: Waveform duration in seconds
            sample_rate: Sample rate in Hz
            bandwidth: Waveform bandwidth in Hz
            generator_func: Function that takes time vector and returns samples
            samples: Pre-computed waveform samples
        """
        super().__init__(duration, sample_rate)
        
        if generator_func is None and samples is None:
            raise ValueError("Either generator_func or samples must be provided")
        
        self._bandwidth = bandwidth
        self.generator_func = generator_func
        self.samples = samples
    
    def generate(self, t: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate custom waveform."""
        if t is None:
            t = self.time_vector
        
        if self.generator_func is not None:
            return self.generator_func(t)
        else:
            # Interpolate pre-computed samples if needed
            if len(self.samples) != len(t):
                sample_times = np.linspace(0, self.duration, len(self.samples))
                real_interp = np.interp(t, sample_times, np.real(self.samples))
                imag_interp = np.interp(t, sample_times, np.imag(self.samples))
                return real_interp + 1j * imag_interp
            else:
                return self.samples.copy()
    
    @property
    def bandwidth(self) -> float:
        """Get waveform bandwidth."""
        return self._bandwidth
