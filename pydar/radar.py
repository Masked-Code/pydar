"""
Core radar system implementation.

This module contains the main RadarSystem class and related components
for simulating radar systems with accurate physics modeling.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from scipy import signal
from scipy.constants import c, k

from .waveforms import Waveform
from .environment import Environment
from .target import Target
from .utils.conversions import db_to_linear, linear_to_db


@dataclass
class Antenna:
    """Represents a radar antenna with its characteristics."""
    
    gain: float  # Antenna gain in dB
    beamwidth_azimuth: float  # Azimuth beamwidth in degrees
    beamwidth_elevation: float  # Elevation beamwidth in degrees
    sidelobe_level: float = -13.0  # Sidelobe level in dB
    efficiency: float = 0.7  # Antenna efficiency (0-1)
    
    def __post_init__(self):
        """Validate antenna parameters."""
        if not 0 < self.efficiency <= 1:
            raise ValueError("Antenna efficiency must be between 0 and 1")
        if self.gain < 0:
            raise ValueError("Antenna gain must be positive (in dB)")
    
    def pattern(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Calculate antenna pattern.
        
        Args:
            theta: Elevation angles in radians
            phi: Azimuth angles in radians
            
        Returns:
            Normalized antenna pattern values
        """
        # Simple Gaussian beam pattern approximation
        theta_3db = np.radians(self.beamwidth_elevation)
        phi_3db = np.radians(self.beamwidth_azimuth)
        
        pattern = np.exp(-2.77 * ((theta/theta_3db)**2 + (phi/phi_3db)**2))
        return pattern


@dataclass
class RadarSystem:
    """Main radar system class implementing radar equation and signal processing."""
    
    frequency: float  # Operating frequency in Hz
    power: float  # Transmit power in Watts
    antenna_gain: float  # Antenna gain in dB
    system_loss: float = 3.0  # System losses in dB
    noise_figure: float = 3.0  # Receiver noise figure in dB
    bandwidth: Optional[float] = None  # Receiver bandwidth in Hz
    antenna: Optional[Antenna] = None
    temperature: float = 290.0  # System temperature in Kelvin
    
    # Derived parameters
    wavelength: float = field(init=False)
    noise_power: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.wavelength = c / self.frequency
        
        # If antenna not provided, create default with specified gain
        if self.antenna is None:
            # Estimate beamwidth from gain (approximation)
            beamwidth = 70 / np.sqrt(db_to_linear(self.antenna_gain))
            self.antenna = Antenna(
                gain=self.antenna_gain,
                beamwidth_azimuth=beamwidth,
                beamwidth_elevation=beamwidth
            )
        
        # Initialize noise_power to avoid AttributeError
        self.noise_power = 0.0
        
        # Calculate noise power if bandwidth is set
        if self.bandwidth is not None:
            self._update_noise_power()
    
    def _update_noise_power(self):
        """Calculate receiver noise power."""
        noise_figure_linear = db_to_linear(self.noise_figure)
        self.noise_power = k * self.temperature * self.bandwidth * noise_figure_linear
    
    def radar_equation(self, target_range: float, rcs: float) -> float:
        """
        Calculate received power using the radar equation.
        
        Args:
            target_range: Distance to target in meters
            rcs: Radar cross section in square meters
            
        Returns:
            Received power in Watts
        """
        # Convert gains and losses to linear scale
        gt = db_to_linear(self.antenna.gain)
        gr = gt  # Assume same antenna for Tx and Rx
        loss = db_to_linear(self.system_loss)
        
        # Radar equation
        pr = (self.power * gt * gr * self.wavelength**2 * rcs) / \
             ((4 * np.pi)**3 * target_range**4 * loss)
        
        return pr
    
    def snr(self, target_range: float, rcs: float) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            target_range: Distance to target in meters
            rcs: Radar cross section in square meters
            
        Returns:
            SNR in dB
        """
        if self.bandwidth is None:
            raise ValueError("Bandwidth must be set to calculate SNR")
        
        # Update noise power if needed
        if self.noise_power == 0.0:
            self._update_noise_power()
        
        pr = self.radar_equation(target_range, rcs)
        snr_linear = pr / self.noise_power
        return linear_to_db(snr_linear)
    
    def range_resolution(self, bandwidth: float) -> float:
        """
        Calculate range resolution.
        
        Args:
            bandwidth: Signal bandwidth in Hz
            
        Returns:
            Range resolution in meters
        """
        return c / (2 * bandwidth)
    
    def doppler_resolution(self, observation_time: float) -> float:
        """
        Calculate Doppler resolution.
        
        Args:
            observation_time: Coherent processing interval in seconds
            
        Returns:
            Doppler resolution in Hz
        """
        return 1 / observation_time
    
    def velocity_resolution(self, observation_time: float) -> float:
        """
        Calculate velocity resolution.
        
        Args:
            observation_time: Coherent processing interval in seconds
            
        Returns:
            Velocity resolution in m/s
        """
        doppler_res = self.doppler_resolution(observation_time)
        return doppler_res * self.wavelength / 2
    
    def max_unambiguous_range(self, prf: float) -> float:
        """
        Calculate maximum unambiguous range.
        
        Args:
            prf: Pulse repetition frequency in Hz
            
        Returns:
            Maximum unambiguous range in meters
        """
        return c / (2 * prf)
    
    def max_unambiguous_velocity(self, prf: float) -> float:
        """
        Calculate maximum unambiguous velocity.
        
        Args:
            prf: Pulse repetition frequency in Hz
            
        Returns:
            Maximum unambiguous velocity in m/s
        """
        return prf * self.wavelength / 4
    
    def scan(self, environment: Environment, waveform: Waveform,
             scan_angles: Optional[Dict[str, Tuple[float, float]]] = None) -> 'ScanResult':
        """
        Perform a radar scan of the environment.
        
        Args:
            environment: Environment containing targets and clutter
            waveform: Waveform to transmit
            scan_angles: Optional dict with 'azimuth' and 'elevation' angle ranges
            
        Returns:
            ScanResult object containing received signals and metadata
        """
        # Set bandwidth if not already set
        if self.bandwidth is None:
            self.bandwidth = waveform.bandwidth
            self._update_noise_power()
        
        # Generate transmit signal
        tx_signal = waveform.generate()
        
        # Initialize received signal
        rx_signal = np.zeros_like(tx_signal, dtype=complex)
        
        # Get all targets from environment
        targets = environment.get_all_targets()
        
        # Process each target
        target_info = []
        for target in targets:
            # Calculate round-trip time
            round_trip_time = 2 * target.range / c
            
            # Calculate Doppler shift
            doppler_shift = 2 * target.velocity * self.frequency / c
            
            # Calculate received power
            pr = self.radar_equation(target.range, target.rcs)
            
            # Generate target return
            # Account for propagation delay and Doppler
            t = waveform.time_vector
            delayed_signal = waveform.generate(t - round_trip_time)
            
            # Apply Doppler shift
            doppler_signal = delayed_signal * np.exp(1j * 2 * np.pi * doppler_shift * t)
            
            # Apply received power scaling
            amplitude = np.sqrt(pr)
            target_return = amplitude * doppler_signal
            
            # Add phase noise if specified
            if hasattr(target, 'phase_noise_std') and target.phase_noise_std > 0:
                phase_noise = np.random.normal(0, target.phase_noise_std, size=len(target_return))
                target_return *= np.exp(1j * phase_noise)
            
            # Add to received signal
            rx_signal += target_return
            
            # Store target info for analysis
            target_info.append({
                'target': target,
                'round_trip_time': round_trip_time,
                'doppler_shift': doppler_shift,
                'received_power': pr,
                'snr': linear_to_db(pr / self.noise_power)
            })
        
        # Add thermal noise
        noise = np.sqrt(self.noise_power / 2) * (
            np.random.normal(size=len(rx_signal)) + 
            1j * np.random.normal(size=len(rx_signal))
        )
        rx_signal += noise
        
        # Create scan result
        from .scan_result import ScanResult
        result = ScanResult(
            tx_signal=tx_signal,
            rx_signal=rx_signal,
            waveform=waveform,
            radar_system=self,
            environment=environment,
            target_info=target_info,
            noise_power=self.noise_power
        )
        
        return result


class ScanResult:
    """Container for radar scan results."""
    
    def __init__(self, tx_signal: np.ndarray, rx_signal: np.ndarray,
                 waveform: Waveform, radar_system: RadarSystem,
                 environment: Environment, target_info: List[Dict[str, Any]],
                 noise_power: float):
        """Initialize scan result."""
        self.tx_signal = tx_signal
        self.rx_signal = rx_signal
        self.waveform = waveform
        self.radar_system = radar_system
        self.environment = environment
        self.target_info = target_info
        self.noise_power = noise_power
        self.sample_rate = waveform.sample_rate
        self.duration = waveform.duration
    
    def summary(self) -> str:
        """Generate summary of scan results."""
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
