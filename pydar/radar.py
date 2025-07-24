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
class DetectionReturn:
    """Simple detection return data."""
    range: float  # Range in meters
    azimuth: float  # Azimuth in degrees
    elevation: float  # Elevation in degrees
    doppler_shift: float  # Doppler shift in Hz
    power: float  # Received power in Watts
    target_id: str  # Target identifier


class SimpleScanResult:
    """Simple scan result container."""
    
    def __init__(self, returns: List[DetectionReturn]):
        self.returns = returns


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
    
    antenna: Antenna  # Required antenna object
    waveform: Waveform  # Required waveform object
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Radar position [x, y, z] in meters
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Radar velocity [vx, vy, vz] in m/s
    transmit_power: float = 1000.0  # Transmit power in Watts
    noise_figure: float = 3.0  # Receiver noise figure in dB  
    losses: float = 3.0  # System losses in dB
    temperature: float = 290.0  # System temperature in Kelvin
    
    # Antenna pointing angles
    antenna_azimuth: float = 0.0  # Antenna azimuth angle in degrees
    antenna_elevation: float = 0.0  # Antenna elevation angle in degrees
    
    # Derived parameters
    wavelength: float = field(init=False)
    noise_power: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.wavelength = c / self.waveform.center_frequency
        self.frequency = self.waveform.center_frequency
        self.bandwidth = self.waveform.bandwidth
        
        # Convert position to numpy array
        self.position = np.array(self.position)
        
        # Initialize noise_power
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
        loss = db_to_linear(self.losses)
        
        # Radar equation
        pr = (self.transmit_power * gt * gr * self.wavelength**2 * rcs) / \
             ((4 * np.pi)**3 * target_range**4 * loss)
        
        return pr
    
    def calculate_snr(self, target_range: float, rcs: float) -> float:
        """Calculate SNR for compatibility."""
        return self.snr(target_range, rcs)
    
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
    
    def scan(self, environment: Environment) -> 'SimpleScanResult':
        """
        Perform a radar scan of the environment.
        
        Args:
            environment: Environment containing targets and clutter
            
        Returns:
            SimpleScanResult object containing detected targets
        """
        from .utils.coordinates import cartesian_to_spherical
        
        # Get all targets from environment
        if hasattr(environment, 'targets') and hasattr(environment.targets, 'targets'):
            targets = environment.targets.targets
        else:
            targets = []
        
        # Create detection returns
        returns = []
        
        for target in targets:
            # Get target position in cartesian coordinates
            if hasattr(target, 'position'):
                target_pos = np.array(target.position)
            else:
                # Convert from spherical to cartesian
                target_pos = np.array(target.to_cartesian())
            
            # Calculate target position relative to radar
            rel_pos = target_pos - self.position
            
            # Convert to spherical coordinates
            r, az, el = cartesian_to_spherical(rel_pos[0], rel_pos[1], rel_pos[2])
            
            # Check if target is within beam
            az_deg = np.degrees(az)
            el_deg = np.degrees(el)
            
            az_diff = abs(az_deg - self.antenna_azimuth)
            el_diff = abs(el_deg - self.antenna_elevation)
            
            # Handle azimuth wrap-around
            if az_diff > 180:
                az_diff = 360 - az_diff
            
            # Check if within beamwidth
            if az_diff <= self.antenna.beamwidth_azimuth/2 and el_diff <= self.antenna.beamwidth_elevation/2:
                # Calculate received power
                pr = self.radar_equation(r, target.rcs)
                
                # Calculate doppler shift from radial velocity
                # target.velocity is already the radial velocity
                radial_velocity = target.velocity
                doppler_shift = 2 * radial_velocity * self.frequency / c
                
                # Create detection return
                detection = DetectionReturn(
                    range=r,
                    azimuth=az_deg,
                    elevation=el_deg,
                    doppler_shift=doppler_shift,
                    power=pr,
                    target_id=target.target_id if hasattr(target, 'target_id') else f"T{id(target)}"
                )
                returns.append(detection)
        
        # Create simple scan result
        result = SimpleScanResult(returns=returns)
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
