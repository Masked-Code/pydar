"""
Environment modeling for radar simulation.

This module handles environmental effects including atmosphere,
clutter, and interference.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from scipy import constants

from .target import Target, TargetCollection


@dataclass
class Atmosphere:
    """Atmospheric model for propagation effects."""
    
    temperature: float = 288.15  # Temperature in Kelvin (15°C)
    pressure: float = 101325  # Pressure in Pascal (sea level)
    humidity: float = 0.5  # Relative humidity (0-1)
    rain_rate: float = 0.0  # Rain rate in mm/hr
    
    def attenuation(self, frequency: float, range: float) -> float:
        """
        Calculate atmospheric attenuation.
        
        Args:
            frequency: Radar frequency in Hz
            range: Propagation distance in meters
            
        Returns:
            Attenuation factor (linear, not dB)
        """
        # Simplified ITU-R P.676 model
        freq_ghz = frequency / 1e9
        
        # Oxygen absorption (simplified)
        gamma_o = 0.0
        if freq_ghz < 57:
            gamma_o = 7.19e-3 + 6.09 / (freq_ghz**2 + 0.227) + \
                      4.81 / ((freq_ghz - 57)**2 + 1.50)
        
        # Water vapor absorption (simplified)
        rho = self.humidity * self.pressure / (461.5 * self.temperature)
        gamma_w = 0.067 * rho * freq_ghz**2
        
        # Rain attenuation (simplified ITU-R P.838)
        gamma_r = 0.0
        if self.rain_rate > 0:
            k = 0.0101 * freq_ghz**0.8
            alpha = 1.0
            gamma_r = k * self.rain_rate**alpha
        
        # Total specific attenuation in dB/km
        gamma_total = gamma_o + gamma_w + gamma_r
        
        # Convert to linear attenuation over range
        atten_db = gamma_total * range / 1000  # Convert range to km
        atten_linear = 10**(-atten_db / 10)
        
        return atten_linear
    
    def refractivity(self, height: float = 0.0) -> float:
        """
        Calculate atmospheric refractivity.
        
        Args:
            height: Height above ground in meters
            
        Returns:
            Refractivity in N-units
        """
        # Exponential atmosphere model
        n0 = 315  # Surface refractivity
        h_scale = 7000  # Scale height in meters
        
        return n0 * np.exp(-height / h_scale)


class Environment:
    """Complete radar environment including targets, clutter, and propagation."""
    
    def __init__(self, atmosphere: Optional[Atmosphere] = None):
        """
        Initialize environment.
        
        Args:
            atmosphere: Atmospheric model (creates default if None)
        """
        self.atmosphere = atmosphere or Atmosphere()
        self.targets = TargetCollection()
        self.interference_sources: List[Dict[str, Any]] = []
    
    def add_target(self, target: Target):
        """Add a target to the environment."""
        self.targets.add_target(target)
    
    def add_interference(self, frequency: float, power: float, 
                         direction: Tuple[float, float], bandwidth: float):
        """
        Add an interference source.
        
        Args:
            frequency: Interference center frequency in Hz
            power: Interference power in Watts
            direction: Tuple of (azimuth, elevation) in degrees
            bandwidth: Interference bandwidth in Hz
        """
        self.interference_sources.append({
            'frequency': frequency,
            'power': power,
            'azimuth': direction[0],
            'elevation': direction[1],
            'bandwidth': bandwidth
        })
    
    def get_all_targets(self) -> List[Target]:
        """Get all targets including clutter."""
        return list(self.targets)
    
    def propagation_loss(self, frequency: float, range: float, 
                         elevation: float = 0.0) -> float:
        """
        Calculate total propagation loss.
        
        Args:
            frequency: Radar frequency in Hz
            range: Propagation distance in meters
            elevation: Elevation angle in degrees
            
        Returns:
            Total loss factor (linear, not dB)
        """
        # Free space loss
        wavelength = constants.c / frequency
        free_space_loss = (wavelength / (4 * np.pi * range))**2
        
        # Atmospheric attenuation
        atmos_loss = self.atmosphere.attenuation(frequency, range)
        
        # Total loss
        total_loss = free_space_loss * atmos_loss
        
        return total_loss
    
    def multipath_factor(self, frequency: float, target_height: float,
                         radar_height: float, range: float) -> complex:
        """
        Calculate multipath propagation factor.
        
        Args:
            frequency: Radar frequency in Hz
            target_height: Target height in meters
            radar_height: Radar height in meters
            range: Horizontal range in meters
            
        Returns:
            Complex propagation factor
        """
        wavelength = constants.c / frequency
        
        # Direct path
        direct_range = np.sqrt(range**2 + (target_height - radar_height)**2)
        
        # Reflected path (simplified - assumes flat earth)
        reflected_range = np.sqrt(range**2 + (target_height + radar_height)**2)
        
        # Path difference
        path_diff = reflected_range - direct_range
        
        # Phase difference
        phase_diff = 2 * np.pi * path_diff / wavelength
        
        # Reflection coefficient (simplified - assumes ground reflection)
        reflection_coeff = -0.9
        
        # Multipath factor
        factor = 1 + reflection_coeff * np.exp(1j * phase_diff)
        
        return factor
    
    def save_scenario(self, filename: str):
        """Save environment scenario to file."""
        import json
        
        scenario = {
            'atmosphere': {
                'temperature': self.atmosphere.temperature,
                'pressure': self.atmosphere.pressure,
                'humidity': self.atmosphere.humidity,
                'rain_rate': self.atmosphere.rain_rate
            },
            'targets': [
                {
                    'range': t.range,
                    'velocity': t.velocity,
                    'rcs': t.rcs,
                    'azimuth': t.azimuth,
                    'elevation': t.elevation
                }
                for t in self.targets
            ],
            'interference': self.interference_sources
        }
        
        with open(filename, 'w') as f:
            json.dump(scenario, f, indent=2)
    
    def load_scenario(self, filename: str):
        """Load environment scenario from file."""
        import json
        
        with open(filename, 'r') as f:
            scenario = json.load(f)
        
        # Load atmosphere
        atmos = scenario['atmosphere']
        self.atmosphere = Atmosphere(**atmos)
        
        # Load targets
        self.targets = TargetCollection()
        for t_data in scenario['targets']:
            target = Target(**t_data)
            self.add_target(target)
        
        # Load interference
        self.interference_sources = scenario.get('interference', [])
