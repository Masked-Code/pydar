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


@dataclass
class SeaClutter:
    """Sea clutter model."""
    
    sea_state: int = 3  # Sea state (0-7)
    grazing_angle: float = 1.0  # Grazing angle in degrees
    polarization: str = 'HH'  # Polarization (HH, VV, HV, VH)
    
    def reflectivity(self, frequency: float) -> float:
        """
        Calculate sea clutter reflectivity (sigma0).
        
        Args:
            frequency: Radar frequency in Hz
            
        Returns:
            Normalized clutter RCS per unit area (m²/m²)
        """
        # Simplified GIT model
        freq_ghz = frequency / 1e9
        grazing_rad = np.radians(self.grazing_angle)
        
        # Base reflectivity depends on sea state
        sigma0_base = -40 + 3 * self.sea_state  # dB
        
        # Frequency dependence
        freq_factor = 10 * np.log10(freq_ghz / 10)
        
        # Grazing angle dependence
        grazing_factor = 20 * np.log10(np.sin(grazing_rad))
        
        # Polarization factor
        pol_factor = 0
        if self.polarization == 'VV':
            pol_factor = 3  # VV typically 3 dB higher than HH
        elif self.polarization in ['HV', 'VH']:
            pol_factor = -10  # Cross-pol much lower
        
        sigma0_db = sigma0_base + freq_factor + grazing_factor + pol_factor
        sigma0_linear = 10**(sigma0_db / 10)
        
        return sigma0_linear


@dataclass
class LandClutter:
    """Land clutter model."""
    
    terrain_type: str = 'rural'  # rural, urban, forest, desert
    grazing_angle: float = 5.0  # Grazing angle in degrees
    wind_speed: float = 5.0  # Wind speed in m/s (affects vegetation motion)
    
    # Typical clutter reflectivity values (dB m²/m²)
    TERRAIN_SIGMA0 = {
        'rural': -20,
        'urban': -10,
        'forest': -15,
        'desert': -25,
        'farmland': -22,
        'mountains': -12
    }
    
    def reflectivity(self, frequency: float) -> float:
        """
        Calculate land clutter reflectivity.
        
        Args:
            frequency: Radar frequency in Hz
            
        Returns:
            Normalized clutter RCS per unit area (m²/m²)
        """
        # Base reflectivity
        sigma0_base = self.TERRAIN_SIGMA0.get(self.terrain_type, -20)
        
        # Frequency dependence (higher frequency sees more detail)
        freq_ghz = frequency / 1e9
        freq_factor = 5 * np.log10(freq_ghz / 10)
        
        # Grazing angle dependence
        grazing_rad = np.radians(self.grazing_angle)
        grazing_factor = 10 * np.log10(np.sin(grazing_rad))
        
        sigma0_db = sigma0_base + freq_factor + grazing_factor
        sigma0_linear = 10**(sigma0_db / 10)
        
        return sigma0_linear
    
    def doppler_spectrum(self, frequency: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate clutter Doppler spectrum.
        
        Args:
            frequency: Radar frequency in Hz
            
        Returns:
            Tuple of (doppler_frequencies, spectrum_amplitude)
        """
        # Vegetation motion creates Doppler spread
        wavelength = constants.c / frequency
        
        # Maximum Doppler from wind-blown vegetation
        max_doppler = 2 * self.wind_speed / wavelength
        
        # Generate Gaussian spectrum
        doppler_freqs = np.linspace(-3*max_doppler, 3*max_doppler, 100)
        spectrum = np.exp(-0.5 * (doppler_freqs / max_doppler)**2)
        
        return doppler_freqs, spectrum / np.max(spectrum)


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
        self.sea_clutter: Optional[SeaClutter] = None
        self.land_clutter: Optional[LandClutter] = None
        self.interference_sources: List[Dict[str, Any]] = []
    
    def add_target(self, target: Target):
        """Add a target to the environment."""
        self.targets.add_target(target)
    
    def add_sea_clutter(self, sea_state: int, area: Tuple[float, float], 
                        resolution: float = 10.0):
        """
        Add sea clutter to environment.
        
        Args:
            sea_state: Sea state (0-7)
            area: Tuple of (range_extent, azimuth_extent) in meters and degrees
            resolution: Clutter patch resolution in meters
        """
        self.sea_clutter = SeaClutter(sea_state=sea_state)
        
        # Generate clutter patches
        range_extent, azimuth_extent = area
        num_range = int(range_extent / resolution)
        # For azimuth, we need to calculate based on degrees, not meters
        azimuth_resolution = azimuth_extent / max(1, int(azimuth_extent))  # degrees per patch
        num_azimuth = max(1, int(azimuth_extent / azimuth_resolution))
        
        # Add clutter patches as distributed targets
        for i in range(num_range):
            for j in range(num_azimuth):
                patch_range = (i + 0.5) * resolution
                patch_azimuth = (j + 0.5) * azimuth_resolution - azimuth_extent / 2
                
                # Clutter RCS depends on patch area and reflectivity
                patch_area = resolution**2
                patch_rcs = self.sea_clutter.reflectivity(10e9) * patch_area
                
                # Add some randomness
                patch_rcs *= np.random.lognormal(0, 0.5)
                
                clutter_target = Target(
                    range=patch_range,
                    velocity=np.random.normal(0, 1),  # Small Doppler spread
                    rcs=patch_rcs,
                    azimuth=patch_azimuth
                )
                self.add_target(clutter_target)
    
    def add_land_clutter(self, terrain_type: str, area: Tuple[float, float],
                         resolution: float = 10.0):
        """
        Add land clutter to environment.
        
        Args:
            terrain_type: Type of terrain
            area: Tuple of (range_extent, azimuth_extent)
            resolution: Clutter patch resolution in meters
        """
        self.land_clutter = LandClutter(terrain_type=terrain_type)
        
        # Similar to sea clutter but with different characteristics
        range_extent, azimuth_extent = area
        num_range = int(range_extent / resolution)
        # For azimuth, we need to calculate based on degrees, not meters
        azimuth_resolution = azimuth_extent / max(1, int(azimuth_extent))  # degrees per patch
        num_azimuth = max(1, int(azimuth_extent / azimuth_resolution))
        
        for i in range(num_range):
            for j in range(num_azimuth):
                patch_range = (i + 0.5) * resolution
                patch_azimuth = (j + 0.5) * azimuth_resolution - azimuth_extent / 2
                
                patch_area = resolution**2
                patch_rcs = self.land_clutter.reflectivity(10e9) * patch_area
                patch_rcs *= np.random.lognormal(0, 0.3)  # Less variation than sea
                
                # Land clutter has less Doppler spread
                velocity = np.random.normal(0, 0.1)
                
                clutter_target = Target(
                    range=patch_range,
                    velocity=velocity,
                    rcs=patch_rcs,
                    azimuth=patch_azimuth
                )
                self.add_target(clutter_target)
    
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
