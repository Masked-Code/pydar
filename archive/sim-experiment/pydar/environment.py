"""
Environment modeling for radar simulation.

This module handles environmental effects including atmosphere,
clutter, and interference.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
from scipy import constants, stats
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
import warnings

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


# ===================== CLUTTER MODELS =====================

class ClutterModel(ABC):
    """Abstract base class for clutter models."""
    
    @abstractmethod
    def generate_clutter_rcs(self, range_bins: np.ndarray, azimuth_bins: np.ndarray, 
                           frequency: float) -> np.ndarray:
        """
        Generate clutter RCS map.
        
        Args:
            range_bins: Range bins in meters
            azimuth_bins: Azimuth bins in degrees
            frequency: Radar frequency in Hz
            
        Returns:
            2D array of clutter RCS values (range x azimuth)
        """
        pass
    
    @abstractmethod
    def get_clutter_spectrum(self, range_bin: float, azimuth_bin: float, 
                           doppler_bins: np.ndarray) -> np.ndarray:
        """
        Get clutter Doppler spectrum at specific location.
        
        Args:
            range_bin: Range bin in meters
            azimuth_bin: Azimuth bin in degrees
            doppler_bins: Doppler frequency bins in Hz
            
        Returns:
            Power spectral density at each Doppler bin
        """
        pass


@dataclass
class SeaClutterModel(ClutterModel):
    """Advanced sea clutter model based on empirical data."""
    
    wind_speed: float = 10.0  # Wind speed in m/s
    wind_direction: float = 0.0  # Wind direction in degrees
    sea_state: int = 4  # Sea state (0-9)
    grazing_angle: float = 1.0  # Grazing angle in degrees
    polarization: str = 'VV'  # Polarization (VV, HH, VH, HV)
    
    def __post_init__(self):
        """Validate parameters."""
        if not 0 <= self.sea_state <= 9:
            raise ValueError("Sea state must be between 0 and 9")
        if self.polarization not in ['VV', 'HH', 'VH', 'HV']:
            raise ValueError("Polarization must be VV, HH, VH, or HV")
    
    def _georgia_tech_model(self, frequency: float, grazing_angle: float) -> float:
        """
        Calculate clutter coefficient using Georgia Tech model.
        
        Args:
            frequency: Radar frequency in Hz
            grazing_angle: Grazing angle in degrees
            
        Returns:
            Clutter coefficient sigma_0 in dB
        """
        freq_ghz = frequency / 1e9
        psi = np.radians(grazing_angle)
        
        # Base sigma_0 from empirical data
        if self.polarization in ['VV', 'HH']:
            # Co-polarized
            sigma_0_base = -60 + 0.03 * self.wind_speed**2
        else:
            # Cross-polarized
            sigma_0_base = -70 + 0.02 * self.wind_speed**2
        
        # Frequency dependence
        freq_factor = 10 * np.log10(freq_ghz / 10.0)  # Normalized to 10 GHz
        
        # Grazing angle dependence
        if psi < np.radians(1):
            angle_factor = 20 * np.log10(np.sin(psi))
        else:
            angle_factor = 10 * np.log10(np.sin(psi))
        
        # Sea state factor
        sea_factor = 2.0 * (self.sea_state - 4)
        
        sigma_0_db = sigma_0_base + freq_factor + angle_factor + sea_factor
        
        return sigma_0_db
    
    def generate_clutter_rcs(self, range_bins: np.ndarray, azimuth_bins: np.ndarray, 
                           frequency: float) -> np.ndarray:
        """
        Generate sea clutter RCS map.
        
        Args:
            range_bins: Range bins in meters
            azimuth_bins: Azimuth bins in degrees
            frequency: Radar frequency in Hz
            
        Returns:
            2D array of clutter RCS values (range x azimuth)
        """
        rcs_map = np.zeros((len(range_bins), len(azimuth_bins)))
        
        for i, r in enumerate(range_bins):
            # Calculate grazing angle (simplified flat earth)
            grazing = np.degrees(np.arctan(10 / r))  # Assume 10m radar height
            grazing = max(grazing, 0.1)  # Minimum grazing angle
            
            # Get base clutter coefficient
            sigma_0_db = self._georgia_tech_model(frequency, grazing)
            sigma_0_linear = 10**(sigma_0_db / 10)
            
            for j, az in enumerate(azimuth_bins):
                # Calculate resolution cell area
                range_res = 150  # Assume 150m range resolution
                az_res = np.radians(3)  # Assume 3 degree azimuth resolution
                cell_area = range_res * r * az_res
                
                # Wind direction effects
                wind_factor = 1 + 0.1 * np.cos(np.radians(az - self.wind_direction))
                
                # Random variation (Rayleigh distributed)
                random_factor = np.random.rayleigh(1.0)
                
                # Total RCS for this cell
                rcs_map[i, j] = sigma_0_linear * cell_area * wind_factor * random_factor
        
        return rcs_map
    
    def get_clutter_spectrum(self, range_bin: float, azimuth_bin: float, 
                           doppler_bins: np.ndarray) -> np.ndarray:
        """
        Generate sea clutter Doppler spectrum.
        
        Sea clutter typically has a narrow spectrum centered around zero Doppler
        with width proportional to wind speed.
        """
        # Bragg frequency for sea waves
        wavelength = constants.c / 9e9  # Assume X-band
        gravity = 9.81
        bragg_freq = np.sqrt(2 * gravity / wavelength) / (2 * np.pi)
        
        # Main clutter spike at zero Doppler
        doppler_spread = 0.1 * self.wind_speed  # Hz
        main_spectrum = stats.norm.pdf(doppler_bins, 0, doppler_spread)
        
        # Bragg lines (much weaker)
        bragg_spectrum = 0.1 * (stats.norm.pdf(doppler_bins, bragg_freq, doppler_spread/2) +
                               stats.norm.pdf(doppler_bins, -bragg_freq, doppler_spread/2))
        
        return main_spectrum + bragg_spectrum


@dataclass
class LandClutterModel(ClutterModel):
    """Land clutter model for various terrain types."""
    
    terrain_type: str = 'rural'  # 'urban', 'rural', 'forest', 'desert', 'mountains'
    vegetation_density: float = 0.5  # Vegetation density (0-1)
    terrain_roughness: float = 1.0  # Terrain roughness factor
    soil_moisture: float = 0.3  # Soil moisture content (0-1)
    
    def __post_init__(self):
        """Validate parameters."""
        valid_terrain = ['urban', 'rural', 'forest', 'desert', 'mountains']
        if self.terrain_type not in valid_terrain:
            raise ValueError(f"Terrain type must be one of {valid_terrain}")
    
    def _get_terrain_parameters(self) -> Dict[str, float]:
        """Get terrain-specific parameters."""
        terrain_params = {
            'urban': {'base_sigma': -20, 'roughness_factor': 3.0, 'vegetation_factor': 0.1},
            'rural': {'base_sigma': -30, 'roughness_factor': 1.5, 'vegetation_factor': 0.8},
            'forest': {'base_sigma': -25, 'roughness_factor': 2.0, 'vegetation_factor': 1.5},
            'desert': {'base_sigma': -40, 'roughness_factor': 0.5, 'vegetation_factor': 0.0},
            'mountains': {'base_sigma': -35, 'roughness_factor': 4.0, 'vegetation_factor': 0.3}
        }
        return terrain_params[self.terrain_type]
    
    def generate_clutter_rcs(self, range_bins: np.ndarray, azimuth_bins: np.ndarray, 
                           frequency: float) -> np.ndarray:
        """
        Generate land clutter RCS map.
        
        Args:
            range_bins: Range bins in meters
            azimuth_bins: Azimuth bins in degrees
            frequency: Radar frequency in Hz
            
        Returns:
            2D array of clutter RCS values (range x azimuth)
        """
        params = self._get_terrain_parameters()
        rcs_map = np.zeros((len(range_bins), len(azimuth_bins)))
        
        freq_ghz = frequency / 1e9
        
        for i, r in enumerate(range_bins):
            # Grazing angle effects
            grazing = np.degrees(np.arctan(10 / r))  # Assume 10m radar height
            grazing_factor = 10 * np.log10(np.sin(np.radians(max(grazing, 0.1))))
            
            for j, az in enumerate(azimuth_bins):
                # Base clutter coefficient
                sigma_0_db = params['base_sigma']
                
                # Frequency dependence
                freq_factor = 5 * np.log10(freq_ghz / 10.0)
                
                # Vegetation effects
                veg_factor = params['vegetation_factor'] * self.vegetation_density * 5
                
                # Terrain roughness
                rough_factor = params['roughness_factor'] * self.terrain_roughness * 3
                
                # Soil moisture (affects permittivity)
                moisture_factor = 2 * np.log10(1 + self.soil_moisture)
                
                # Total sigma_0
                total_sigma_0 = (sigma_0_db + freq_factor + grazing_factor + 
                               veg_factor + rough_factor + moisture_factor)
                
                # Resolution cell area
                range_res = 150  # Assume 150m range resolution
                az_res = np.radians(3)  # Assume 3 degree azimuth resolution
                cell_area = range_res * r * az_res
                
                # Random variation (log-normal for land clutter)
                random_factor = np.random.lognormal(0, 0.5)
                
                # Convert to RCS
                sigma_0_linear = 10**(total_sigma_0 / 10)
                rcs_map[i, j] = sigma_0_linear * cell_area * random_factor
        
        return rcs_map
    
    def get_clutter_spectrum(self, range_bin: float, azimuth_bin: float, 
                           doppler_bins: np.ndarray) -> np.ndarray:
        """
        Generate land clutter Doppler spectrum.
        
        Land clutter is typically very narrow in Doppler, centered at zero.
        """
        # Very narrow spectrum for stationary land clutter
        doppler_spread = 0.1  # Hz
        
        # Add small amount of spectral spreading due to vegetation movement
        if self.vegetation_density > 0:
            doppler_spread += 0.5 * self.vegetation_density
        
        spectrum = stats.norm.pdf(doppler_bins, 0, doppler_spread)
        
        return spectrum


# ===================== ENHANCED PROPAGATION MODELS =====================

@dataclass
class PropagationModel:
    """Advanced propagation model with ducting and terrain effects."""
    
    def __init__(self, atmosphere: Atmosphere):
        self.atmosphere = atmosphere
        self._setup_ducting_model()
    
    def _setup_ducting_model(self):
        """Setup atmospheric ducting model."""
        # Standard atmosphere profile (height vs refractivity)
        heights = np.array([0, 100, 500, 1000, 2000, 5000, 10000])  # meters
        refractivity = np.array([315, 310, 295, 280, 250, 200, 150])  # N-units
        
        # Create interpolation function
        self.refractivity_profile = interp1d(heights, refractivity, 
                                           kind='linear', fill_value='extrapolate')
    
    def atmospheric_ducting_loss(self, frequency: float, range: float, 
                                height: float, target_height: float) -> float:
        """
        Calculate propagation loss including atmospheric ducting effects.
        
        Args:
            frequency: Radar frequency in Hz
            range: Horizontal range in meters
            height: Radar height in meters
            target_height: Target height in meters
            
        Returns:
            Propagation factor (linear)
        """
        wavelength = constants.c / frequency
        
        # Get refractivity at radar and target heights
        n_radar = self.refractivity_profile(height)
        n_target = self.refractivity_profile(target_height)
        
        # Calculate modified earth radius for ducting
        earth_radius = 6371000  # meters
        dn_dh = (n_target - n_radar) / max(abs(target_height - height), 1)
        
        # Effective earth radius
        if dn_dh < -157:  # Ducting condition
            k_factor = 4/3 / (1 + dn_dh/157)
            effective_radius = k_factor * earth_radius
        else:
            effective_radius = 4/3 * earth_radius  # Standard atmosphere
        
        # Ray bending effects
        ray_path_length = self._calculate_ray_path(range, height, target_height, 
        effective_radius)

        # Free space loss with ray path correction
        free_space_loss = (wavelength / (4 * np.pi * ray_path_length))**2

        # Additional ducting enhancement/loss
        if dn_dh < -157:  # Super-refraction/ducting
            ducting_factor = min(2.0, 1 + abs(dn_dh + 157) / 100)
        else:
            ducting_factor = 1.0

        # Include additional atmospheric scattering effects
        atmos_scattering = self.atmosphere.attenuation(frequency, range)

        return free_space_loss * ducting_factor * atmos_scattering

    def _calculate_ray_path(self, range: float, h1: float, h2: float, 
                          effective_radius: float) -> float:
        """
        Calculate ray path length considering earth curvature and refraction.
        
        Args:
            range: Horizontal range
            h1: Starting height
            h2: Ending height
            effective_radius: Effective earth radius
            
        Returns:
            Ray path length
        """
        # Simplified ray path calculation
        # Account for earth curvature
        range_squared = range**2
        height_diff_squared = (h2 - h1)**2
        curvature_correction = range**4 / (8 * effective_radius**2)
        
        ray_length = np.sqrt(range_squared + height_diff_squared + curvature_correction)
        
        return ray_length
    
    def terrain_shadowing(self, range_profile: np.ndarray, height_profile: np.ndarray,
                         radar_height: float, target_height: float, 
                         target_range: float) -> float:
        """
        Calculate terrain shadowing effects.
        
        Args:
            range_profile: Range points along path
            height_profile: Terrain height profile
            radar_height: Radar height above ground
            target_height: Target height above ground
            target_range: Range to target
            
        Returns:
            Shadowing factor (0 = completely shadowed, 1 = no shadowing)
        """
        if len(range_profile) != len(height_profile):
            raise ValueError("Range and height profiles must have same length")
        
        # Line of sight from radar to target
        radar_abs_height = height_profile[0] + radar_height
        target_abs_height = height_profile[-1] + target_height
        
        # Calculate line of sight elevation at each range point
        los_heights = np.linspace(radar_abs_height, target_abs_height, len(range_profile))
        
        # Check for terrain obstruction
        terrain_clearance = los_heights - height_profile
        min_clearance = np.min(terrain_clearance)
        
        if min_clearance < 0:
            # Complete shadowing
            return 0.0
        elif min_clearance < 100:  # Within first Fresnel zone
            # Partial shadowing using Fresnel zone calculation
            frequency = 10e9  # Assume X-band for calculation
            wavelength = constants.c / frequency
            fresnel_radius = np.sqrt(wavelength * target_range / 4)
            
            if min_clearance < fresnel_radius:
                # Fresnel zone partially blocked
                blocking_factor = min_clearance / fresnel_radius
                return 0.5 + 0.5 * blocking_factor
        
        return 1.0  # No shadowing
    
    def ionospheric_effects(self, frequency: float, range: float, 
                          time_of_day: str = 'day', solar_activity: str = 'medium') -> float:
        """
        Calculate ionospheric propagation effects (mainly for VHF/UHF).
        
        Args:
            frequency: Radar frequency in Hz
            range: Propagation distance in meters
            time_of_day: 'day' or 'night'
            solar_activity: 'low', 'medium', 'high'
            
        Returns:
            Ionospheric propagation factor
        """
        freq_mhz = frequency / 1e6
        
        # Ionospheric effects are negligible above ~3 GHz
        if freq_mhz > 3000:
            return 1.0
        
        # Total Electron Content (TEC) estimation
        tec_values = {
            ('day', 'low'): 20,
            ('day', 'medium'): 50,
            ('day', 'high'): 100,
            ('night', 'low'): 5,
            ('night', 'medium'): 15,
            ('night', 'high'): 30
        }
        
        tec = tec_values.get((time_of_day, solar_activity), 50)
        
        # Faraday rotation and phase delay effects
        # Simplified model - actual ionospheric effects are very complex
        if freq_mhz < 100:  # VHF
            iono_factor = 1 - 0.1 * tec / freq_mhz**2
        elif freq_mhz < 1000:  # UHF
            iono_factor = 1 - 0.01 * tec / freq_mhz**2
        else:  # L-band and above
            iono_factor = 1 - 0.001 * tec / freq_mhz**2
        
        return max(iono_factor, 0.1)  # Prevent negative values


# ===================== ENHANCED ENVIRONMENT CLASS =====================

class EnhancedEnvironment(Environment):
    """Enhanced environment with comprehensive clutter and propagation models."""
    
    def __init__(self, atmosphere: Optional[Atmosphere] = None):
        super().__init__(atmosphere)
        self.clutter_models: List[ClutterModel] = []
        self.propagation_model = PropagationModel(self.atmosphere)
        self.terrain_profile: Optional[Dict[str, np.ndarray]] = None
    
    def add_clutter_model(self, clutter_model: ClutterModel):
        """Add a clutter model to the environment."""
        self.clutter_models.append(clutter_model)
    
    def set_terrain_profile(self, ranges: np.ndarray, heights: np.ndarray, 
                          azimuths: Optional[np.ndarray] = None):
        """
        Set terrain height profile for shadowing calculations.
        
        Args:
            ranges: Range points in meters
            heights: Terrain heights in meters
            azimuths: Optional azimuth angles for 3D terrain
        """
        self.terrain_profile = {
            'ranges': ranges,
            'heights': heights,
            'azimuths': azimuths
        }
    
    def generate_clutter_map(self, range_bins: np.ndarray, azimuth_bins: np.ndarray, 
                           frequency: float) -> np.ndarray:
        """
        Generate combined clutter RCS map from all clutter models.
        
        Args:
            range_bins: Range bins in meters
            azimuth_bins: Azimuth bins in degrees
            frequency: Radar frequency in Hz
            
        Returns:
            Combined clutter RCS map
        """
        if not self.clutter_models:
            warnings.warn("No clutter models defined. Returning zero clutter map.")
            return np.zeros((len(range_bins), len(azimuth_bins)))
        
        total_clutter = np.zeros((len(range_bins), len(azimuth_bins)))
        
        for model in self.clutter_models:
            model_clutter = model.generate_clutter_rcs(range_bins, azimuth_bins, frequency)
            total_clutter += model_clutter
        
        return total_clutter
    
    def calculate_enhanced_propagation_loss(self, frequency: float, target_range: float,
                                         radar_height: float, target_height: float,
                                         azimuth: float = 0.0) -> float:
        """
        Calculate propagation loss with all environmental effects.
        
        Args:
            frequency: Radar frequency in Hz
            target_range: Range to target in meters
            radar_height: Radar height in meters
            target_height: Target height in meters
            azimuth: Azimuth angle in degrees
            
        Returns:
            Total propagation loss factor (linear)
        """
        # Basic atmospheric propagation
        basic_loss = self.propagation_loss(frequency, target_range)
        
        # Atmospheric ducting effects
        ducting_loss = self.propagation_model.atmospheric_ducting_loss(
            frequency, target_range, radar_height, target_height)
        
        # Terrain shadowing
        shadowing_factor = 1.0
        if self.terrain_profile is not None:
            shadowing_factor = self.propagation_model.terrain_shadowing(
                self.terrain_profile['ranges'],
                self.terrain_profile['heights'],
                radar_height, target_height, target_range
            )
        
        # Ionospheric effects (if applicable)
        iono_factor = self.propagation_model.ionospheric_effects(frequency, target_range)
        
        # Combine all effects
        total_loss = ducting_loss * shadowing_factor * iono_factor
        
        return total_loss
    
    def get_clutter_targets(self, range_bins: np.ndarray, azimuth_bins: np.ndarray,
                          frequency: float, min_rcs: float = 0.01) -> TargetCollection:
        """
        Generate clutter targets from clutter models.
        
        Args:
            range_bins: Range bins in meters
            azimuth_bins: Azimuth bins in degrees
            frequency: Radar frequency in Hz
            min_rcs: Minimum RCS threshold for clutter targets
            
        Returns:
            Collection of clutter targets
        """
        clutter_map = self.generate_clutter_map(range_bins, azimuth_bins, frequency)
        clutter_targets = TargetCollection()
        
        for i, r in enumerate(range_bins):
            for j, az in enumerate(azimuth_bins):
                rcs = clutter_map[i, j]
                if rcs >= min_rcs:
                    # Create clutter target
                    clutter_target = Target(
                        range=r,
                        velocity=0.0,  # Stationary clutter
                        rcs=rcs,
                        azimuth=az,
                        elevation=0.0  # Assume ground clutter
                    )
                    clutter_targets.add_target(clutter_target)
        
        return clutter_targets
