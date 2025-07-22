"""
Target modeling for radar simulation.

This module provides classes for modeling radar targets with various
characteristics including RCS, motion, and fluctuation models.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from enum import Enum
import numpy as np
from scipy import stats


class SwerlingModel(Enum):
    """Swerling target fluctuation models."""
    SWERLING_0 = 0  # Non-fluctuating
    SWERLING_1 = 1  # Scan-to-scan fluctuation, chi-squared with 2 DOF
    SWERLING_2 = 2  # Pulse-to-pulse fluctuation, chi-squared with 2 DOF
    SWERLING_3 = 3  # Scan-to-scan fluctuation, chi-squared with 4 DOF
    SWERLING_4 = 4  # Pulse-to-pulse fluctuation, chi-squared with 4 DOF


@dataclass
class Target:
    """Represents a radar target with its characteristics."""
    
    range: float  # Range in meters
    velocity: float  # Radial velocity in m/s
    rcs: float  # Radar cross section in square meters
    azimuth: float = 0.0  # Azimuth angle in degrees
    elevation: float = 0.0  # Elevation angle in degrees
    acceleration: float = 0.0  # Radial acceleration in m/s^2
    swerling_model: SwerlingModel = SwerlingModel.SWERLING_0
    phase_noise_std: float = 0.0  # Phase noise standard deviation in radians
    
    # Internal state
    _time: float = field(default=0.0, init=False)
    _initial_range: float = field(init=False)
    _initial_velocity: float = field(init=False)
    
    def __post_init__(self):
        """Store initial values for motion modeling."""
        self._initial_range = self.range
        self._initial_velocity = self.velocity
        
        # Validate parameters
        if self.range < 0:
            raise ValueError("Target range must be positive")
        if self.rcs <= 0:
            raise ValueError("RCS must be positive")
    
    def update_position(self, time: float):
        """
        Update target position based on motion model.
        
        Args:
            time: Time elapsed since start in seconds
        """
        self._time = time
        self.range = self._initial_range + self._initial_velocity * time + 0.5 * self.acceleration * time**2
        self.velocity = self._initial_velocity + self.acceleration * time
    
    def get_rcs_sample(self, num_samples: int = 1) -> Union[float, np.ndarray]:
        """
        Get RCS samples based on Swerling model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            RCS sample(s) in square meters
        """
        if self.swerling_model == SwerlingModel.SWERLING_0:
            # Non-fluctuating
            return self.rcs if num_samples == 1 else np.full(num_samples, self.rcs)
        
        elif self.swerling_model in [SwerlingModel.SWERLING_1, SwerlingModel.SWERLING_2]:
            # Chi-squared with 2 DOF (exponential distribution)
            samples = np.random.exponential(scale=self.rcs, size=num_samples)
            
        else:  # SWERLING_3 or SWERLING_4
            # Chi-squared with 4 DOF
            samples = np.random.gamma(shape=2, scale=self.rcs/2, size=num_samples)
        
        return samples[0] if num_samples == 1 else samples
    
    def to_cartesian(self) -> Tuple[float, float, float]:
        """
        Convert spherical coordinates to Cartesian.
        
        Returns:
            Tuple of (x, y, z) coordinates in meters
        """
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        
        x = self.range * np.cos(el_rad) * np.cos(az_rad)
        y = self.range * np.cos(el_rad) * np.sin(az_rad)
        z = self.range * np.sin(el_rad)
        
        return x, y, z
    
    def from_cartesian(self, x: float, y: float, z: float):
        """
        Set target position from Cartesian coordinates.
        
        Args:
            x, y, z: Cartesian coordinates in meters
        """
        self.range = np.sqrt(x**2 + y**2 + z**2)
        self.azimuth = np.degrees(np.arctan2(y, x))
        self.elevation = np.degrees(np.arcsin(z / self.range))
    
    def __repr__(self) -> str:
        """String representation of target."""
        return (f"Target(range={self.range:.1f}m, velocity={self.velocity:.1f}m/s, "
                f"rcs={self.rcs:.1f}m², az={self.azimuth:.1f}°, el={self.elevation:.1f}°)")


@dataclass
class ExtendedTarget(Target):
    """Extended target with spatial extent."""
    
    extent_range: float = 10.0  # Extent in range dimension (meters)
    extent_cross_range: float = 5.0  # Extent in cross-range dimension (meters)
    num_scatterers: int = 10  # Number of point scatterers
    
    def get_scatterers(self) -> List[Target]:
        """
        Generate point scatterers for extended target.
        
        Returns:
            List of Target objects representing individual scatterers
        """
        scatterers = []
        
        # Generate random positions within target extent
        range_offsets = np.random.uniform(-self.extent_range/2, self.extent_range/2, 
                                          self.num_scatterers)
        cross_range_offsets = np.random.uniform(-self.extent_cross_range/2, 
                                                self.extent_cross_range/2, 
                                                self.num_scatterers)
        
        # RCS distribution among scatterers
        rcs_values = np.random.exponential(scale=self.rcs/self.num_scatterers, 
                                           size=self.num_scatterers)
        
        for i in range(self.num_scatterers):
            # Calculate scatterer position
            scatterer_range = self.range + range_offsets[i]
            
            # Cross-range offset affects azimuth
            az_offset = np.degrees(cross_range_offsets[i] / self.range)
            scatterer_az = self.azimuth + az_offset
            
            scatterer = Target(
                range=scatterer_range,
                velocity=self.velocity,
                rcs=rcs_values[i],
                azimuth=scatterer_az,
                elevation=self.elevation,
                acceleration=self.acceleration,
                swerling_model=self.swerling_model
            )
            scatterers.append(scatterer)
        
        return scatterers


class TargetCollection:
    """Collection of targets for batch processing."""
    
    def __init__(self, targets: Optional[List[Target]] = None):
        """Initialize target collection."""
        self.targets = targets or []
    
    def add_target(self, target: Target):
        """Add a target to the collection."""
        self.targets.append(target)
    
    def remove_target(self, target: Target):
        """Remove a target from the collection."""
        self.targets.remove(target)
    
    def update_all(self, time: float):
        """Update positions of all targets."""
        for target in self.targets:
            target.update_position(time)
    
    def get_by_range(self, min_range: float, max_range: float) -> List[Target]:
        """Get targets within specified range interval."""
        return [t for t in self.targets if min_range <= t.range <= max_range]
    
    def get_by_velocity(self, min_velocity: float, max_velocity: float) -> List[Target]:
        """Get targets within specified velocity interval."""
        return [t for t in self.targets if min_velocity <= t.velocity <= max_velocity]
    
    def get_by_sector(self, az_min: float, az_max: float, 
                      el_min: float = -90, el_max: float = 90) -> List[Target]:
        """Get targets within specified angular sector."""
        targets_in_sector = []
        for t in self.targets:
            # Handle azimuth wrap-around
            az = t.azimuth
            if az_min > az_max:  # Sector crosses 0°
                if az >= az_min or az <= az_max:
                    if el_min <= t.elevation <= el_max:
                        targets_in_sector.append(t)
            else:
                if az_min <= az <= az_max and el_min <= t.elevation <= el_max:
                    targets_in_sector.append(t)
        
        return targets_in_sector
    
    def sort_by_range(self, ascending: bool = True):
        """Sort targets by range."""
        self.targets.sort(key=lambda t: t.range, reverse=not ascending)
    
    def sort_by_rcs(self, ascending: bool = True):
        """Sort targets by RCS."""
        self.targets.sort(key=lambda t: t.rcs, reverse=not ascending)
    
    def __len__(self) -> int:
        """Number of targets in collection."""
        return len(self.targets)
    
    def __iter__(self):
        """Iterate over targets."""
        return iter(self.targets)
    
    def __getitem__(self, index: int) -> Target:
        """Get target by index."""
        return self.targets[index]


def generate_random_targets(num_targets: int, 
                            range_bounds: Tuple[float, float] = (100, 10000),
                            velocity_bounds: Tuple[float, float] = (-100, 100),
                            rcs_bounds: Tuple[float, float] = (0.1, 10),
                            azimuth_bounds: Tuple[float, float] = (-180, 180),
                            elevation_bounds: Tuple[float, float] = (-10, 10)) -> TargetCollection:
    """
    Generate a collection of random targets.
    
    Args:
        num_targets: Number of targets to generate
        range_bounds: Min and max range in meters
        velocity_bounds: Min and max velocity in m/s
        rcs_bounds: Min and max RCS in square meters
        azimuth_bounds: Min and max azimuth in degrees
        elevation_bounds: Min and max elevation in degrees
    
    Returns:
        TargetCollection containing random targets
    """
    collection = TargetCollection()
    
    for _ in range(num_targets):
        target = Target(
            range=np.random.uniform(*range_bounds),
            velocity=np.random.uniform(*velocity_bounds),
            rcs=np.random.uniform(*rcs_bounds),
            azimuth=np.random.uniform(*azimuth_bounds),
            elevation=np.random.uniform(*elevation_bounds),
            swerling_model=np.random.choice(list(SwerlingModel))
        )
        collection.add_target(target)
    
    return collection
