"""
Target modeling for radar simulation.

This module provides classes for modeling radar targets with various
characteristics including RCS, motion, and fluctuation models.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import stats


@dataclass
class Target:
    """Represents a radar target with its characteristics."""
    
    range: float  # Range in meters
    velocity: float  # Radial velocity in m/s
    rcs: float  # Radar cross section in square meters
    azimuth: float = 0.0  # Azimuth angle in degrees
    elevation: float = 0.0  # Elevation angle in degrees
    acceleration: float = 0.0  # Radial acceleration in m/s^2
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
            elevation=np.random.uniform(*elevation_bounds)
        )
        collection.add_target(target)
    
    return collection
