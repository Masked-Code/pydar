"""
Coordinate system conversion utilities.

This module provides functions for converting between different
coordinate systems used in radar applications.
"""

import numpy as np
from typing import Tuple, Union


def spherical_to_cartesian(range_m: Union[float, np.ndarray], 
                          azimuth_deg: Union[float, np.ndarray], 
                          elevation_deg: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        range_m: Range in meters
        azimuth_deg: Azimuth angle in degrees (0° = North, clockwise)
        elevation_deg: Elevation angle in degrees (0° = horizontal)
        
    Returns:
        Tuple of (x, y, z) coordinates in meters
        x: East direction
        y: North direction  
        z: Up direction
    """
    # Convert to radians
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    
    # Convert to Cartesian
    x = range_m * np.cos(el_rad) * np.sin(az_rad)
    y = range_m * np.cos(el_rad) * np.cos(az_rad)
    z = range_m * np.sin(el_rad)
    
    return x, y, z


def cartesian_to_spherical(x: Union[float, np.ndarray], 
                          y: Union[float, np.ndarray], 
                          z: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        x: East coordinate in meters
        y: North coordinate in meters
        z: Up coordinate in meters
        
    Returns:
        Tuple of (range, azimuth, elevation)
        range: Range in meters
        azimuth: Azimuth angle in degrees
        elevation: Elevation angle in degrees
    """
    # Calculate range
    range_m = np.sqrt(x**2 + y**2 + z**2)
    
    # Calculate azimuth (handle special case when x=y=0)
    azimuth_rad = np.arctan2(x, y)
    azimuth_deg = np.degrees(azimuth_rad)
    
    # Calculate elevation
    horizontal_range = np.sqrt(x**2 + y**2)
    elevation_rad = np.arctan2(z, horizontal_range)
    elevation_deg = np.degrees(elevation_rad)
    
    return range_m, azimuth_deg, elevation_deg


def enu_to_ecef(east: Union[float, np.ndarray], 
                north: Union[float, np.ndarray], 
                up: Union[float, np.ndarray],
                lat0_deg: float, lon0_deg: float, h0_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert East-North-Up (ENU) coordinates to Earth-Centered Earth-Fixed (ECEF).
    
    Args:
        east: East coordinate in meters
        north: North coordinate in meters  
        up: Up coordinate in meters
        lat0_deg: Reference latitude in degrees
        lon0_deg: Reference longitude in degrees
        h0_m: Reference height in meters
        
    Returns:
        Tuple of (x, y, z) ECEF coordinates in meters
    """
    # Convert reference point to radians
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    
    # Rotation matrix from ENU to ECEF
    cos_lat = np.cos(lat0)
    sin_lat = np.sin(lat0)
    cos_lon = np.cos(lon0)
    sin_lon = np.sin(lon0)
    
    # Transform
    x = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    y = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    z = cos_lat * north + sin_lat * up
    
    # Add reference point offset (simplified, assumes spherical Earth)
    earth_radius = 6371000  # meters
    ref_radius = earth_radius + h0_m
    
    x0 = ref_radius * cos_lat * cos_lon
    y0 = ref_radius * cos_lat * sin_lon
    z0 = ref_radius * sin_lat
    
    return x + x0, y + y0, z + z0


def ground_range_to_slant_range(ground_range: Union[float, np.ndarray],
                               target_height: Union[float, np.ndarray],
                               radar_height: Union[float, np.ndarray] = 0.0) -> np.ndarray:
    """
    Convert ground range to slant range.
    
    Args:
        ground_range: Horizontal distance in meters
        target_height: Target height above ground in meters
        radar_height: Radar height above ground in meters
        
    Returns:
        Slant range in meters
    """
    height_diff = target_height - radar_height
    slant_range = np.sqrt(ground_range**2 + height_diff**2)
    return slant_range


def slant_range_to_ground_range(slant_range: Union[float, np.ndarray],
                               elevation_deg: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert slant range to ground range.
    
    Args:
        slant_range: Slant range in meters
        elevation_deg: Elevation angle in degrees
        
    Returns:
        Ground range in meters
    """
    elevation_rad = np.radians(elevation_deg)
    ground_range = slant_range * np.cos(elevation_rad)
    return ground_range
