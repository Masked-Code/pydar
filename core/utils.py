import math
from typing import Tuple
import config


def clamp(value: float, vmin: float, vmax: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: The value to clamp
        vmin: Minimum allowed value
        vmax: Maximum allowed value
        
    Returns:
        The clamped value
    """
    return vmax if value > vmax else vmin if value < vmin else value


def wrap_angle_deg(angle_deg: float) -> float:
    """
    Wrap an angle in degrees to the range [0, 360).
    
    Args:
        angle_deg: Angle in degrees
        
    Returns:
        Angle wrapped to [0, 360) degrees
    """
    return (angle_deg + config.DEG_FULL_CIRCLE) % config.DEG_FULL_CIRCLE


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Args:
        a: Start value
        b: End value  
        t: Interpolation parameter (0.0 = a, 1.0 = b)
        
    Returns:
        Interpolated value
    """
    return a + (b - a) * t


def pol2cart(radius: float, angle_deg: float) -> Tuple[float, float]:
    """
    Convert polar coordinates to Cartesian coordinates.
    
    Args:
        radius: Radial distance
        angle_deg: Angle in degrees
        
    Returns:
        Tuple of (x, y) Cartesian coordinates
    """
    theta = angle_deg * config.DEG_TO_RAD
    return radius * math.cos(theta), radius * math.sin(theta)


def cart2pol(x: float, y: float) -> Tuple[float, float]:
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        
    Returns:
        Tuple of (radius, angle_deg) where angle is in degrees [0, 360)
    """
    radius = math.hypot(x, y)
    angle_deg = math.atan2(y, x) * config.RAD_TO_DEG
    return radius, wrap_angle_deg(angle_deg)
