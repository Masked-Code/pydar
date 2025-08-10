"""
Unit conversion utilities for radar calculations.
"""

import numpy as np
from typing import Union, Optional


def db_to_linear(db_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert decibel value to linear scale.
    
    Args:
        db_value: Value in decibels
        
    Returns:
        Linear value
    """
    return 10**(db_value / 10)


def linear_to_db(linear_value: Union[float, np.ndarray], 
                 min_value: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Convert linear value to decibels.
    
    Args:
        linear_value: Linear value
        min_value: Minimum value to clamp to (prevents -inf)
        
    Returns:
        Value in decibels
    """
    if min_value is not None:
        linear_value = np.maximum(linear_value, min_value)
    return 10 * np.log10(linear_value)


def dbm_to_watts(dbm_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert dBm to Watts.
    
    Args:
        dbm_value: Power in dBm
        
    Returns:
        Power in Watts
    """
    return 10**((dbm_value - 30) / 10)


def watts_to_dbm(watts: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert Watts to dBm.
    
    Args:
        watts: Power in Watts
        
    Returns:
        Power in dBm
    """
    return 10 * np.log10(watts) + 30


def freq_to_wavelength(frequency: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert frequency to wavelength.
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        Wavelength in meters
    """
    from scipy.constants import c
    return c / frequency


def wavelength_to_freq(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert wavelength to frequency.
    
    Args:
        wavelength: Wavelength in meters
        
    Returns:
        Frequency in Hz
    """
    from scipy.constants import c
    return c / wavelength


def velocity_to_doppler(velocity: Union[float, np.ndarray], 
                        frequency: float) -> Union[float, np.ndarray]:
    """
    Convert radial velocity to Doppler frequency.
    
    Args:
        velocity: Radial velocity in m/s
        frequency: Radar frequency in Hz
        
    Returns:
        Doppler frequency in Hz
    """
    from scipy.constants import c
    return 2 * velocity * frequency / c


def doppler_to_velocity(doppler: Union[float, np.ndarray], 
                        frequency: float) -> Union[float, np.ndarray]:
    """
    Convert Doppler frequency to radial velocity.
    
    Args:
        doppler: Doppler frequency in Hz
        frequency: Radar frequency in Hz
        
    Returns:
        Radial velocity in m/s
    """
    from scipy.constants import c
    return doppler * c / (2 * frequency)


def range_to_time(range_m: Union[float, np.ndarray], 
                  two_way: bool = True) -> Union[float, np.ndarray]:
    """
    Convert range to time delay.
    
    Args:
        range_m: Range in meters
        two_way: If True, calculate round-trip time
        
    Returns:
        Time delay in seconds
    """
    from scipy.constants import c
    factor = 2 if two_way else 1
    return factor * range_m / c


def time_to_range(time_s: Union[float, np.ndarray], 
                  two_way: bool = True) -> Union[float, np.ndarray]:
    """
    Convert time delay to range.
    
    Args:
        time_s: Time delay in seconds
        two_way: If True, assumes round-trip time
        
    Returns:
        Range in meters
    """
    from scipy.constants import c
    factor = 2 if two_way else 1
    return c * time_s / factor


def rcs_to_dbsm(rcs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert RCS from square meters to dBsm.
    
    Args:
        rcs: RCS in square meters
        
    Returns:
        RCS in dBsm
    """
    return linear_to_db(rcs)


def dbsm_to_rcs(dbsm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert RCS from dBsm to square meters.
    
    Args:
        dbsm: RCS in dBsm
        
    Returns:
        RCS in square meters
    """
    return db_to_linear(dbsm)


def snr_improvement(num_pulses: int, duty_cycle: float = 1.0) -> float:
    """
    Calculate SNR improvement from coherent integration.
    
    Args:
        num_pulses: Number of pulses integrated
        duty_cycle: Duty cycle (0-1)
        
    Returns:
        SNR improvement in dB
    """
    # Coherent integration gain
    coherent_gain = linear_to_db(num_pulses)
    
    # Duty cycle loss
    duty_loss = linear_to_db(duty_cycle)
    
    return coherent_gain + duty_loss


def radar_horizon(radar_height: float, target_height: float = 0, 
                  earth_radius: float = 6371000) -> float:
    """
    Calculate radar horizon distance.
    
    Args:
        radar_height: Radar height above ground in meters
        target_height: Target height above ground in meters
        earth_radius: Earth radius in meters
        
    Returns:
        Horizon distance in meters
    """
    # Account for 4/3 Earth radius (atmospheric refraction)
    effective_radius = 4/3 * earth_radius
    
    # Horizon distance for radar
    d_radar = np.sqrt(2 * effective_radius * radar_height)
    
    # Horizon distance for target
    d_target = np.sqrt(2 * effective_radius * target_height)
    
    return d_radar + d_target
