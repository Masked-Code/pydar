"""
Tests for utility functions.
"""

import pytest
import numpy as np
from scipy.constants import c

from pydar.utils.conversions import (
    db_to_linear, linear_to_db, dbm_to_watts, watts_to_dbm,
    freq_to_wavelength, wavelength_to_freq, velocity_to_doppler,
    doppler_to_velocity, range_to_time, time_to_range,
    rcs_to_dbsm, dbsm_to_rcs, snr_improvement, radar_horizon
)


class TestConversions:
    """Test conversion utility functions."""
    
    def test_db_conversions(self):
        """Test dB to linear conversions."""
        # Test scalar
        assert db_to_linear(0) == pytest.approx(1.0)
        assert db_to_linear(10) == pytest.approx(10.0)
        assert db_to_linear(20) == pytest.approx(100.0)
        
        # Test array
        db_values = np.array([0, 10, 20])
        linear_values = db_to_linear(db_values)
        assert np.allclose(linear_values, [1, 10, 100])
        
        # Test inverse
        assert linear_to_db(1) == pytest.approx(0)
        assert linear_to_db(10) == pytest.approx(10)
        assert linear_to_db(100) == pytest.approx(20)
        
        # Test with minimum value
        assert linear_to_db(0, min_value=1e-10) == pytest.approx(-100)
    
    def test_power_conversions(self):
        """Test power conversions between Watts and dBm."""
        # Test known values
        assert dbm_to_watts(0) == pytest.approx(0.001)  # 0 dBm = 1 mW
        assert dbm_to_watts(30) == pytest.approx(1.0)   # 30 dBm = 1 W
        
        # Test inverse
        assert watts_to_dbm(0.001) == pytest.approx(0)
        assert watts_to_dbm(1.0) == pytest.approx(30)
        
        # Test array
        dbm_array = np.array([0, 10, 20, 30])
        watts_array = dbm_to_watts(dbm_array)
        assert watts_to_dbm(watts_array) == pytest.approx(dbm_array)
    
    def test_frequency_wavelength_conversions(self):
        """Test frequency to wavelength conversions."""
        # Test known values
        freq = 10e9  # 10 GHz
        wavelength = freq_to_wavelength(freq)
        assert wavelength == pytest.approx(c / freq)
        assert wavelength == pytest.approx(0.03, rel=1e-3)  # ~3 cm
        
        # Test inverse
        assert wavelength_to_freq(wavelength) == pytest.approx(freq)
        
        # Test array
        freqs = np.array([1e9, 10e9, 100e9])
        wavelengths = freq_to_wavelength(freqs)
        assert np.allclose(wavelength_to_freq(wavelengths), freqs)
    
    def test_velocity_doppler_conversions(self):
        """Test velocity to Doppler conversions."""
        freq = 10e9  # 10 GHz
        velocity = 100  # 100 m/s
        
        doppler = velocity_to_doppler(velocity, freq)
        expected_doppler = 2 * velocity * freq / c
        assert doppler == pytest.approx(expected_doppler)
        
        # Test inverse
        assert doppler_to_velocity(doppler, freq) == pytest.approx(velocity)
        
        # Test array
        velocities = np.array([-100, 0, 100, 200])
        dopplers = velocity_to_doppler(velocities, freq)
        recovered_velocities = doppler_to_velocity(dopplers, freq)
        assert np.allclose(recovered_velocities, velocities)
    
    def test_range_time_conversions(self):
        """Test range to time conversions."""
        range_m = 1500  # 1.5 km
        
        # Two-way time
        two_way_time = range_to_time(range_m, two_way=True)
        expected_time = 2 * range_m / c
        assert two_way_time == pytest.approx(expected_time)
        
        # One-way time
        one_way_time = range_to_time(range_m, two_way=False)
        assert one_way_time == pytest.approx(range_m / c)
        
        # Test inverse
        assert time_to_range(two_way_time, two_way=True) == pytest.approx(range_m)
        assert time_to_range(one_way_time, two_way=False) == pytest.approx(range_m)
    
    def test_rcs_conversions(self):
        """Test RCS conversions."""
        rcs_m2 = 10  # 10 m²
        
        rcs_dbsm = rcs_to_dbsm(rcs_m2)
        assert rcs_dbsm == pytest.approx(10)  # 10 m² = 10 dBsm
        
        # Test inverse
        assert dbsm_to_rcs(rcs_dbsm) == pytest.approx(rcs_m2)
        
        # Test array
        rcs_values = np.array([0.1, 1, 10, 100])
        dbsm_values = rcs_to_dbsm(rcs_values)
        assert np.allclose(dbsm_to_rcs(dbsm_values), rcs_values)
    
    def test_snr_improvement(self):
        """Test SNR improvement calculation."""
        # Coherent integration of 10 pulses
        improvement = snr_improvement(10, duty_cycle=1.0)
        assert improvement == pytest.approx(10)  # 10 log10(10) = 10 dB
        
        # With 50% duty cycle
        improvement = snr_improvement(10, duty_cycle=0.5)
        assert improvement == pytest.approx(10 - 3.01, rel=0.01)  # 10 - 10*log10(2)
    
    def test_radar_horizon(self):
        """Test radar horizon calculation."""
        # Sea level radar, target at sea level
        horizon = radar_horizon(10, 0)  # 10m radar height
        # Approximate formula: sqrt(2 * h * R * 4/3)
        expected = np.sqrt(2 * 10 * 6371000 * 4/3) + 0
        assert horizon == pytest.approx(expected, rel=0.01)
        
        # Both radar and target elevated
        horizon = radar_horizon(100, 100)
        radar_horizon_dist = np.sqrt(2 * 100 * 6371000 * 4/3)
        target_horizon_dist = np.sqrt(2 * 100 * 6371000 * 4/3)
        expected = radar_horizon_dist + target_horizon_dist
        assert horizon == pytest.approx(expected, rel=0.01)
