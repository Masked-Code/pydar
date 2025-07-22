"""
Tests for radar system functionality.
"""

import pytest
import numpy as np
from scipy.constants import c

from pydar import RadarSystem, Antenna
from pydar.utils.conversions import db_to_linear, linear_to_db


class TestRadarSystem:
    """Test RadarSystem class."""
    
    def test_radar_initialization(self):
        """Test radar system initialization."""
        radar = RadarSystem(
            frequency=10e9,
            power=1000,
            antenna_gain=30
        )
        
        assert radar.frequency == 10e9
        assert radar.power == 1000
        assert radar.antenna_gain == 30
        assert radar.wavelength == pytest.approx(c / 10e9)
    
    def test_radar_equation(self, basic_radar):
        """Test radar equation calculation."""
        target_range = 10000  # 10 km
        rcs = 10  # 10 m²
        
        pr = basic_radar.radar_equation(target_range, rcs)
        
        # Check received power is positive and reasonable
        assert pr > 0
        assert pr < basic_radar.power  # Received power should be less than transmitted
        
        # Test range dependency (R^4)
        pr2 = basic_radar.radar_equation(2 * target_range, rcs)
        assert pr2 == pytest.approx(pr / 16, rel=1e-6)
    
    def test_snr_calculation(self, basic_radar):
        """Test SNR calculation."""
        basic_radar.bandwidth = 10e6  # Set bandwidth for SNR calc
        
        target_range = 5000
        rcs = 10
        
        snr_db = basic_radar.snr(target_range, rcs)
        
        # SNR should be reasonable for this scenario
        assert -20 < snr_db < 60  # Reasonable range for SNR
    
    def test_resolution_calculations(self, basic_radar):
        """Test range and velocity resolution calculations."""
        # Range resolution
        bandwidth = 50e6
        range_res = basic_radar.range_resolution(bandwidth)
        expected_res = c / (2 * bandwidth)
        assert range_res == pytest.approx(expected_res)
        
        # Doppler/velocity resolution
        obs_time = 0.1  # 100 ms
        doppler_res = basic_radar.doppler_resolution(obs_time)
        assert doppler_res == pytest.approx(1 / obs_time)
        
        vel_res = basic_radar.velocity_resolution(obs_time)
        expected_vel_res = basic_radar.wavelength / (2 * obs_time)
        assert vel_res == pytest.approx(expected_vel_res)
    
    def test_ambiguity_calculations(self, basic_radar):
        """Test ambiguity calculations."""
        prf = 1000  # 1 kHz
        
        # Max unambiguous range
        max_range = basic_radar.max_unambiguous_range(prf)
        expected_range = c / (2 * prf)
        assert max_range == pytest.approx(expected_range)
        
        # Max unambiguous velocity
        max_vel = basic_radar.max_unambiguous_velocity(prf)
        expected_vel = prf * basic_radar.wavelength / 4
        assert max_vel == pytest.approx(expected_vel)
    
    @pytest.mark.slow
    def test_scan_operation(self, basic_radar, test_environment, chirp_waveform, random_seed):
        """Test radar scan operation."""
        result = basic_radar.scan(test_environment, chirp_waveform)
        
        # Check result structure
        assert result.tx_signal.shape == result.rx_signal.shape
        assert len(result.target_info) == len(test_environment.targets)
        
        # Check that received signal contains energy
        assert np.abs(result.rx_signal).max() > 0
        
        # Check target info
        for info in result.target_info:
            assert 'target' in info
            assert 'snr' in info
            assert 'doppler_shift' in info
            assert info['snr'] > -50  # Should detect something


class TestAntenna:
    """Test Antenna class."""
    
    def test_antenna_initialization(self):
        """Test antenna initialization."""
        antenna = Antenna(
            gain=30,
            beamwidth_azimuth=3,
            beamwidth_elevation=3
        )
        
        assert antenna.gain == 30
        assert antenna.beamwidth_azimuth == 3
        assert antenna.beamwidth_elevation == 3
        assert antenna.efficiency == 0.7
    
    def test_antenna_validation(self):
        """Test antenna parameter validation."""
        # Invalid efficiency
        with pytest.raises(ValueError):
            Antenna(gain=30, beamwidth_azimuth=3, beamwidth_elevation=3, efficiency=1.5)
        
        # Invalid gain
        with pytest.raises(ValueError):
            Antenna(gain=-10, beamwidth_azimuth=3, beamwidth_elevation=3)
    
    def test_antenna_pattern(self):
        """Test antenna pattern calculation."""
        antenna = Antenna(gain=30, beamwidth_azimuth=3, beamwidth_elevation=3)
        
        # Test at boresight
        pattern = antenna.pattern(np.array([0]), np.array([0]))
        assert pattern[0] == pytest.approx(1.0)
        
        # Test at 3dB point
        theta_3db = np.radians(antenna.beamwidth_elevation)
        pattern = antenna.pattern(np.array([theta_3db]), np.array([0]))
        # At theta_3db, the pattern should be exp(-2.77) ≈ 0.0627
        assert pattern[0] == pytest.approx(np.exp(-2.77), rel=0.01)
