"""
Additional tests for complete radar coverage.
"""

import pytest
import numpy as np

from pydar import RadarSystem, Target, Environment, LinearFMChirp
from pydar.radar import ScanResult


class TestRadarComplete:
    """Complete coverage tests for radar module."""
    
    def test_radar_with_custom_antenna(self):
        """Test radar with custom antenna."""
        from pydar import Antenna
        
        antenna = Antenna(
            gain=35,
            beamwidth_azimuth=2,
            beamwidth_elevation=3,
            sidelobe_level=-20,
            efficiency=0.8
        )
        
        radar = RadarSystem(
            frequency=24e9,
            power=100,
            antenna_gain=35,
            antenna=antenna
        )
        
        # Should use the provided antenna
        assert radar.antenna == antenna
        assert radar.antenna.gain == 35
    
    def test_bandwidth_not_set_error(self):
        """Test SNR calculation without bandwidth."""
        radar = RadarSystem(frequency=10e9, power=1000, antenna_gain=30)
        
        # Initially bandwidth should be None
        assert radar.bandwidth is None
        
        # SNR calculation should raise error
        with pytest.raises(ValueError):
            radar.snr(5000, 10)
    
    def test_scan_with_phase_noise(self):
        """Test scanning with target phase noise."""
        radar = RadarSystem(frequency=10e9, power=1000, antenna_gain=30)
        
        # Create target with phase noise
        target = Target(range=1000, velocity=0, rcs=10)
        target.phase_noise_std = 0.1  # Add phase noise
        
        env = Environment()
        env.add_target(target)
        
        waveform = LinearFMChirp(duration=10e-6, sample_rate=100e6, bandwidth=50e6)
        
        result = radar.scan(env, waveform)
        
        # Should complete without error
        assert len(result.target_info) == 1
    
    def test_scan_result_summary_empty(self):
        """Test ScanResult summary with no targets."""
        radar = RadarSystem(frequency=10e9, power=1000, antenna_gain=30)
        env = Environment()  # Empty environment
        waveform = LinearFMChirp(duration=10e-6, sample_rate=100e6, bandwidth=50e6)
        
        # Create result directly to test summary
        result = radar.scan(env, waveform)
        
        summary = result.summary()
        assert "Number of targets: 0" in summary
        assert "Target Details:" not in summary


class TestTargetComplete:
    """Complete coverage tests for target module."""
    
    def test_target_repr(self):
        """Test target string representation."""
        target = Target(range=5000, velocity=50, rcs=10, azimuth=30, elevation=5)
        
        repr_str = repr(target)
        assert "Target(" in repr_str
        assert "range=5000.0m" in repr_str
        assert "velocity=50.0m/s" in repr_str


class TestProcessingInit:
    """Test processing module initialization."""
    
    def test_processing_imports(self):
        """Test that processing module imports work."""
        from pydar.processing import RangeDopplerProcessor, CFARDetector, SimpleTracker
        
        # Should be able to import these classes
        assert RangeDopplerProcessor is not None
        assert CFARDetector is not None
        assert SimpleTracker is not None
