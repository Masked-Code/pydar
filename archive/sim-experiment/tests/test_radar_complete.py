"""
Additional tests for complete radar coverage.
"""

import pytest
import numpy as np

from pydar import RadarSystem, Target, Environment, LinearFMChirp
from pydar.radar import SimpleScanResult as ScanResult
# Adjusted import to SimpleScanResult


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
        
        waveform = LinearFMChirp(
            duration=10e-6,
            sample_rate=100e6,
            bandwidth=50e6,
            center_frequency=24e9
        )
        
        radar = RadarSystem(
            antenna=antenna,
            waveform=waveform,
            transmit_power=100
        )
        
        # Should use the provided antenna
        assert radar.antenna == antenna
        assert radar.antenna.gain == 35
    
    # Note: Bandwidth is now part of waveform, always set
    
    # Note: Phase noise modeling is planned for future implementation
    
    def test_scan_result_summary_empty(self, basic_radar):
        """Test scan result with no targets."""
        env = Environment()  # Empty environment
        
        # Create result directly to test
        result = basic_radar.scan(env)
        
        # Check that no targets are detected
        assert len(result.returns) == 0


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
