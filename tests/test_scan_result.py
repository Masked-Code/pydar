"""
Tests for scan result functionality.
"""

import pytest
import numpy as np
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from pydar import RadarSystem, Target, Environment, LinearFMChirp
from pydar.scan_result import ScanResult
from pydar.utils.conversions import linear_to_db


class TestScanResult:
    """Test ScanResult class."""
    
    @pytest.fixture
    def scan_result(self, basic_radar, test_environment, chirp_waveform):
        """Create a scan result for testing."""
        return basic_radar.scan(test_environment, chirp_waveform)
    
    def test_scan_result_initialization(self, scan_result):
        """Test scan result initialization."""
        assert scan_result.tx_signal is not None
        assert scan_result.rx_signal is not None
        assert scan_result.waveform is not None
        assert scan_result.radar_system is not None
        assert scan_result.environment is not None
        assert len(scan_result.target_info) > 0
    
    def test_summary(self, scan_result):
        """Test summary generation."""
        summary = scan_result.summary()
        
        assert "Radar Scan Summary" in summary
        assert "Waveform:" in summary
        assert "Duration:" in summary
        assert "Target Details:" in summary
        assert "SNR:" in summary
    
    def test_matched_filter(self, scan_result):
        """Test matched filter processing."""
        mf_output = scan_result.matched_filter()
        
        assert mf_output is not None
        assert len(mf_output) == len(scan_result.rx_signal)
        assert mf_output.dtype == complex
        
        # Should cache result
        mf_output2 = scan_result.matched_filter()
        assert np.array_equal(mf_output, mf_output2)
    
    def test_range_profile(self, scan_result):
        """Test range profile extraction."""
        ranges, amplitudes = scan_result.range_profile()
        
        assert len(ranges) == len(amplitudes)
        assert np.all(ranges >= 0)  # Ranges should be positive
        assert np.all(amplitudes >= 0)  # Amplitudes should be positive
        
        # Should cache result
        ranges2, amplitudes2 = scan_result.range_profile()
        assert np.array_equal(ranges, ranges2)
    
    def test_plot_signals(self, scan_result):
        """Test signal plotting."""
        # Should not raise any errors
        scan_result.plot_signals(figsize=(10, 8))
        matplotlib.pyplot.close('all')
    
    def test_plot_range_profile(self, scan_result):
        """Test range profile plotting."""
        # Test with default parameters
        scan_result.plot_range_profile()
        matplotlib.pyplot.close('all')
        
        # Test with max range
        scan_result.plot_range_profile(max_range=15000)
        matplotlib.pyplot.close('all')
        
        # Test linear scale
        scan_result.plot_range_profile(db_scale=False)
        matplotlib.pyplot.close('all')
    
    def test_plot_spectrogram(self, scan_result):
        """Test spectrogram plotting."""
        scan_result.plot_spectrogram(window_size=128, overlap=0.5)
        matplotlib.pyplot.close('all')
    
    def test_save_load_hdf5(self, scan_result):
        """Test HDF5 save/load functionality."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save to HDF5
            scan_result.save_to_hdf5(temp_file)
            
            # Verify file exists
            assert os.path.exists(temp_file)
            
            # Load from HDF5
            loaded = ScanResult.load_from_hdf5(temp_file)
            
            # Verify loaded data
            assert np.array_equal(loaded.tx_signal, scan_result.tx_signal)
            assert np.array_equal(loaded.rx_signal, scan_result.rx_signal)
            assert loaded.sample_rate == scan_result.sample_rate
            assert loaded.duration == scan_result.duration
            assert loaded.noise_power == scan_result.noise_power
            
        finally:
            os.unlink(temp_file)
    
    def test_empty_targets(self, basic_radar, chirp_waveform):
        """Test scan result with no targets."""
        empty_env = Environment()
        result = basic_radar.scan(empty_env, chirp_waveform)
        
        assert len(result.target_info) == 0
        summary = result.summary()
        assert "Number of targets: 0" in summary
