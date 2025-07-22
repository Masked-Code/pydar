"""
Tests for range-Doppler processing.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from pydar.processing.range_doppler import RangeDopplerProcessor, RangeDopplerMap


class TestRangeDopplerProcessor:
    """Test RangeDopplerProcessor class."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = RangeDopplerProcessor(
            num_pulses=64,
            prf=5000,
            range_bins=256,
            window='hann'
        )
        
        assert processor.num_pulses == 64
        assert processor.prf == 5000
        assert processor.range_bins == 256
        assert processor.window == 'hann'
        assert len(processor.doppler_window) == 64
    
    def test_process_single_pulse_error(self):
        """Test that single pulse raises error."""
        processor = RangeDopplerProcessor(
            num_pulses=64, prf=5000, range_bins=256
        )
        
        # Single pulse should raise error
        data = np.random.randn(256) + 1j * np.random.randn(256)
        
        with pytest.raises(ValueError):
            processor.process(data)
    
    def test_process_multi_pulse(self):
        """Test processing multiple pulses."""
        processor = RangeDopplerProcessor(
            num_pulses=64, prf=5000, range_bins=256
        )
        
        # Create synthetic pulse data
        data = np.random.randn(256, 64) + 1j * np.random.randn(256, 64)
        
        # Add a simulated target
        range_bin = 100
        doppler_bin = 10
        for i in range(64):
            phase = 2 * np.pi * doppler_bin * i / 64
            data[range_bin, i] += 10 * np.exp(1j * phase)
        
        rd_map = processor.process(data)
        
        assert isinstance(rd_map, RangeDopplerMap)
        assert rd_map.data.shape == (256, 64)
        assert rd_map.prf == 5000
    
    def test_process_with_transpose(self):
        """Test processing with transposed data."""
        processor = RangeDopplerProcessor(
            num_pulses=64, prf=5000, range_bins=256
        )
        
        # Create data with wrong orientation
        data = np.random.randn(64, 256) + 1j * np.random.randn(64, 256)
        
        # Should handle transpose automatically
        rd_map = processor.process(data)
        assert rd_map.data.shape == (256, 64)


class TestRangeDopplerMap:
    """Test RangeDopplerMap class."""
    
    @pytest.fixture
    def rd_map(self):
        """Create a test range-Doppler map."""
        # Create synthetic data with lower noise floor
        data = (np.random.randn(128, 64) + 1j * np.random.randn(128, 64)) * 0.1
        
        # Add some strong targets with high SNR
        data[50, 32] += 100  # Zero Doppler target
        data[80, 40] += 80   # Moving target
        
        range_axis = np.arange(128) * 10  # 10m spacing
        doppler_axis = np.linspace(-2500, 2500, 64)  # Hz
        velocity_axis = doppler_axis * 0.03  # m/s (assuming 10 GHz)
        
        return RangeDopplerMap(
            data=data,
            range_axis=range_axis,
            doppler_axis=doppler_axis,
            velocity_axis=velocity_axis,
            prf=5000,
            frequency=10e9
        )
    
    def test_rd_map_initialization(self, rd_map):
        """Test RangeDopplerMap initialization."""
        assert rd_map.data.shape == (128, 64)
        assert len(rd_map.range_axis) == 128
        assert len(rd_map.doppler_axis) == 64
        assert len(rd_map.velocity_axis) == 64
        assert rd_map.magnitude.shape == rd_map.data.shape
        assert rd_map.magnitude_db.shape == rd_map.data.shape
    
    def test_plot_methods(self, rd_map):
        """Test plotting methods."""
        # Test basic plot
        rd_map.plot()
        matplotlib.pyplot.close('all')
        
        # Test with different options
        rd_map.plot(db_scale=False, velocity_scale=False)
        matplotlib.pyplot.close('all')
        
        # Test with color limits
        rd_map.plot(clim=(-40, 0))
        matplotlib.pyplot.close('all')
    
    def test_detect_targets(self, rd_map):
        """Test simple threshold detection."""
        detections = rd_map.detect_targets(threshold_db=20.0)
        
        assert detections.ndim == 2
        assert detections.shape[1] == 2  # (range_idx, doppler_idx) pairs
        assert len(detections) >= 2  # Should detect at least our two targets
    
    def test_extract_target_info(self, rd_map):
        """Test target information extraction."""
        detections = rd_map.detect_targets(threshold_db=20.0)
        targets = rd_map.extract_target_info(detections)
        
        assert len(targets) == len(detections)
        
        for target in targets:
            assert 'range' in target
            assert 'doppler' in target
            assert 'velocity' in target
            assert 'magnitude' in target
            assert 'magnitude_db' in target
            assert 'range_idx' in target
            assert 'doppler_idx' in target
    
    def test_apply_cfar(self, rd_map):
        """Test CFAR detection integration."""
        detections = rd_map.apply_cfar(
            guard_cells=2,
            training_cells=4,
            pfa=1e-3
        )
        
        assert detections.shape == rd_map.data.shape
        assert detections.dtype == bool
