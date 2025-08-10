"""
Tests for CFAR detection algorithms.
"""

import pytest
import numpy as np

from pydar.processing.cfar import (
    CFARDetector, CellAveragingCFAR, OrderedStatisticsCFAR,
    cfar_2d, adaptive_cfar
)


class TestCellAveragingCFAR:
    """Test CA-CFAR detector."""
    
    def test_ca_cfar_initialization(self):
        """Test CA-CFAR initialization."""
        cfar = CellAveragingCFAR(guard_cells=2, training_cells=8, pfa=1e-6)
        
        assert cfar.guard_cells == 2
        assert cfar.training_cells == 8
        assert cfar.pfa == 1e-6
        assert cfar.threshold_factor > 0
    
    def test_ca_cfar_detection(self):
        """Test CA-CFAR detection on synthetic data."""
        cfar = CellAveragingCFAR(guard_cells=2, training_cells=8, pfa=1e-3)
        
        # Create test signal with target
        data = np.random.exponential(1.0, 100)  # Noise
        data[50] = 20.0  # Strong target
        
        detections = cfar.detect(data)
        
        assert detections.dtype == bool
        assert len(detections) == len(data)
        assert detections[50] == True  # Should detect strong target
    
    def test_ca_cfar_noise_only(self):
        """Test CA-CFAR on noise only."""
        cfar = CellAveragingCFAR(guard_cells=2, training_cells=8, pfa=1e-3)
        
        # Pure noise
        np.random.seed(42)
        data = np.random.exponential(1.0, 1000)
        
        detections = cfar.detect(data)
        
        # False alarm rate should be close to PFA
        fa_rate = np.sum(detections) / len(detections)
        assert fa_rate < 0.01  # Should be low


class TestOrderedStatisticsCFAR:
    """Test OS-CFAR detector."""
    
    def test_os_cfar_initialization(self):
        """Test OS-CFAR initialization."""
        cfar = OrderedStatisticsCFAR(
            guard_cells=2, training_cells=8, pfa=1e-6, k=3
        )
        
        assert cfar.k == 3
        assert cfar.threshold_factor > 0
    
    def test_os_cfar_detection(self):
        """Test OS-CFAR detection."""
        cfar = OrderedStatisticsCFAR(
            guard_cells=2, training_cells=8, pfa=1e-3, k=3
        )
        
        # Create test signal with target and interference
        data = np.random.exponential(1.0, 100)
        data[50] = 20.0  # Target
        data[45] = 10.0  # Interference
        
        detections = cfar.detect(data)
        
        assert detections[50] == True  # Should still detect target


class TestCFAR2D:
    """Test 2D CFAR detection."""
    
    def test_cfar_2d_ca(self):
        """Test 2D CA-CFAR."""
        # Create 2D test data
        data = np.random.exponential(1.0, (50, 50))
        data[25, 25] = 50.0  # Strong target
        
        detections = cfar_2d(
            data, guard_cells=2, training_cells=4, pfa=1e-3, method='ca'
        )
        
        assert detections.shape == data.shape
        assert detections.dtype == bool
        assert detections[25, 25] == True
    
    def test_cfar_2d_os(self):
        """Test 2D OS-CFAR."""
        data = np.random.exponential(1.0, (50, 50))
        data[25, 25] = 50.0
        
        detections = cfar_2d(
            data, guard_cells=2, training_cells=4, pfa=1e-3, method='os'
        )
        
        assert detections[25, 25] == True
    
    def test_cfar_2d_invalid_method(self):
        """Test invalid CFAR method."""
        data = np.random.rand(10, 10)
        
        with pytest.raises(ValueError):
            cfar_2d(data, method='invalid')


class TestAdaptiveCFAR:
    """Test adaptive CFAR."""
    
    def test_adaptive_cfar_homogeneous(self):
        """Test adaptive CFAR in homogeneous environment."""
        # Homogeneous noise
        np.random.seed(42)
        data = np.random.exponential(1.0, 100)
        data[50] = 20.0  # Target
        
        detections = adaptive_cfar(
            data, guard_cells=2, training_cells=8, pfa=1e-3
        )
        
        assert detections[50] == True
    
    def test_adaptive_cfar_heterogeneous(self):
        """Test adaptive CFAR in heterogeneous environment."""
        # Create heterogeneous environment
        data = np.concatenate([
            np.random.exponential(1.0, 40),
            np.random.exponential(5.0, 20),  # Different statistics
            np.random.exponential(1.0, 40)
        ])
        data[50] = 50.0  # Target in transition region
        
        detections = adaptive_cfar(
            data, guard_cells=2, training_cells=8, pfa=1e-3
        )
        
        # Should handle transition region better than CA-CFAR
        assert len(detections) == len(data)
