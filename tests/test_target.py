"""
Tests for target modeling.
"""

import pytest
import numpy as np

from pydar import Target, TargetCollection
from pydar.target import Target, TargetCollection
# Adjusted imports to available classes


class TestTarget:
    """Test Target class."""
    
    def test_target_initialization(self):
        """Test target initialization."""
        target = Target(
            range=5000,
            velocity=50,
            rcs=10,
            azimuth=30,
            elevation=5
        )
        
        assert target.range == 5000
        assert target.velocity == 50
        assert target.rcs == 10
        assert target.azimuth == 30
        assert target.elevation == 5
    
    def test_target_validation(self):
        """Test target parameter validation."""
        # Invalid range
        with pytest.raises(ValueError):
            Target(range=-100, velocity=50, rcs=10)
        
        # Invalid RCS
        with pytest.raises(ValueError):
            Target(range=5000, velocity=50, rcs=-1)
    
    # Note: Target motion model with acceleration is planned for future implementation
    
    # Note: Swerling RCS fluctuation models are planned for future implementation
    
    def test_coordinate_conversion(self):
        """Test spherical to Cartesian coordinate conversion."""
        target = Target(range=1000, velocity=0, rcs=1, azimuth=0, elevation=0)
        
        # At azimuth=0, elevation=0, should be along x-axis
        x, y, z = target.to_cartesian()
        assert x == pytest.approx(1000)
        assert y == pytest.approx(0)
        assert z == pytest.approx(0)


# Note: ExtendedTarget class is planned for future implementation


class TestTargetCollection:
    """Test TargetCollection class."""
    
    def test_collection_operations(self):
        """Test basic collection operations."""
        collection = TargetCollection()
        
        # Add targets
        t1 = Target(range=1000, velocity=20, rcs=1)
        t2 = Target(range=5000, velocity=-30, rcs=10)
        t3 = Target(range=10000, velocity=100, rcs=5)
        
        collection.add_target(t1)
        collection.add_target(t2)
        collection.add_target(t3)
        
        assert len(collection) == 3
        assert collection[0] == t1
        
        # Remove target
        collection.remove_target(t2)
        assert len(collection) == 2
    
    # Note: Collection filtering methods are planned for future implementation
    
    # Note: Collection sorting methods are planned for future implementation
    
    # Note: Batch update functionality is planned for future implementation
