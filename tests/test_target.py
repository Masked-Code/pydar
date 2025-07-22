"""
Tests for target modeling.
"""

import pytest
import numpy as np

from pydar import Target, TargetCollection
from pydar.target import SwerlingModel, ExtendedTarget, generate_random_targets


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
    
    def test_target_motion(self):
        """Test target motion model."""
        target = Target(range=1000, velocity=100, rcs=1, acceleration=10)
        
        # Update position after 1 second
        target.update_position(1.0)
        
        # Range should increase: r = r0 + v*t + 0.5*a*t^2
        expected_range = 1000 + 100*1 + 0.5*10*1**2
        assert target.range == pytest.approx(expected_range)
        
        # Velocity should increase: v = v0 + a*t
        expected_velocity = 100 + 10*1
        assert target.velocity == pytest.approx(expected_velocity)
    
    def test_swerling_models(self, random_seed):
        """Test Swerling RCS fluctuation models."""
        base_rcs = 10.0
        
        # Swerling 0 (non-fluctuating)
        target0 = Target(range=1000, velocity=0, rcs=base_rcs, 
                        swerling_model=SwerlingModel.SWERLING_0)
        samples0 = target0.get_rcs_sample(100)
        assert np.all(samples0 == base_rcs)
        
        # Swerling 1/2 (exponential)
        target1 = Target(range=1000, velocity=0, rcs=base_rcs,
                        swerling_model=SwerlingModel.SWERLING_1)
        samples1 = target1.get_rcs_sample(1000)
        assert np.mean(samples1) == pytest.approx(base_rcs, rel=0.1)
        assert np.std(samples1) > 0  # Should have variation
        
        # Swerling 3/4 (chi-squared with 4 DOF)
        target3 = Target(range=1000, velocity=0, rcs=base_rcs,
                        swerling_model=SwerlingModel.SWERLING_3)
        samples3 = target3.get_rcs_sample(1000)
        assert np.mean(samples3) == pytest.approx(base_rcs, rel=0.1)
        assert np.std(samples3) < np.std(samples1)  # Less variation than Swerling 1
    
    def test_coordinate_conversion(self):
        """Test spherical to Cartesian coordinate conversion."""
        target = Target(range=1000, velocity=0, rcs=1, azimuth=0, elevation=0)
        
        # At azimuth=0, elevation=0, should be along x-axis
        x, y, z = target.to_cartesian()
        assert x == pytest.approx(1000)
        assert y == pytest.approx(0)
        assert z == pytest.approx(0)
        
        # Test round-trip conversion
        target.from_cartesian(x, y, z)
        assert target.range == pytest.approx(1000)
        assert target.azimuth == pytest.approx(0)
        assert target.elevation == pytest.approx(0)


class TestExtendedTarget:
    """Test ExtendedTarget class."""
    
    def test_extended_target_initialization(self):
        """Test extended target initialization."""
        target = ExtendedTarget(
            range=5000,
            velocity=50,
            rcs=100,
            extent_range=20,
            extent_cross_range=10,
            num_scatterers=5
        )
        
        assert target.extent_range == 20
        assert target.extent_cross_range == 10
        assert target.num_scatterers == 5
    
    def test_scatterer_generation(self, random_seed):
        """Test scatterer generation for extended target."""
        target = ExtendedTarget(
            range=5000,
            velocity=50,
            rcs=100,
            extent_range=20,
            extent_cross_range=10,
            num_scatterers=10
        )
        
        scatterers = target.get_scatterers()
        
        assert len(scatterers) == 10
        
        # Check scatterers are distributed around main target
        ranges = [s.range for s in scatterers]
        assert min(ranges) >= target.range - target.extent_range/2 - 1
        assert max(ranges) <= target.range + target.extent_range/2 + 1
        
        # Total RCS should be approximately conserved
        total_rcs = sum(s.rcs for s in scatterers)
        # Using exponential distribution, the sum can vary significantly
        assert total_rcs == pytest.approx(target.rcs, rel=0.5)


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
    
    def test_collection_filtering(self):
        """Test collection filtering methods."""
        collection = generate_random_targets(
            num_targets=20,
            range_bounds=(100, 10000),
            velocity_bounds=(-100, 100)
        )
        
        # Filter by range
        near_targets = collection.get_by_range(100, 1000)
        assert all(100 <= t.range <= 1000 for t in near_targets)
        
        # Filter by velocity
        fast_targets = collection.get_by_velocity(50, 100)
        assert all(50 <= t.velocity <= 100 for t in fast_targets)
    
    def test_collection_sorting(self):
        """Test collection sorting methods."""
        collection = generate_random_targets(num_targets=10)
        
        # Sort by range
        collection.sort_by_range()
        ranges = [t.range for t in collection]
        assert ranges == sorted(ranges)
        
        # Sort by RCS
        collection.sort_by_rcs(ascending=False)
        rcs_values = [t.rcs for t in collection]
        assert rcs_values == sorted(rcs_values, reverse=True)
    
    def test_batch_update(self):
        """Test batch position update."""
        collection = TargetCollection()
        
        # Add targets with different velocities
        for v in [10, 20, 30]:
            collection.add_target(Target(range=1000, velocity=v, rcs=1))
        
        # Update all positions
        collection.update_all(1.0)
        
        # Check all targets moved
        expected_ranges = [1010, 1020, 1030]
        actual_ranges = [t.range for t in collection]
        assert actual_ranges == pytest.approx(expected_ranges)
