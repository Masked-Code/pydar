"""
Tests for environment modeling.
"""

import pytest
import numpy as np
import json
import tempfile
import os

from pydar import Environment, Target
from pydar.environment import Atmosphere
# Removed SeaClutter and LandClutter imports as they're not implemented


class TestAtmosphere:
    """Test Atmosphere class."""
    
    def test_atmosphere_initialization(self):
        """Test atmosphere initialization."""
        atmos = Atmosphere()
        assert atmos.temperature == 288.15
        assert atmos.pressure == 101325
        assert atmos.humidity == 0.5
        assert atmos.rain_rate == 0.0
    
    def test_atmosphere_custom_initialization(self):
        """Test atmosphere with custom parameters."""
        atmos = Atmosphere(
            temperature=300,
            pressure=100000,
            humidity=0.8,
            rain_rate=10.0
        )
        assert atmos.temperature == 300
        assert atmos.rain_rate == 10.0
    
    def test_attenuation_calculation(self):
        """Test atmospheric attenuation calculation."""
        atmos = Atmosphere()
        
        # Test at 10 GHz, 1 km range
        atten = atmos.attenuation(10e9, 1000)
        assert 0 < atten <= 1  # Attenuation factor should be between 0 and 1
        
        # Test with rain
        atmos_rain = Atmosphere(rain_rate=50.0)
        atten_rain = atmos_rain.attenuation(10e9, 1000)
        assert atten_rain < atten  # More attenuation with rain
    
    def test_attenuation_high_frequency(self):
        """Test attenuation at high frequency (>57 GHz)."""
        atmos = Atmosphere()
        # Test at 60 GHz where oxygen absorption is high
        atten = atmos.attenuation(60e9, 1000)
        assert 0 < atten < 1
    
    def test_refractivity(self):
        """Test atmospheric refractivity calculation."""
        atmos = Atmosphere()
        
        # Surface refractivity
        n0 = atmos.refractivity(0)
        assert n0 == 315
        
        # Refractivity at height
        n_height = atmos.refractivity(1000)
        assert n_height < n0  # Should decrease with height


# Note: SeaClutter and LandClutter classes are planned for future implementation


class TestEnvironment:
    """Test Environment class."""
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        env = Environment()
        assert env.atmosphere is not None
        # Check if targets is a TargetCollection
        assert hasattr(env, 'targets')
    
    def test_add_targets(self):
        """Test adding targets to environment."""
        env = Environment()
        
        target1 = Target(range=1000, velocity=50, rcs=10)
        target2 = Target(range=2000, velocity=-30, rcs=5)
        
        env.add_target(target1)
        env.add_target(target2)
        
        # Verify targets were added
        assert len(env.targets.targets) == 2
    
    # Note: Sea and land clutter methods are planned for future implementation
    
    # Note: Interference sources are planned for future implementation
    
    # Note: Propagation loss and multipath calculations are planned for future implementation
    
    # Note: Save/load scenario functionality is planned for future implementation
