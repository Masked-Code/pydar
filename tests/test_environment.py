"""
Tests for environment modeling.
"""

import pytest
import numpy as np
import json
import tempfile
import os

from pydar import Environment, Target
from pydar.environment import Atmosphere, SeaClutter, LandClutter


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


class TestSeaClutter:
    """Test SeaClutter class."""
    
    def test_sea_clutter_initialization(self):
        """Test sea clutter initialization."""
        clutter = SeaClutter()
        assert clutter.sea_state == 3
        assert clutter.grazing_angle == 1.0
        assert clutter.polarization == 'HH'
    
    def test_reflectivity_calculation(self):
        """Test sea clutter reflectivity."""
        clutter = SeaClutter(sea_state=4, grazing_angle=5.0)
        
        sigma0 = clutter.reflectivity(10e9)
        assert sigma0 > 0  # Should be positive
        
        # Test different polarizations with same sea state and grazing angle
        clutter_vv = SeaClutter(sea_state=4, grazing_angle=5.0, polarization='VV')
        sigma0_vv = clutter_vv.reflectivity(10e9)
        
        clutter_hv = SeaClutter(sea_state=4, grazing_angle=5.0, polarization='HV')
        sigma0_hv = clutter_hv.reflectivity(10e9)
        
        # VV should be about 3dB higher than HH (factor of 2)
        assert sigma0_vv > sigma0  # VV typically higher than HH
        assert sigma0_hv < sigma0  # Cross-pol much lower


class TestLandClutter:
    """Test LandClutter class."""
    
    def test_land_clutter_initialization(self):
        """Test land clutter initialization."""
        clutter = LandClutter()
        assert clutter.terrain_type == 'rural'
        assert clutter.grazing_angle == 5.0
        assert clutter.wind_speed == 5.0
    
    def test_terrain_types(self):
        """Test different terrain types."""
        for terrain in ['rural', 'urban', 'forest', 'desert', 'farmland', 'mountains']:
            clutter = LandClutter(terrain_type=terrain)
            sigma0 = clutter.reflectivity(10e9)
            assert sigma0 > 0
    
    def test_doppler_spectrum(self):
        """Test clutter Doppler spectrum generation."""
        clutter = LandClutter(wind_speed=10.0)
        
        freqs, spectrum = clutter.doppler_spectrum(10e9)
        
        assert len(freqs) == len(spectrum)
        assert np.max(spectrum) == pytest.approx(1.0)  # Normalized
        assert np.all(spectrum >= 0)  # All positive


class TestEnvironment:
    """Test Environment class."""
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        env = Environment()
        assert env.atmosphere is not None
        assert len(env.targets) == 0
        assert env.sea_clutter is None
        assert env.land_clutter is None
    
    def test_add_targets(self):
        """Test adding targets to environment."""
        env = Environment()
        
        target1 = Target(range=1000, velocity=50, rcs=10)
        target2 = Target(range=2000, velocity=-30, rcs=5)
        
        env.add_target(target1)
        env.add_target(target2)
        
        assert len(env.targets) == 2
        assert len(env.get_all_targets()) == 2
    
    def test_add_sea_clutter(self):
        """Test adding sea clutter."""
        env = Environment()
        
        initial_targets = len(env.targets)
        env.add_sea_clutter(sea_state=3, area=(1000, 10), resolution=100)
        
        assert env.sea_clutter is not None
        assert env.sea_clutter.sea_state == 3
        assert len(env.targets) > initial_targets  # Should add clutter patches
    
    def test_add_land_clutter(self):
        """Test adding land clutter."""
        env = Environment()
        
        initial_targets = len(env.targets)
        env.add_land_clutter(terrain_type='urban', area=(1000, 10), resolution=100)
        
        assert env.land_clutter is not None
        assert env.land_clutter.terrain_type == 'urban'
        assert len(env.targets) > initial_targets
    
    def test_add_interference(self):
        """Test adding interference sources."""
        env = Environment()
        
        env.add_interference(
            frequency=10e9,
            power=100,
            direction=(45, 10),
            bandwidth=1e6
        )
        
        assert len(env.interference_sources) == 1
        assert env.interference_sources[0]['frequency'] == 10e9
        assert env.interference_sources[0]['azimuth'] == 45
    
    def test_propagation_loss(self):
        """Test propagation loss calculation."""
        env = Environment()
        
        loss = env.propagation_loss(10e9, 10000)
        assert 0 < loss < 1  # Loss factor should be between 0 and 1
        
        # Longer range should have more loss
        loss_far = env.propagation_loss(10e9, 20000)
        assert loss_far < loss
    
    def test_multipath_factor(self):
        """Test multipath propagation factor."""
        env = Environment()
        
        factor = env.multipath_factor(
            frequency=10e9,
            target_height=100,
            radar_height=10,
            range=5000
        )
        
        assert isinstance(factor, complex)
        assert np.abs(factor) <= 2  # Magnitude bounded by direct + reflected
    
    def test_save_load_scenario(self):
        """Test saving and loading scenarios."""
        env = Environment()
        
        # Add some content
        env.add_target(Target(range=1000, velocity=50, rcs=10, azimuth=30, elevation=5))
        env.add_target(Target(range=2000, velocity=-20, rcs=5, azimuth=-10, elevation=0))
        env.add_interference(10e9, 100, (45, 10), 1e6)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            env.save_scenario(temp_file)
            
            # Load into new environment
            new_env = Environment()
            new_env.load_scenario(temp_file)
            
            # Verify content
            assert len(new_env.targets) == 2
            assert new_env.targets[0].range == 1000
            assert len(new_env.interference_sources) == 1
            assert new_env.atmosphere.temperature == env.atmosphere.temperature
        
        finally:
            os.unlink(temp_file)
