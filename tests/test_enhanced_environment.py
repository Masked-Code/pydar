"""
Tests for enhanced environment functionality including clutter models and propagation effects.
"""

import pytest
import numpy as np
from scipy import constants

from pydar.environment import (
    EnhancedEnvironment, Atmosphere, SeaClutterModel, LandClutterModel,
    PropagationModel
)
from pydar import Target, TargetCollection


class TestSeaClutterModel:
    """Test SeaClutterModel class."""
    
    def test_sea_clutter_initialization(self):
        """Test sea clutter model initialization."""
        model = SeaClutterModel(
            wind_speed=15.0,
            wind_direction=45.0,
            sea_state=5,
            polarization='VV'
        )
        
        assert model.wind_speed == 15.0
        assert model.wind_direction == 45.0
        assert model.sea_state == 5
        assert model.polarization == 'VV'
    
    def test_sea_clutter_validation(self):
        """Test parameter validation."""
        # Invalid sea state
        with pytest.raises(ValueError, match="Sea state must be between 0 and 9"):
            SeaClutterModel(sea_state=10)
        
        # Invalid polarization
        with pytest.raises(ValueError, match="Polarization must be VV, HH, VH, or HV"):
            SeaClutterModel(polarization='XX')
    
    def test_georgia_tech_model(self):
        """Test Georgia Tech clutter model calculation."""
        model = SeaClutterModel(wind_speed=10.0, sea_state=4, polarization='VV')
        
        # Test calculation
        sigma_0 = model._georgia_tech_model(frequency=10e9, grazing_angle=1.0)
        
        # Should return reasonable sigma_0 value in dB
        assert -80 < sigma_0 < -40
    
    def test_generate_clutter_rcs(self):
        """Test clutter RCS map generation."""
        # Set random seed for reproducible results
        np.random.seed(42)
        
        model = SeaClutterModel(wind_speed=10.0, sea_state=4)
        
        range_bins = np.array([1000, 2000, 5000, 10000])
        azimuth_bins = np.array([-10, 0, 10, 20])
        frequency = 10e9
        
        rcs_map = model.generate_clutter_rcs(range_bins, azimuth_bins, frequency)
        
        # Check output shape
        assert rcs_map.shape == (len(range_bins), len(azimuth_bins))
        
        # All values should be positive
        assert np.all(rcs_map > 0)
        
        # Higher ranges should generally have higher RCS due to larger cells
        total_increase = 0
        for _ in range(100):
            test_map = model.generate_clutter_rcs(range_bins, azimuth_bins, frequency)
            test_means = np.mean(test_map, axis=1)
            if test_means[-1] > test_means[0]:
                total_increase += 1
        
        # Check trend in a larger number of samples
        assert total_increase > 50
    
    def test_clutter_spectrum(self):
        """Test clutter spectrum generation."""
        model = SeaClutterModel(wind_speed=10.0)
        
        doppler_bins = np.linspace(-50, 50, 101)
        spectrum = model.get_clutter_spectrum(1000, 0, doppler_bins)
        
        # Spectrum should be positive
        assert np.all(spectrum >= 0)
        
        # Peak should be near zero Doppler
        peak_idx = np.argmax(spectrum)
        peak_doppler = doppler_bins[peak_idx]
        assert abs(peak_doppler) < 5  # Within 5 Hz of zero


class TestLandClutterModel:
    """Test LandClutterModel class."""
    
    def test_land_clutter_initialization(self):
        """Test land clutter model initialization."""
        model = LandClutterModel(
            terrain_type='forest',
            vegetation_density=0.8,
            terrain_roughness=2.0,
            soil_moisture=0.4
        )
        
        assert model.terrain_type == 'forest'
        assert model.vegetation_density == 0.8
        assert model.terrain_roughness == 2.0
        assert model.soil_moisture == 0.4
    
    def test_land_clutter_validation(self):
        """Test parameter validation."""
        # Invalid terrain type
        with pytest.raises(ValueError, match="Terrain type must be one of"):
            LandClutterModel(terrain_type='ocean')
    
    def test_terrain_parameters(self):
        """Test terrain parameter retrieval."""
        model = LandClutterModel(terrain_type='urban')
        params = model._get_terrain_parameters()
        
        assert 'base_sigma' in params
        assert 'roughness_factor' in params
        assert 'vegetation_factor' in params
        
        # Urban should have higher base sigma than desert
        urban_model = LandClutterModel(terrain_type='urban')
        desert_model = LandClutterModel(terrain_type='desert')
        
        urban_params = urban_model._get_terrain_parameters()
        desert_params = desert_model._get_terrain_parameters()
        
        assert urban_params['base_sigma'] > desert_params['base_sigma']
    
    def test_generate_clutter_rcs(self):
        """Test land clutter RCS generation."""
        model = LandClutterModel(terrain_type='rural', vegetation_density=0.5)
        
        range_bins = np.array([500, 1000, 2000, 5000])
        azimuth_bins = np.array([-5, 0, 5])
        frequency = 10e9
        
        rcs_map = model.generate_clutter_rcs(range_bins, azimuth_bins, frequency)
        
        # Check output shape
        assert rcs_map.shape == (len(range_bins), len(azimuth_bins))
        
        # All values should be positive
        assert np.all(rcs_map > 0)
    
    def test_clutter_spectrum(self):
        """Test land clutter spectrum generation."""
        model = LandClutterModel(vegetation_density=0.3)
        
        doppler_bins = np.linspace(-10, 10, 101)
        spectrum = model.get_clutter_spectrum(1000, 0, doppler_bins)
        
        # Spectrum should be positive
        assert np.all(spectrum >= 0)
        
        # Should be very narrow around zero Doppler
        peak_idx = np.argmax(spectrum)
        peak_doppler = doppler_bins[peak_idx]
        assert abs(peak_doppler) < 1  # Within 1 Hz of zero


class TestPropagationModel:
    """Test PropagationModel class."""
    
    def test_propagation_model_initialization(self):
        """Test propagation model initialization."""
        atmosphere = Atmosphere()
        prop_model = PropagationModel(atmosphere)
        
        assert prop_model.atmosphere is atmosphere
        assert hasattr(prop_model, 'refractivity_profile')
    
    def test_atmospheric_ducting_loss(self):
        """Test atmospheric ducting loss calculation."""
        atmosphere = Atmosphere()
        prop_model = PropagationModel(atmosphere)
        
        # Test normal propagation
        loss = prop_model.atmospheric_ducting_loss(
            frequency=10e9,
            range=10000,
            height=100,
            target_height=1000
        )
        
        assert loss > 0
        assert loss < 1  # Should be a loss factor
    
    def test_ray_path_calculation(self):
        """Test ray path length calculation."""
        atmosphere = Atmosphere()
        prop_model = PropagationModel(atmosphere)
        
        # Simple case
        ray_length = prop_model._calculate_ray_path(
            range=10000,
            h1=100,
            h2=200,
            effective_radius=8500000
        )
        
        # Should be slightly longer than straight line distance
        straight_line = np.sqrt(10000**2 + 100**2)
        assert ray_length >= straight_line
    
    def test_terrain_shadowing(self):
        """Test terrain shadowing calculation."""
        atmosphere = Atmosphere()
        prop_model = PropagationModel(atmosphere)
        
        # Clear line of sight
        ranges = np.array([0, 5000, 10000])
        heights = np.array([0, 10, 20])  # Gradually rising terrain
        
        shadow_factor = prop_model.terrain_shadowing(
            range_profile=ranges,
            height_profile=heights,
            radar_height=50,
            target_height=100,
            target_range=10000
        )
        
        assert 0 <= shadow_factor <= 1
        assert shadow_factor > 0.5  # Should have minimal shadowing
        
        # Blocked line of sight
        heights_blocked = np.array([0, 200, 20])  # Hill in the middle
        
        shadow_factor_blocked = prop_model.terrain_shadowing(
            range_profile=ranges,
            height_profile=heights_blocked,
            radar_height=50,
            target_height=100,
            target_range=10000
        )
        
        assert shadow_factor_blocked < shadow_factor  # More shadowing
    
    def test_ionospheric_effects(self):
        """Test ionospheric effects calculation."""
        atmosphere = Atmosphere()
        prop_model = PropagationModel(atmosphere)
        
        # High frequency (X-band) - should have minimal effect
        iono_factor_x = prop_model.ionospheric_effects(
            frequency=10e9,
            range=100000
        )
        assert iono_factor_x == 1.0  # No effect above 3 GHz
        
        # VHF frequency - should have significant effect
        iono_factor_vhf = prop_model.ionospheric_effects(
            frequency=100e6,  # 100 MHz
            range=100000,
            time_of_day='day',
            solar_activity='high'
        )
        assert 0.1 <= iono_factor_vhf < 1.0  # Should have some effect


class TestEnhancedEnvironment:
    """Test EnhancedEnvironment class."""
    
    def test_enhanced_environment_initialization(self):
        """Test enhanced environment initialization."""
        env = EnhancedEnvironment()
        
        assert isinstance(env.atmosphere, Atmosphere)
        assert isinstance(env.clutter_models, list)
        assert len(env.clutter_models) == 0
        assert isinstance(env.propagation_model, PropagationModel)
        assert env.terrain_profile is None
    
    def test_add_clutter_model(self):
        """Test adding clutter models."""
        env = EnhancedEnvironment()
        
        sea_model = SeaClutterModel(wind_speed=10)
        land_model = LandClutterModel(terrain_type='rural')
        
        env.add_clutter_model(sea_model)
        env.add_clutter_model(land_model)
        
        assert len(env.clutter_models) == 2
        assert sea_model in env.clutter_models
        assert land_model in env.clutter_models
    
    def test_set_terrain_profile(self):
        """Test setting terrain profile."""
        env = EnhancedEnvironment()
        
        ranges = np.array([0, 1000, 2000, 3000])
        heights = np.array([0, 50, 30, 100])
        
        env.set_terrain_profile(ranges, heights)
        
        assert env.terrain_profile is not None
        assert np.array_equal(env.terrain_profile['ranges'], ranges)
        assert np.array_equal(env.terrain_profile['heights'], heights)
        assert env.terrain_profile['azimuths'] is None
    
    def test_generate_clutter_map(self):
        """Test clutter map generation."""
        env = EnhancedEnvironment()
        
        # Add clutter models
        sea_model = SeaClutterModel(wind_speed=8)
        land_model = LandClutterModel(terrain_type='forest')
        
        env.add_clutter_model(sea_model)
        env.add_clutter_model(land_model)
        
        range_bins = np.array([1000, 2000, 5000])
        azimuth_bins = np.array([-10, 0, 10])
        frequency = 10e9
        
        clutter_map = env.generate_clutter_map(range_bins, azimuth_bins, frequency)
        
        # Check output shape
        assert clutter_map.shape == (len(range_bins), len(azimuth_bins))
        
        # Should be combination of both models
        assert np.all(clutter_map > 0)
    
    def test_generate_clutter_map_no_models(self):
        """Test clutter map generation with no models."""
        env = EnhancedEnvironment()
        
        range_bins = np.array([1000, 2000])
        azimuth_bins = np.array([0, 10])
        frequency = 10e9
        
        with pytest.warns(UserWarning, match="No clutter models defined"):
            clutter_map = env.generate_clutter_map(range_bins, azimuth_bins, frequency)
        
        # Should return zero map
        assert clutter_map.shape == (len(range_bins), len(azimuth_bins))
        assert np.all(clutter_map == 0)
    
    def test_calculate_enhanced_propagation_loss(self):
        """Test enhanced propagation loss calculation."""
        env = EnhancedEnvironment()
        
        # Set terrain profile
        ranges = np.array([0, 5000, 10000])
        heights = np.array([0, 20, 50])
        env.set_terrain_profile(ranges, heights)
        
        loss = env.calculate_enhanced_propagation_loss(
            frequency=10e9,
            target_range=10000,
            radar_height=100,
            target_height=200,
            azimuth=0
        )
        
        assert loss > 0
        assert loss < 1  # Should be a loss factor
    
    def test_get_clutter_targets(self):
        """Test clutter target generation."""
        env = EnhancedEnvironment()
        
        # Add a land clutter model
        land_model = LandClutterModel(terrain_type='urban')
        env.add_clutter_model(land_model)
        
        range_bins = np.array([1000, 2000])
        azimuth_bins = np.array([0, 10])
        frequency = 10e9
        min_rcs = 1.0  # High threshold to limit number of targets
        
        clutter_targets = env.get_clutter_targets(
            range_bins, azimuth_bins, frequency, min_rcs
        )
        
        assert isinstance(clutter_targets, TargetCollection)
        
        # Check that clutter targets have expected properties
        for target in clutter_targets:
            assert isinstance(target, Target)
            assert target.velocity == 0.0  # Stationary clutter
            assert target.elevation == 0.0  # Ground clutter
            assert target.rcs >= min_rcs


class TestIntegration:
    """Integration tests for enhanced environment functionality."""
    
    def test_comprehensive_environment_scenario(self):
        """Test a comprehensive environment scenario."""
        # Create enhanced environment
        atmosphere = Atmosphere(temperature=290, humidity=0.6, rain_rate=2.0)
        env = EnhancedEnvironment(atmosphere)
        
        # Add clutter models
        sea_model = SeaClutterModel(
            wind_speed=12.0,
            wind_direction=270,
            sea_state=5,
            polarization='VV'
        )
        
        land_model = LandClutterModel(
            terrain_type='rural',
            vegetation_density=0.7,
            terrain_roughness=1.5,
            soil_moisture=0.4
        )
        
        env.add_clutter_model(sea_model)
        env.add_clutter_model(land_model)
        
        # Set terrain profile
        ranges = np.linspace(0, 20000, 21)
        heights = 50 * np.sin(ranges / 5000) + 100  # Rolling hills
        env.set_terrain_profile(ranges, heights)
        
        # Add some targets
        target1 = Target(range=5000, velocity=50, rcs=10, azimuth=0)
        target2 = Target(range=15000, velocity=-30, rcs=5, azimuth=10)
        env.add_target(target1)
        env.add_target(target2)
        
        # Generate clutter map
        range_bins = np.linspace(1000, 20000, 20)
        azimuth_bins = np.linspace(-15, 15, 7)
        frequency = 9.5e9
        
        clutter_map = env.generate_clutter_map(range_bins, azimuth_bins, frequency)
        
        # Verify results
        assert clutter_map.shape == (len(range_bins), len(azimuth_bins))
        assert np.all(clutter_map > 0)
        
        # Test enhanced propagation
        loss = env.calculate_enhanced_propagation_loss(
            frequency=frequency,
            target_range=10000,
            radar_height=50,
            target_height=1000
        )
        
        assert 0 < loss < 1
        
        # Get clutter targets
        clutter_targets = env.get_clutter_targets(
            range_bins, azimuth_bins, frequency, min_rcs=0.1
        )
        
        assert len(clutter_targets) > 0
    
    def test_atmospheric_conditions_effects(self):
        """Test effects of different atmospheric conditions."""
        # Clear weather (dry conditions)
        clear_atmos = Atmosphere(temperature=288, humidity=0.1, rain_rate=0.0)
        
        # Heavy rain (very wet conditions at higher frequency where rain has more effect)
        rainy_atmos = Atmosphere(temperature=285, humidity=0.95, rain_rate=25.0)
        
        clear_env = EnhancedEnvironment(clear_atmos)
        rainy_env = EnhancedEnvironment(rainy_atmos)
        
        # Use higher frequency where atmospheric effects are more pronounced
        frequency = 35e9  # Ka-band where rain attenuation is significant
        target_range = 20000  # Longer range to amplify differences
        
        # Calculate basic atmospheric propagation loss
        clear_loss = clear_env.atmosphere.attenuation(frequency, target_range)
        rainy_loss = rainy_env.atmosphere.attenuation(frequency, target_range)
        
        # Rainy conditions should have higher attenuation (lower attenuation factor)
        assert rainy_loss < clear_loss
        
        # Also test with full enhanced propagation
        clear_enhanced = clear_env.calculate_enhanced_propagation_loss(
            frequency, target_range, 100, 500
        )
        
        rainy_enhanced = rainy_env.calculate_enhanced_propagation_loss(
            frequency, target_range, 100, 500
        )
        
        # The ratio between rainy and clear should show pronounced rain effect
        rain_effect = rainy_enhanced / clear_enhanced
        assert rain_effect < 0.5
    
    def test_different_clutter_types(self):
        """Test behavior with different clutter types."""
        env = EnhancedEnvironment()
        
        # Test different terrain types
        terrain_types = ['urban', 'rural', 'forest', 'desert', 'mountains']
        
        range_bins = np.array([1000, 5000, 10000])
        azimuth_bins = np.array([-5, 0, 5])
        frequency = 10e9
        
        clutter_maps = {}
        
        for terrain in terrain_types:
            env.clutter_models = []  # Clear previous models
            land_model = LandClutterModel(terrain_type=terrain)
            env.add_clutter_model(land_model)
            
            clutter_map = env.generate_clutter_map(range_bins, azimuth_bins, frequency)
            clutter_maps[terrain] = np.mean(clutter_map)
        
        # Urban should generally have higher clutter than desert
        assert clutter_maps['urban'] > clutter_maps['desert']
        
        # Forest should have higher clutter than rural due to vegetation
        assert clutter_maps['forest'] > clutter_maps['rural']


if __name__ == '__main__':
    pytest.main([__file__])
