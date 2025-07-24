"""
Pytest fixtures for PyDar tests.
"""

import pytest
import numpy as np

from pydar import RadarSystem, Target, Environment, LinearFMChirp, Antenna


@pytest.fixture
def basic_radar():
    """Create a basic radar system for testing."""
    antenna = Antenna(gain=30, beamwidth_azimuth=3.0, beamwidth_elevation=3.0)
    waveform = LinearFMChirp(
        duration=10e-6,
        sample_rate=100e6,
        bandwidth=50e6,
        center_frequency=10e9
    )
    return RadarSystem(
        antenna=antenna,
        waveform=waveform,
        transmit_power=1000,
        noise_figure=3,
        losses=3
    )


@pytest.fixture
def simple_target():
    """Create a simple target for testing."""
    return Target(
        range=5000,      # 5 km
        velocity=50,     # 50 m/s
        rcs=10          # 10 m²
    )


@pytest.fixture
def test_environment():
    """Create a test environment with targets."""
    env = Environment()
    
    # Add some targets
    env.add_target(Target(range=1000, velocity=20, rcs=1))
    env.add_target(Target(range=5000, velocity=-30, rcs=10))
    env.add_target(Target(range=10000, velocity=100, rcs=5))
    
    return env


@pytest.fixture
def chirp_waveform():
    """Create a chirp waveform for testing."""
    return LinearFMChirp(
        duration=10e-6,     # 10 μs
        sample_rate=100e6,  # 100 MHz
        bandwidth=50e6,     # 50 MHz
        center_frequency=10e9  # 10 GHz
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset seed after test
    np.random.seed(None)
