"""
Tests for waveform generation.
"""

import pytest
import numpy as np

from pydar.waveforms import LinearFMChirp, PulseTrain, BarkerCode, SteppedFrequency


class TestLinearFMChirp:
    """Test LinearFMChirp class."""
    
    def test_chirp_initialization(self):
        """Test chirp initialization attributes."""
        chirp = LinearFMChirp(duration=10e-6, sample_rate=100e6, bandwidth=50e6)
        
        assert chirp.duration == 10e-6
        assert chirp.sample_rate == 100e6
        assert chirp.bandwidth == 50e6
        assert chirp.chirp_rate == pytest.approx(5e12)


class TestPulseTrain:
    """Test PulseTrain class."""
    
    def test_pulse_train_initialization(self):
        """Test pulse train initialization attributes."""
        pulse_train = PulseTrain(
            pulse_width=1e-6,
            prf=1e3,
            num_pulses=10,
            sample_rate=10e6
        )
        
        assert pulse_train.pulse_width == 1e-6
        assert pulse_train.prf == 1e3
        assert pulse_train.num_pulses == 10
        assert pulse_train.duty_cycle() == pytest.approx(0.001)


class TestBarkerCode:
    """Test BarkerCode class."""
    
    def test_barker_code_initialization(self):
        """Test Barker code initialization."""
        barker = BarkerCode(code_length=7, chip_width=1e-6, sample_rate=10e6)
        
        assert barker.code_length == 7
        assert barker.bandwidth == pytest.approx(1e6)
        
    def test_barker_code_invalid_length(self):
        """Test Barker code invalid length."""
        with pytest.raises(ValueError):
            BarkerCode(code_length=6, chip_width=1e-6, sample_rate=10e6)


class TestSteppedFrequency:
    """Test SteppedFrequency class."""
    
    def test_stepped_frequency_initialization(self):
        """Test stepped frequency initialization attributes."""
        stepped_freq = SteppedFrequency(
            num_steps=10,
            step_duration=1e-6,
            start_frequency=1e9,
            frequency_step=1e6,
            sample_rate=10e6
        )
        
        assert stepped_freq.num_steps == 10
        assert stepped_freq.bandwidth == 10e6
        assert stepped_freq.start_frequency == 1e9

    def test_stepped_frequency_synthetic_bandwidth(self):
        """Test stepped frequency synthetic bandwidth calculation."""
        stepped_freq = SteppedFrequency(
            num_steps=10,
            step_duration=1e-6,
            start_frequency=1e9,
            frequency_step=1e6,
            sample_rate=10e6
        )
        
        assert stepped_freq.synthetic_bandwidth() == 10e6
