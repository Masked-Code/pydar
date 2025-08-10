"""
Comprehensive tests for waveform generation covering all methods.
"""

import pytest
import numpy as np

from pydar.waveforms import (
    LinearFMChirp, PulseTrain, BarkerCode, SteppedFrequency, 
    CustomWaveform, Waveform
)


class TestWaveformBase:
    """Test base Waveform class methods."""
    
    def test_autocorrelation(self):
        """Test waveform autocorrelation."""
        chirp = LinearFMChirp(duration=10e-6, sample_rate=100e6, bandwidth=50e6)
        
        lags, autocorr = chirp.autocorrelation()
        
        assert len(lags) == len(autocorr)
        assert np.max(np.abs(autocorr)) == pytest.approx(1.0)  # Normalized
        assert autocorr[len(autocorr)//2] == pytest.approx(1.0)  # Peak at zero lag
    
    def test_ambiguity_function(self):
        """Test ambiguity function calculation."""
        chirp = LinearFMChirp(duration=10e-6, sample_rate=100e6, bandwidth=50e6)
        
        max_delay = 5e-6
        max_doppler = 10e3
        num_delay = 50
        num_doppler = 50
        
        delays, dopplers, ambig = chirp.ambiguity_function(
            max_delay=max_delay, max_doppler=max_doppler, 
            num_delay=num_delay, num_doppler=num_doppler
        )
        
        assert delays.shape == (50,)
        assert dopplers.shape == (50,)
        assert ambig.shape == (50, 50)
        assert np.max(ambig) == pytest.approx(1.0)  # Normalized
        
        # Peak should be near zero delay and zero Doppler
        peak_idx = np.unravel_index(np.argmax(ambig), ambig.shape)
        assert abs(delays[peak_idx[1]]) < 1e-6
        # For a simple chirp, the peak should be at zero Doppler
        # Allow some tolerance due to discretization
        doppler_resolution = 2 * max_doppler / num_doppler
        assert abs(dopplers[peak_idx[0]]) <= doppler_resolution


class TestLinearFMChirpComplete:
    """Complete tests for LinearFMChirp."""
    
    def test_chirp_generation(self):
        """Test chirp signal generation."""
        chirp = LinearFMChirp(
            duration=10e-6, 
            sample_rate=100e6, 
            bandwidth=50e6,
            center_frequency=1e6
        )
        
        signal = chirp.generate()
        
        assert len(signal) == chirp.num_samples
        assert signal.dtype == complex
        assert np.all(np.abs(signal) == pytest.approx(1.0))  # Unit amplitude
        
        # Test with custom time vector
        t_custom = np.linspace(0, 5e-6, 500)
        signal_custom = chirp.generate(t_custom)
        assert len(signal_custom) == 500
    
    def test_time_bandwidth_product(self):
        """Test time-bandwidth product calculation."""
        chirp = LinearFMChirp(duration=10e-6, sample_rate=100e6, bandwidth=50e6)
        
        tb_product = chirp.time_bandwidth_product()
        expected = 10e-6 * 50e6
        assert tb_product == pytest.approx(expected)


class TestPulseTrainComplete:
    """Complete tests for PulseTrain."""
    
    def test_pulse_train_generation(self):
        """Test pulse train generation."""
        pulse_train = PulseTrain(
            pulse_width=1e-6,
            prf=1e3,
            num_pulses=5,
            sample_rate=10e6,
            pulse_type='rect'
        )
        
        signal = pulse_train.generate()
        
        assert len(signal) == pulse_train.num_samples
        assert signal.dtype == complex
        
        # Check that pulses exist
        assert np.max(np.abs(signal)) > 0
    
    def test_gaussian_pulse(self):
        """Test Gaussian pulse generation."""
        pulse_train = PulseTrain(
            pulse_width=1e-6,
            prf=1e3,
            num_pulses=3,
            sample_rate=10e6,
            pulse_type='gaussian'
        )
        
        signal = pulse_train.generate()
        assert np.max(np.abs(signal)) <= 1.0  # Gaussian peaks at 1
    
    def test_sinc_pulse(self):
        """Test sinc pulse generation."""
        pulse_train = PulseTrain(
            pulse_width=1e-6,
            prf=1e3,
            num_pulses=3,
            sample_rate=10e6,
            pulse_type='sinc'
        )
        
        signal = pulse_train.generate()
        assert len(signal) > 0
        # Sinc can have negative values
        assert np.any(np.real(signal) < 0)


class TestBarkerCodeComplete:
    """Complete tests for BarkerCode."""
    
    def test_barker_generation(self):
        """Test Barker code generation."""
        barker = BarkerCode(code_length=13, chip_width=1e-6, sample_rate=10e6)
        
        signal = barker.generate()
        
        assert len(signal) == barker.num_samples
        assert signal.dtype == complex
        # Barker codes are bipolar
        assert set(np.unique(np.real(signal[signal != 0]))) == {-1, 1}
    
    def test_all_barker_codes(self):
        """Test all valid Barker code lengths."""
        valid_lengths = [2, 3, 4, 5, 7, 11, 13]
        
        for length in valid_lengths:
            barker = BarkerCode(code_length=length, chip_width=1e-6, sample_rate=10e6)
            signal = barker.generate()
            assert len(signal) > 0


class TestSteppedFrequencyComplete:
    """Complete tests for SteppedFrequency."""
    
    def test_stepped_frequency_generation(self):
        """Test stepped frequency waveform generation."""
        stepped = SteppedFrequency(
            num_steps=5,
            step_duration=1e-6,
            start_frequency=1e9,
            frequency_step=10e6,
            sample_rate=100e6
        )
        
        signal = stepped.generate()
        
        assert len(signal) == stepped.num_samples
        assert signal.dtype == complex
        
        # Each step should have constant frequency
        samples_per_step = int(stepped.step_duration * stepped.sample_rate)
        
        # Check first step has expected frequency content
        first_step = signal[:samples_per_step]
        assert len(first_step) > 0


class TestCustomWaveform:
    """Test CustomWaveform class."""
    
    def test_custom_with_function(self):
        """Test custom waveform with generator function."""
        def my_waveform(t):
            return np.exp(1j * 2 * np.pi * 1e6 * t)
        
        custom = CustomWaveform(
            duration=10e-6,
            sample_rate=10e6,
            bandwidth=1e6,
            generator_func=my_waveform
        )
        
        signal = custom.generate()
        assert len(signal) == custom.num_samples
        assert signal.dtype == complex
    
    def test_custom_with_samples(self):
        """Test custom waveform with pre-computed samples."""
        samples = np.exp(1j * np.linspace(0, 2*np.pi, 100))
        
        custom = CustomWaveform(
            duration=10e-6,
            sample_rate=10e6,
            bandwidth=1e6,
            samples=samples
        )
        
        signal = custom.generate()
        assert len(signal) == custom.num_samples
        
        # Test with different time vector (should interpolate)
        t_new = np.linspace(0, custom.duration, 200)
        signal_interp = custom.generate(t_new)
        assert len(signal_interp) == 200
    
    def test_custom_validation(self):
        """Test custom waveform validation."""
        with pytest.raises(ValueError):
            # Neither function nor samples provided
            CustomWaveform(
                duration=10e-6,
                sample_rate=10e6,
                bandwidth=1e6
            )
