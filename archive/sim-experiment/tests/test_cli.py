"""
Tests for the PyDar CLI module.

This module tests the command-line interface functionality.
"""

import pytest
from click.testing import CliRunner
from pydar.cli import cli


class TestCLI:
    """Test cases for the CLI interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'PyDar - Python Radar Simulator Command Line Interface' in result.output
        assert 'Commands:' in result.output
    
    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output
    
    def test_info_command(self):
        """Test info command."""
        result = self.runner.invoke(cli, ['info'])
        assert result.exit_code == 0
        assert 'PyDar - Python Radar Simulator' in result.output
        assert 'Version: 0.1.0' in result.output
        assert 'Features:' in result.output
    
    def test_examples_command(self):
        """Test examples command."""
        result = self.runner.invoke(cli, ['examples'])
        assert result.exit_code == 0
        assert 'Available Examples:' in result.output
        assert 'basic' in result.output
        assert 'static' in result.output
        assert 'live_3d' in result.output
    
    def test_calc_help(self):
        """Test calc subcommand help."""
        result = self.runner.invoke(cli, ['calc', '--help'])
        assert result.exit_code == 0
        assert 'Radar calculation utilities' in result.output
        assert 'wavelength' in result.output
        assert 'range-resolution' in result.output
        assert 'max-range' in result.output
        assert 'max-velocity' in result.output
    
    def test_calc_wavelength(self):
        """Test wavelength calculation."""
        result = self.runner.invoke(cli, ['calc', 'wavelength', '-f', '10.0'])
        assert result.exit_code == 0
        assert 'Frequency: 10.00 GHz' in result.output
        assert 'Wavelength: 0.0300 m (3.00 cm)' in result.output
    
    def test_calc_wavelength_invalid(self):
        """Test wavelength calculation with invalid input."""
        result = self.runner.invoke(cli, ['calc', 'wavelength', '-f', '-5'])
        assert result.exit_code == 1
        assert '✗ Frequency must be positive!' in result.output
    
    def test_calc_range_resolution(self):
        """Test range resolution calculation."""
        result = self.runner.invoke(cli, ['calc', 'range-resolution', '-b', '50'])
        assert result.exit_code == 0
        assert 'Bandwidth: 50.00 MHz' in result.output
        assert 'Range resolution: 3.00 m' in result.output
    
    def test_calc_range_resolution_invalid(self):
        """Test range resolution calculation with invalid input."""
        result = self.runner.invoke(cli, ['calc', 'range-resolution', '-b', '-10'])
        assert result.exit_code == 1
        assert '✗ Bandwidth must be positive!' in result.output
    
    def test_calc_max_range(self):
        """Test maximum range calculation."""
        result = self.runner.invoke(cli, ['calc', 'max-range', '--prf', '1000'])
        assert result.exit_code == 0
        assert 'PRF: 1000 Hz' in result.output
        assert 'Maximum unambiguous range:' in result.output
        assert 'km' in result.output
    
    def test_calc_max_range_invalid(self):
        """Test maximum range calculation with invalid input."""
        result = self.runner.invoke(cli, ['calc', 'max-range', '--prf', '-500'])
        assert result.exit_code == 1
        assert '✗ PRF must be positive!' in result.output
    
    def test_calc_max_velocity(self):
        """Test maximum velocity calculation."""
        result = self.runner.invoke(cli, ['calc', 'max-velocity', '--prf', '1000', '-f', '10.0'])
        assert result.exit_code == 0
        assert 'PRF: 1000 Hz' in result.output
        assert 'Frequency: 10.00 GHz' in result.output
        assert 'Maximum unambiguous velocity: ±7.5 m/s' in result.output
    
    def test_calc_max_velocity_invalid_prf(self):
        """Test maximum velocity calculation with invalid PRF."""
        result = self.runner.invoke(cli, ['calc', 'max-velocity', '--prf', '-1000', '-f', '10.0'])
        assert result.exit_code == 1
        assert '✗ PRF and frequency must be positive!' in result.output
    
    def test_calc_max_velocity_invalid_frequency(self):
        """Test maximum velocity calculation with invalid frequency."""
        result = self.runner.invoke(cli, ['calc', 'max-velocity', '--prf', '1000', '-f', '-10.0'])
        assert result.exit_code == 1
        assert '✗ PRF and frequency must be positive!' in result.output
    
    def test_run_example_invalid(self):
        """Test running an invalid example."""
        result = self.runner.invoke(cli, ['run-example', 'nonexistent'])
        assert result.exit_code == 2  # Click validation error
    
    def test_run_example_valid_choice(self):
        """Test that valid example choices are accepted."""
        # Test that basic is a valid choice (we won't actually run it in tests)
        result = self.runner.invoke(cli, ['run-example', '--help'])
        assert result.exit_code == 0
        assert 'basic' in result.output
        assert 'static' in result.output
        assert 'live_3d' in result.output


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_multiple_calculations(self):
        """Test running multiple calculations in sequence."""
        # Test wavelength calculation
        result1 = self.runner.invoke(cli, ['calc', 'wavelength', '-f', '5.0'])
        assert result1.exit_code == 0
        assert '0.0600 m (6.00 cm)' in result1.output
        
        # Test range resolution
        result2 = self.runner.invoke(cli, ['calc', 'range-resolution', '-b', '100'])
        assert result2.exit_code == 0
        assert 'Range resolution: 1.50 m' in result2.output
        
        # Test max range
        result3 = self.runner.invoke(cli, ['calc', 'max-range', '--prf', '500'])
        assert result3.exit_code == 0
        assert '299792 m' in result3.output
    
    def test_edge_cases(self):
        """Test edge cases for calculations."""
        # Very high frequency
        result = self.runner.invoke(cli, ['calc', 'wavelength', '-f', '100.0'])
        assert result.exit_code == 0
        assert '0.0030 m (0.30 cm)' in result.output
        
        # Very small bandwidth
        result = self.runner.invoke(cli, ['calc', 'range-resolution', '-b', '0.1'])
        assert result.exit_code == 0
        assert 'Range resolution: 1498.96 m' in result.output
    
    def test_command_chaining_compatibility(self):
        """Test that commands can be used independently."""
        # Each command should work independently
        commands = [
            ['info'],
            ['examples'],
            ['calc', 'wavelength', '-f', '10'],
            ['calc', 'range-resolution', '-b', '50'],
            ['calc', 'max-range', '--prf', '1000'],
            ['calc', 'max-velocity', '--prf', '1000', '-f', '10']
        ]
        
        for cmd in commands:
            result = self.runner.invoke(cli, cmd)
            assert result.exit_code == 0, f"Command {cmd} failed with output: {result.output}"


if __name__ == '__main__':
    pytest.main([__file__])
