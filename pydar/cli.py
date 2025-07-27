"""
Command-line interface for the PyDar radar simulator.

This module provides command-line access to PyDar's key features and examples.
"""

import sys
import os
from pathlib import Path

import click
from scipy.constants import c


@click.group()
@click.version_option(version='0.1.0', prog_name='pydar')
def cli():
    """PyDar - Python Radar Simulator Command Line Interface.
    
    A high-fidelity radar simulation library for educational and research purposes.
    """
    pass


@cli.command()
@click.argument('example', type=click.Choice(['basic', 'static', 'live_3d'], case_sensitive=False))
def run_example(example):
    """Run a specified example simulation.
    
    EXAMPLE: The example to run (basic, static, or live_3d)
    """
    example = example.lower()
    
    try:
        click.echo(f'Running {example} example...')
        
        if example == 'basic':
            from examples.basic import main
        elif example == 'static':
            from examples.static import main
        elif example == 'live_3d':
            from examples.live_3d import main
            
        main()
        click.echo(f'✓ {example.title()} example completed successfully!')
        
    except ImportError as e:
        click.echo(f'✗ Error importing example: {e}', err=True)
        click.echo('Make sure PyDar is properly installed with examples.', err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f'✗ Error running example: {e}', err=True)
        sys.exit(1)


@cli.group()
def calc():
    """Radar calculation utilities."""
    pass


@calc.command()
@click.option('--frequency', '-f', type=float, required=True, 
              help='Radar frequency in GHz.')
def wavelength(frequency):
    """Calculate the wavelength for a given frequency."""
    if frequency <= 0:
        click.echo('✗ Frequency must be positive!', err=True)
        sys.exit(1)
        
    wavelength_m = c / (frequency * 1e9)
    wavelength_cm = wavelength_m * 100
    
    click.echo(f'Frequency: {frequency:.2f} GHz')
    click.echo(f'Wavelength: {wavelength_m:.4f} m ({wavelength_cm:.2f} cm)')


@calc.command()
@click.option('--bandwidth', '-b', type=float, required=True,
              help='Signal bandwidth in MHz.')
def range_resolution(bandwidth):
    """Calculate range resolution from bandwidth."""
    if bandwidth <= 0:
        click.echo('✗ Bandwidth must be positive!', err=True)
        sys.exit(1)
        
    bandwidth_hz = bandwidth * 1e6
    resolution = c / (2 * bandwidth_hz)
    
    click.echo(f'Bandwidth: {bandwidth:.2f} MHz')
    click.echo(f'Range resolution: {resolution:.2f} m')


@calc.command()
@click.option('--prf', type=float, required=True,
              help='Pulse repetition frequency in Hz.')
def max_range(prf):
    """Calculate maximum unambiguous range from PRF."""
    if prf <= 0:
        click.echo('✗ PRF must be positive!', err=True)
        sys.exit(1)
        
    max_range_m = c / (2 * prf)
    max_range_km = max_range_m / 1000
    
    click.echo(f'PRF: {prf:.0f} Hz')
    click.echo(f'Maximum unambiguous range: {max_range_m:.0f} m ({max_range_km:.2f} km)')


@calc.command()
@click.option('--prf', type=float, required=True,
              help='Pulse repetition frequency in Hz.')
@click.option('--frequency', '-f', type=float, required=True,
              help='Radar frequency in GHz.')
def max_velocity(prf, frequency):
    """Calculate maximum unambiguous velocity from PRF and frequency."""
    if prf <= 0 or frequency <= 0:
        click.echo('✗ PRF and frequency must be positive!', err=True)
        sys.exit(1)
        
    wavelength_m = c / (frequency * 1e9)
    max_vel = prf * wavelength_m / 4
    
    click.echo(f'PRF: {prf:.0f} Hz')
    click.echo(f'Frequency: {frequency:.2f} GHz')
    click.echo(f'Maximum unambiguous velocity: ±{max_vel:.1f} m/s')


@cli.command()
def info():
    """Display PyDar version and system information."""
    click.echo('PyDar - Python Radar Simulator')
    click.echo('Version: 0.1.0')
    click.echo('Author: PyDar Development Team')
    click.echo()
    click.echo('Features:')
    click.echo('  • Radar system modeling')
    click.echo('  • Target simulation')
    click.echo('  • Signal processing algorithms')
    click.echo('  • 3D visualization')
    click.echo('  • Multi-target tracking')
    click.echo()
    click.echo('Use "pydar --help" for available commands.')


@cli.command()
def examples():
    """List available examples with descriptions."""
    examples_info = {
        'basic': 'Basic radar simulation with simple visualization',
        'static': '3D static visualization example', 
        'live_3d': 'Live simulation with Dash web interface'
    }
    
    click.echo('Available Examples:')
    click.echo()
    for name, description in examples_info.items():
        click.echo(f'  {name:<10} - {description}')
    
    click.echo()
    click.echo('Run an example with: pydar run-example [EXAMPLE_NAME]')

def main():
    """Main entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main()
