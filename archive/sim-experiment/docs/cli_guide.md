# PyDar Command Line Interface Guide

The PyDar CLI provides convenient access to radar calculations and examples directly from the command line.

## Installation

After installing PyDar with `pip install -e .`, the CLI is available through:

```bash
python -m pydar.cli [COMMAND]
```

## Available Commands

### General Commands

#### `info`
Display PyDar version and feature information.

```bash
python -m pydar.cli info
```

#### `examples`
List all available example simulations with descriptions.

```bash
python -m pydar.cli examples
```

#### `run-example EXAMPLE`
Run a specific example simulation. Available examples: `basic`, `static`, `live_3d`.

```bash
python -m pydar.cli run-example basic
python -m pydar.cli run-example static
python -m pydar.cli run-example live_3d
```

### Radar Calculations (`calc` subcommands)

#### `calc wavelength -f FREQUENCY`
Calculate wavelength from frequency (in GHz).

```bash
python -m pydar.cli calc wavelength -f 10.0
# Output: Frequency: 10.00 GHz, Wavelength: 0.0300 m (3.00 cm)
```

#### `calc range-resolution -b BANDWIDTH`
Calculate range resolution from bandwidth (in MHz).

```bash
python -m pydar.cli calc range-resolution -b 50
# Output: Bandwidth: 50.00 MHz, Range resolution: 3.00 m
```

#### `calc max-range --prf PRF`
Calculate maximum unambiguous range from PRF (in Hz).

```bash
python -m pydar.cli calc max-range --prf 1000
# Output: PRF: 1000 Hz, Maximum unambiguous range: 149896 m (149.90 km)
```

#### `calc max-velocity --prf PRF -f FREQUENCY`
Calculate maximum unambiguous velocity from PRF (Hz) and frequency (GHz).

```bash
python -m pydar.cli calc max-velocity --prf 1000 -f 10.0  
# Output: PRF: 1000 Hz, Frequency: 10.00 GHz, Maximum unambiguous velocity: ±7.5 m/s
```

## Common Usage Examples

### Quick Radar System Analysis
```bash
# Analyze a 10 GHz radar with 50 MHz bandwidth and 1 kHz PRF
python -m pydar.cli calc wavelength -f 10.0
python -m pydar.cli calc range-resolution -b 50
python -m pydar.cli calc max-range --prf 1000
python -m pydar.cli calc max-velocity --prf 1000 -f 10.0
```

### Running Examples
```bash
# List available examples
python -m pydar.cli examples

# Run basic simulation
python -m pydar.cli run-example basic

# Run 3D visualization
python -m pydar.cli run-example static
```

## Error Handling

The CLI includes comprehensive error checking:

- Negative frequencies, bandwidths, or PRF values are rejected
- Invalid example names show available options
- Missing required parameters display helpful error messages

## Help System

Get help for any command using `--help`:

```bash
python -m pydar.cli --help                    # Main help
python -m pydar.cli calc --help               # Calculator help
python -m pydar.cli run-example --help        # Example runner help
python -m pydar.cli calc wavelength --help    # Specific command help
```

## Version Information

Check the installed PyDar version:

```bash
python -m pydar.cli --version
```

## Integration with PyDar Library

The CLI complements the PyDar Python library. Use the CLI for quick calculations and exploring examples, then integrate the library into your Python scripts for complex simulations:

```python
from pydar import RadarSystem, Target, Environment, LinearFMChirp, Antenna

# Use CLI calculations to inform your simulation parameters
# Then create detailed simulations with the library
```
