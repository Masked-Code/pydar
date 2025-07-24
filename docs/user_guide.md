# PyDar User Guide

This guide provides detailed instructions on how to use PyDar for radar simulation.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Concepts](#basic-concepts)
3. [Creating a Radar System](#creating-a-radar-system)
4. [Waveform Design](#waveform-design)
5. [Target Modeling](#target-modeling)
6. [Environment Setup](#environment-setup)
7. [Running Simulations](#running-simulations)
8. [Signal Processing](#signal-processing)
9. [Visualization](#visualization)
10. [Advanced Topics](#advanced-topics)

## Getting Started

### Installation

First, install PyDar and its dependencies:

```bash
pip install -e .
```

### Basic Example

Here's a simple example to get you started:

```python
from pydar import RadarSystem, Target, Environment, LinearFMChirp, Antenna

# Create antenna
antenna = Antenna(gain=30, beamwidth_azimuth=2.0, beamwidth_elevation=2.0)

# Create waveform
waveform = LinearFMChirp(
    duration=10e-6,
    sample_rate=100e6,
    bandwidth=50e6,
    center_frequency=10e9
)

# Create radar
radar = RadarSystem(
    antenna=antenna,
    waveform=waveform,
    transmit_power=1000,
    noise_figure=3
)

# Create environment with target
env = Environment()
env.add_target(Target(range=5000, velocity=50, rcs=10, azimuth=10))

# Run simulation
result = radar.scan(env)
print(f"Detected {len(result.returns)} targets")
```

## Basic Concepts

### Radar System Parameters

- **Frequency**: Operating frequency in Hz (e.g., 10 GHz = 10e9 Hz)
- **Power**: Transmit power in Watts
- **Antenna Gain**: Antenna gain in dB
- **System Loss**: Total system losses in dB
- **Noise Figure**: Receiver noise figure in dB

### Coordinate System

PyDar uses a spherical coordinate system:
- **Range**: Distance from radar in meters
- **Azimuth**: Angle in horizontal plane (degrees)
- **Elevation**: Angle from horizontal plane (degrees)

## Creating a Radar System

### Basic Radar

```python
# First create antenna and waveform
antenna = Antenna(gain=30, beamwidth_azimuth=3.0, beamwidth_elevation=3.0)
waveform = LinearFMChirp(
    duration=10e-6,
    sample_rate=100e6,
    bandwidth=50e6,
    center_frequency=10e9
)

# Then create radar system
radar = RadarSystem(
    antenna=antenna,
    waveform=waveform,
    transmit_power=1000,  # 1 kW
    noise_figure=3,       # 3 dB
    losses=3              # 3 dB system losses
)
```

### Custom Antenna

```python
from pydar import Antenna

antenna = Antenna(
    gain=35,
    beamwidth_azimuth=2,
    beamwidth_elevation=2,
    sidelobe_level=-20,
    efficiency=0.8
)

waveform = LinearFMChirp(
    duration=10e-6,
    sample_rate=200e6,
    bandwidth=100e6,
    center_frequency=24e9
)

radar = RadarSystem(
    antenna=antenna,
    waveform=waveform,
    transmit_power=100
)
```

## Waveform Design

### Linear FM Chirp

Most common radar waveform:

```python
chirp = LinearFMChirp(
    duration=20e-6,          # 20 microseconds
    sample_rate=200e6,       # 200 MHz sampling
    bandwidth=100e6,         # 100 MHz bandwidth
    center_frequency=10e9    # 10 GHz center frequency
)
```

### Pulse Train

For Doppler processing:

```python
from pydar.waveforms import PulseTrain

pulses = PulseTrain(
    pulse_width=1e-6,        # 1 microsecond pulses
    prf=5000,                # 5 kHz PRF
    num_pulses=64,           # 64 pulses
    sample_rate=100e6,
    center_frequency=10e9,   # 10 GHz center frequency
    pulse_type='rect'        # Rectangular pulses
)
```

### Barker Code

For pulse compression:

```python
from pydar.waveforms import BarkerCode

barker = BarkerCode(
    code_length=13,          # 13-bit Barker code
    chip_width=1e-6,         # 1 microsecond chips
    sample_rate=50e6,
    center_frequency=10e9    # 10 GHz center frequency
)
```

## Target Modeling

### Point Target

```python
target = Target(
    range=10000,         # 10 km
    velocity=200,        # 200 m/s approaching
    rcs=5,               # 5 m² RCS
    azimuth=45,          # 45° azimuth
    elevation=10         # 10° elevation
)
```

### Moving Target

```python
target = Target(
    range=5000,
    velocity=50,
    rcs=10,
    azimuth=0,
    elevation=0
)

# Note: For moving targets with acceleration, implement update_position method
# This is a future enhancement
```

### Fluctuating Target (Future Enhancement)

```python
# Swerling models are planned for future implementation
# For now, targets have constant RCS
target = Target(
    range=8000,
    velocity=0,
    rcs=20,
    azimuth=0,
    elevation=0
)
```

### Multiple Targets

```python
from pydar import TargetCollection

# Create a collection of targets
targets = TargetCollection()
targets.add_target(Target(range=5000, velocity=30, rcs=10, azimuth=10))
targets.add_target(Target(range=7000, velocity=-20, rcs=5, azimuth=-5))
targets.add_target(Target(range=10000, velocity=100, rcs=50, azimuth=0))

# Add to environment
env = Environment()
env.targets = targets
```

## Environment Setup

### Basic Environment

```python
env = Environment()

# Add individual targets
env.add_target(Target(range=5000, velocity=30, rcs=10))
env.add_target(Target(range=7000, velocity=-20, rcs=5))
```

### Atmospheric Effects

```python
from pydar import Atmosphere

# Custom atmosphere
atmosphere = Atmosphere(
    temperature=293.15,  # 20°C
    pressure=101325,     # Sea level
    humidity=0.7,        # 70% humidity
    rain_rate=5.0        # 5 mm/hr rain
)

env = Environment(atmosphere=atmosphere)

# Note: Clutter models (sea and land) are planned for future implementation
```


## Running Simulations

### Basic Scan

```python
# Perform scan
result = radar.scan(env)

# Access results
print(f"Detected targets: {len(result.returns)}")
for detection in result.returns:
    print(f"  Range: {detection.range:.1f} m")
    print(f"  Azimuth: {detection.azimuth:.1f}°")
    print(f"  Power: {10*np.log10(detection.power):.1f} dBm")
```

### Processing Results

```python
# Note: Advanced processing (matched filtering, range profiles) 
# requires additional implementation beyond the basic scan

# For now, work with the detection returns directly
for detection in result.returns:
    snr = radar.snr(detection.range, 10.0)  # Assuming 10 m² RCS
    print(f"Target at {detection.range:.0f}m, SNR: {snr:.1f} dB")
```

## Signal Processing

### Range-Doppler Processing

```python
from pydar.processing import RangeDopplerProcessor

# Create processor
processor = RangeDopplerProcessor(
    num_pulses=64,
    prf=5000,
    range_bins=1024
)

# Process data
rd_map = processor.process(pulse_data)

# Visualize
rd_map.plot(db_scale=True, velocity_scale=True)
```

### CFAR Detection

```python
from pydar.processing import CFARDetector

# Create detector
cfar = CFARDetector(
    guard_cells=3,
    training_cells=10,
    pfa=1e-6
)

# Detect targets
detections = cfar.detect(range_profile)
```

### Target Tracking

```python
from pydar.processing import SimpleTracker

# Create tracker
tracker = SimpleTracker(
    max_distance=100,
    max_missed=3,
    min_hits=3
)

# Update with detections
tracks = tracker.update(detections, timestamp=0.1)

# Get confirmed tracks
confirmed = tracker.get_confirmed_tracks()
```

## Visualization

### 3D Visualization

```python
from pydar import Radar3DVisualizer, VisualizationConfig

# Configure visualization
config = VisualizationConfig(
    figure_width=1200,
    figure_height=800,
    show_statistics=True,
    show_doppler=True,
    show_rcs=True
)

# Create visualizer
visualizer = Radar3DVisualizer(config)

# Create visualization
fig = visualizer.visualize(radar, environment, scan_result=result)
fig.show()
```

### Live Visualization with Dash

See the `examples/live_3d.py` for a complete example of live visualization using Dash.

## Advanced Topics

### Future Enhancements

The following features are planned for future releases:

- **Multi-Static Radar**: Support for bistatic and multistatic configurations
- **Phased Array**: Electronic beam steering and multiple simultaneous beams
- **Extended Targets**: Targets with spatial extent and multiple scattering centers
- **Swerling Models**: Statistical RCS fluctuation models
- **Advanced Clutter**: Sea and land clutter models with realistic statistics

### Data Management

```python
# Note: Save/load functionality is planned for future implementation
# For now, use standard Python serialization methods like pickle
# or save data manually using numpy/pandas
```

## Best Practices

1. **Sampling Rate**: Use at least 2× the signal bandwidth
2. **Waveform Selection**: 
   - Chirp for range resolution
   - Pulse train for Doppler
   - Barker codes for low power
3. **Environment Modeling**: Include clutter for realistic scenarios
4. **Processing**: Apply appropriate windowing for spectral analysis
5. **Validation**: Compare results with theoretical predictions

## Troubleshooting

### Common Issues

1. **Low SNR**: Increase power, antenna gain, or integration time
2. **Range Ambiguity**: Reduce PRF or use multiple PRFs
3. **Velocity Ambiguity**: Increase PRF or use staggered PRF
4. **Processing Artifacts**: Check windowing and FFT parameters

### Performance Tips

- Pre-allocate arrays for large simulations
- Use vectorized operations where possible
- Consider parallel processing for multiple targets
- Cache frequently computed values
