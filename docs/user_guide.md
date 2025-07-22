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
from pydar import RadarSystem, Target, Environment, LinearFMChirp

# Create radar
radar = RadarSystem(frequency=10e9, power=1000, antenna_gain=30)

# Create waveform
waveform = LinearFMChirp(duration=10e-6, sample_rate=100e6, bandwidth=50e6)

# Create environment with target
env = Environment()
env.add_target(Target(range=5000, velocity=50, rcs=10))

# Run simulation
result = radar.scan(env, waveform)
print(result.summary())
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
radar = RadarSystem(
    frequency=10e9,      # 10 GHz
    power=1000,          # 1 kW
    antenna_gain=30,     # 30 dB
    system_loss=3,       # 3 dB loss
    noise_figure=3       # 3 dB noise figure
)
```

### Custom Antenna

```python
from pydar import Antenna

antenna = Antenna(
    gain=35,
    beamwidth_azimuth=2,
    beamwidth_elevation=2,
    sidelobe_level=-20
)

radar = RadarSystem(
    frequency=24e9,
    power=100,
    antenna_gain=35,
    antenna=antenna
)
```

## Waveform Design

### Linear FM Chirp

Most common radar waveform:

```python
chirp = LinearFMChirp(
    duration=20e-6,      # 20 microseconds
    sample_rate=200e6,   # 200 MHz sampling
    bandwidth=100e6      # 100 MHz bandwidth
)
```

### Pulse Train

For Doppler processing:

```python
pulses = PulseTrain(
    pulse_width=1e-6,    # 1 microsecond pulses
    prf=5000,            # 5 kHz PRF
    num_pulses=64,       # 64 pulses
    sample_rate=100e6,
    pulse_type='rect'    # Rectangular pulses
)
```

### Barker Code

For pulse compression:

```python
barker = BarkerCode(
    code_length=13,      # 13-bit Barker code
    chip_width=1e-6,     # 1 microsecond chips
    sample_rate=50e6
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
    acceleration=5       # 5 m/s² acceleration
)

# Update position after 2 seconds
target.update_position(2.0)
```

### Fluctuating Target (Swerling Models)

```python
from pydar.target import SwerlingModel

target = Target(
    range=8000,
    velocity=0,
    rcs=20,
    swerling_model=SwerlingModel.SWERLING_1
)
```

### Extended Target

```python
from pydar.target import ExtendedTarget

ship = ExtendedTarget(
    range=15000,
    velocity=10,
    rcs=1000,
    extent_range=100,     # 100m length
    extent_cross_range=20, # 20m width
    num_scatterers=50
)
```

## Environment Setup

### Basic Environment

```python
env = Environment()

# Add individual targets
env.add_target(Target(range=5000, velocity=30, rcs=10))
env.add_target(Target(range=7000, velocity=-20, rcs=5))
```

### Adding Clutter

```python
# Sea clutter
env.add_sea_clutter(
    sea_state=3,         # Sea state 3
    area=(20000, 60),    # 20km range, 60° azimuth
    resolution=50        # 50m patches
)

# Land clutter
env.add_land_clutter(
    terrain_type='urban',
    area=(10000, 30),
    resolution=25
)
```

### Atmospheric Effects

```python
from pydar.environment import Atmosphere

# Custom atmosphere
atmosphere = Atmosphere(
    temperature=293.15,  # 20°C
    pressure=101325,     # Sea level
    humidity=0.7,        # 70% humidity
    rain_rate=5.0        # 5 mm/hr rain
)

env = Environment(atmosphere=atmosphere)
```

## Running Simulations

### Basic Scan

```python
# Perform scan
result = radar.scan(env, waveform)

# Access results
print(f"Transmitted samples: {len(result.tx_signal)}")
print(f"Received samples: {len(result.rx_signal)}")
print(f"Detected targets: {len(result.target_info)}")
```

### Processing Results

```python
# Get matched filter output
mf_output = result.matched_filter()

# Get range profile
ranges, amplitudes = result.range_profile()

# Plot results
result.plot_range_profile(max_range=20000)
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

### Range Profile Plot

```python
import matplotlib.pyplot as plt

ranges, amplitudes = result.range_profile()
plt.figure(figsize=(10, 6))
plt.plot(ranges/1000, 20*np.log10(amplitudes))
plt.xlabel('Range (km)')
plt.ylabel('Amplitude (dB)')
plt.grid(True)
plt.show()
```

### Spectrogram

```python
result.plot_spectrogram(
    window_size=256,
    overlap=0.8,
    figsize=(12, 8)
)
```

### Range-Doppler Map

```python
rd_map.plot(
    db_scale=True,
    velocity_scale=True,
    clim=(-40, 0)  # Color limits in dB
)
```

## Advanced Topics

### Multi-Static Radar

```python
# Create multiple radar systems
radar1 = RadarSystem(frequency=10e9, power=1000, antenna_gain=30)
radar2 = RadarSystem(frequency=10e9, power=500, antenna_gain=25)

# Position radars (would need position attributes)
# Simulate bistatic geometry
```

### Phased Array

```python
# Future feature: Phased array modeling
# Would include beam steering, multiple simultaneous beams
```

### Save/Load Scenarios

```python
# Save environment
env.save_scenario('scenario.json')

# Load environment
new_env = Environment()
new_env.load_scenario('scenario.json')
```

### HDF5 Data Storage

```python
# Save scan results
result.save_to_hdf5('scan_data.h5')

# Load results
from pydar.scan_result import ScanResult
loaded = ScanResult.load_from_hdf5('scan_data.h5')
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
