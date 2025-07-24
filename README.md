# PyDar - Python Radar Simulator

A high-fidelity radar simulation library for Python, designed for educational and research purposes. PyDar provides comprehensive radar system modeling, signal processing, and visualization capabilities.

## Features

### Core Capabilities
- **Radar System Modeling**: Complete radar system simulation including antenna patterns, waveforms, and signal processing
- **Target Modeling**: Various target types with RCS models, motion dynamics, and Swerling fluctuation models
- **Environment Simulation**: Atmospheric effects, clutter modeling, and propagation effects
- **Signal Processing**: CFAR detection, range-Doppler processing, and multi-target tracking algorithms
- **Visualization**: Real-time 3D visualization using Plotly and Dash with web-based interface

### Key Components
- Phased array and parabolic antenna models with realistic beam patterns
- Multiple waveform types (Linear FM chirp, pulse trains, Barker codes, stepped frequency)
- Advanced detection algorithms (CA-CFAR, OS-CFAR, GO-CFAR)
- Multi-target tracking with Kalman filtering
- Live web-based visualization with automatic browser launch

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/yourusername/pydar.git
cd pydar
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Simulation Example

```python
from pydar import RadarSystem, Antenna, LinearFMChirp, Target, Environment

# Create radar components
antenna = Antenna(gain=30, beamwidth_azimuth=2.0, beamwidth_elevation=2.0)
waveform = LinearFMChirp(
    duration=10e-6,
    sample_rate=100e6,
    bandwidth=50e6,
    center_frequency=10e9
)

# Create radar system
radar = RadarSystem(
    antenna=antenna,
    waveform=waveform,
    position=(0, 0, 0),
    transmit_power=1000,
    noise_figure=3,
    losses=3
)

# Create environment with targets
env = Environment()
env.add_target(Target(range=5000, velocity=50, rcs=10, azimuth=10))

# Run simulation
result = radar.scan(env)
print(f"Detected {len(result.returns)} targets")
```

### 3D Visualization Example

```python
from pydar import Radar3DVisualizer, VisualizationConfig
from pydar import RadarSystem, Target, Environment, LinearFMChirp, Antenna

# Create visualization config
config = VisualizationConfig(
    figure_width=1200,
    figure_height=800,
    show_statistics=True,
    show_doppler=True
)

# Create radar and environment (as above)
# ...

# Create visualizer
visualizer = Radar3DVisualizer(config)
fig = visualizer.visualize(radar, environment, scan_result=result)
fig.show()
```

## Documentation

Full documentation is available in the `docs/` directory:

- [User Guide](docs/user_guide.md) - Detailed usage instructions
- [Examples](examples/) - Example scripts demonstrating various features

## Project Structure

```
pydar/
├── __init__.py          # Package initialization
├── radar.py             # Radar system and antenna models
├── target.py            # Target models and dynamics
├── environment.py       # Environmental effects
├── waveforms.py         # Waveform definitions
├── visualization.py     # 3D visualization module
├── processing/          # Signal processing algorithms
│   ├── __init__.py
│   ├── cfar.py         # CFAR detectors
│   ├── range_doppler.py # Range-Doppler processing
│   └── tracking.py      # Target tracking algorithms
└── utils/              # Utility functions
    ├── __init__.py
    ├── conversions.py   # Unit conversions
    └── coordinates.py   # Coordinate transformations

examples/
├── basic.py            # Basic radar simulation
├── static.py           # 3D visualization example
└── live_3d.py          # Live simulation with Dash
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pydar

# Run specific test file
pytest tests/test_radar.py
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on radar theory from Skolnik's "Radar Handbook"
- Signal processing algorithms adapted from Richards' "Fundamentals of Radar Signal Processing"
