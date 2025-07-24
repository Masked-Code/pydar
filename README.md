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

### Static Simulation Example

```python
from pydar import RadarSystem, Antenna, LinearFMChirp, Target, Environment
from pydar import Radar3DVisualizer

# Create radar system
antenna = Antenna(beamwidth_azimuth=3.0, beamwidth_elevation=3.0, gain=30.0)
waveform = LinearFMChirp(center_frequency=10e9, bandwidth=50e6, pulse_width=10e-6)
radar = RadarSystem(antenna=antenna, waveform=waveform, transmit_power=1000)

# Create target
target = Target(position=[10000, 5000, 3000], velocity=[-100, 0, 0], rcs=10.0)

# Create environment and simulate
environment = Environment()
environment.add_target(target)

# Perform scan
scan_result = radar.scan(environment)

# Visualize
visualizer = Radar3DVisualizer()
fig = visualizer.visualize(radar, environment, scan_result=scan_result)
fig.show()
```

### Live Simulation Example

```python
from pydar import LiveRadarSimulation, SimulationConfig, DashRadarVisualizer
import threading

# Configure simulation
config = SimulationConfig(
    duration=60.0,
    update_rate=20.0,
    scan_rate=5.0,
    enable_tracking=True
)

# Create and run simulation
simulation = LiveRadarSimulation(radar, environment, config)
visualizer = DashRadarVisualizer(simulation)

# Start simulation in background
sim_thread = threading.Thread(target=simulation.run, daemon=True)
sim_thread.start()

# Run visualization (opens web browser)
visualizer.run()
```

## Documentation

Full documentation is available in the `docs/` directory:

- [User Guide](docs/user_guide.md) - Detailed usage instructions
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Theory Guide](docs/theory.md) - Radar theory and implementation details
- [Examples](examples/) - Example scripts and notebooks

## Project Structure

```
pydar/
├── __init__.py          # Package initialization
├── radar.py             # Radar system and antenna models
├── target.py            # Target models and dynamics
├── environment.py       # Environmental effects and clutter
├── waveforms.py         # Waveform definitions
├── visualization.py     # Unified visualization module
├── live_simulation.py   # Real-time simulation engine
├── scan_result.py       # Scan result data structures
├── processing/          # Signal processing algorithms
│   ├── __init__.py
│   ├── cfar.py         # CFAR detectors
│   ├── range_doppler.py # Range-Doppler processing
│   └── tracking.py      # Target tracking algorithms
├── analysis/            # Analysis tools
│   ├── __init__.py
│   └── performance.py   # Performance metrics
└── utils/              # Utility functions
    ├── __init__.py
    ├── constants.py    # Physical constants
    ├── geometry.py     # Coordinate transformations
    └── signal.py       # Signal processing utilities

examples/
├── static_simulation.py  # Static simulation examples
├── live_simulation.py    # Live simulation with visualization
└── ...                  # Additional examples
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
