# PyDar - Python Radar Simulator

A high-fidelity radar simulator written in Python, focusing on accurate physics modeling and educational visualization.

## Features

- **Accurate Physics Modeling**: Implements fundamental radar equations and signal processing
- **Multiple Radar Types**: Pulse, FMCW (Frequency Modulated Continuous Wave), and phased array support
- **Target Modeling**: Point targets, extended targets, and clutter simulation
- **Signal Processing**: Range-Doppler processing, CFAR detection, and tracking algorithms
- **Live 3D Visualization**: Real-time 3D radar simulation with continuous scanning
- **Interactive Displays**: 3D scatter plots, PPI displays, range-height plots, and track visualization
- **Real-time Simulation**: Multi-threaded architecture for smooth live updates
- **Target Tracking**: Automatic target detection and tracking with visualization
- **Extensible Architecture**: Plugin system for custom waveforms and processing algorithms

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

### Basic Simulation

```python
from pydar import RadarSystem, Target, Environment, LinearFMChirp

# Create a radar system
radar = RadarSystem(
    frequency=10e9,  # 10 GHz
    power=1000,      # 1 kW
    antenna_gain=30  # 30 dB
)

# Define a waveform
waveform = LinearFMChirp(
    bandwidth=50e6,  # 50 MHz
    duration=10e-6   # 10 microseconds
)

# Create environment with a target
env = Environment()
env.add_target(Target(range=5000, velocity=50, rcs=10))

# Run simulation
results = radar.scan(env, waveform)
```

### Live 3D Simulation

```python
from pydar import LiveRadarSimulation, Radar3DVisualizer
from pydar import SimulationConfig, VisualizationConfig

# Configure live simulation
sim_config = SimulationConfig(
    update_rate=30.0,  # 30 FPS
    scan_rate=2.0,     # 2 scans/second
    max_range=30000    # 30 km max range
)

# Create live simulation
simulation = LiveRadarSimulation(radar, env, waveform, sim_config)

# Create 3D visualizer
visualizer = Radar3DVisualizer()

# Start live simulation
simulation.start()
visualizer.start()

# Update visualization in real-time
while simulation.is_running:
    visualizer.update_from_simulation(simulation)
    time.sleep(0.1)
```

### Enhanced Wave Propagation Visualization

```python
from pydar import Enhanced3DRadarVisualizer, EnhancedVisualizationConfig

# Configure enhanced visualization
config = EnhancedVisualizationConfig(
    wave_speed_factor=5000,  # Speed up waves for visibility
    show_power_decay=True,   # Show signal attenuation
    frame_rate=60.0         # 60 FPS smooth animation
)

# Create enhanced visualizer
visualizer = Enhanced3DRadarVisualizer(config)

# Add targets to scene
visualizer.add_target(
    target_id="aircraft",
    position=(10000, 5000, 3000),  # x, y, z in meters
    velocity=(150, 0, 0),           # velocity vector
    rcs=50.0                        # radar cross section
)

# Start animated visualization
visualizer.start()

# Emit radar pulses
visualizer.emit_pulse(azimuth=30, elevation=10)
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
├── pydar/              # Main package
│   ├── __init__.py
│   ├── radar.py        # Core radar system classes
│   ├── target.py       # Target modeling
│   ├── waveforms.py    # Waveform generators
│   ├── environment.py  # Environmental modeling
│   ├── processing/     # Signal processing algorithms
│   ├── display/        # Visualization components
│   └── utils/          # Utility functions
├── tests/              # Test suite
├── examples/           # Example scripts
├── docs/               # Documentation
├── setup.py            # Package setup
├── requirements.txt    # Dependencies
└── README.md           # This file
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
