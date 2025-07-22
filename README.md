# PyDar - Python Radar Simulator

A high-fidelity radar simulator written in Python, focusing on accurate physics modeling and educational visualization.

## Features

- **Accurate Physics Modeling**: Implements fundamental radar equations and signal processing
- **Multiple Radar Types**: Pulse, FMCW (Frequency Modulated Continuous Wave), and phased array support
- **Target Modeling**: Point targets, extended targets, and clutter simulation
- **Signal Processing**: Range-Doppler processing, CFAR detection, and tracking algorithms
- **Visualization**: Real-time displays including PPI (Plan Position Indicator) and A-scope
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

```python
from pydar import RadarSystem, Target, Environment
from pydar.waveforms import LinearFMChirp

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

# Process and display results
from pydar.processing import RangeDopplerProcessor
processor = RangeDopplerProcessor()
processed = processor.process(results)
processed.plot()
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
