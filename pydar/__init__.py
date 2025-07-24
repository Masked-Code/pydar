"""
PyDar - Python Radar Simulator

A high-fidelity radar simulator for educational and research purposes.
"""

__version__ = "0.1.0"
__author__ = "PyDar Development Team"

# Import main classes for easier access
from .radar import RadarSystem, Antenna
from .target import Target, TargetCollection
from .environment import Environment, Atmosphere
from .waveforms import LinearFMChirp
from .visualization import Radar3DVisualizer, DashRadarVisualizer, VisualizationConfig

__all__ = [
    # Core radar classes
    "RadarSystem",
    "Antenna",
    
    # Target classes
    "Target",
    "TargetCollection",
    
    # Environment classes
    "Environment",
    "Atmosphere",
    
    # Waveform classes
    "LinearFMChirp",
    
    # Visualization
    "Radar3DVisualizer",
    "DashRadarVisualizer",
    "VisualizationConfig",
]
