"""
PyDar - Python Radar Simulator

A high-fidelity radar simulator for educational and research purposes.
"""

__version__ = "0.1.0"
__author__ = "PyDar Development Team"

# Import main classes for easier access
from .radar import RadarSystem, Antenna
from .target import Target, TargetCollection, ExtendedTarget, SwerlingModel
from .environment import Environment, Atmosphere, SeaClutter, LandClutter
from .waveforms import (
    LinearFMChirp, PulseTrain, Waveform, BarkerCode,
    SteppedFrequency, CustomWaveform
)
from .scan_result import ScanResult
from .live_simulation import LiveRadarSimulation, SimulationConfig
from .visualization_3d import Radar3DVisualizer, VisualizationConfig
from .visualization_3d_enhanced import Enhanced3DRadarVisualizer, EnhancedVisualizationConfig

__all__ = [
    # Core radar classes
    "RadarSystem",
    "Antenna",
    
    # Target classes
    "Target",
    "TargetCollection",
    "ExtendedTarget",
    "SwerlingModel",
    
    # Environment classes
    "Environment",
    "Atmosphere",
    "SeaClutter",
    "LandClutter",
    
    # Waveform classes
    "LinearFMChirp",
    "PulseTrain",
    "Waveform",
    "BarkerCode",
    "SteppedFrequency",
    "CustomWaveform",
    
    # Results and analysis
    "ScanResult",
    
    # Live simulation
    "LiveRadarSimulation",
    "SimulationConfig",
    "Radar3DVisualizer",
    "VisualizationConfig",
    "Enhanced3DRadarVisualizer",
    "EnhancedVisualizationConfig",
]
