"""
PyDar - Python Radar Simulator

A high-fidelity radar simulator for educational and research purposes.
"""

__version__ = "0.1.0"
__author__ = "PyDar Development Team"

# Import main classes for easier access
from .radar import RadarSystem, Antenna
from .target import Target, TargetCollection
from .environment import Environment
from .waveforms import LinearFMChirp, PulseTrain, Waveform

__all__ = [
    "RadarSystem",
    "Antenna",
    "Target",
    "TargetCollection",
    "Environment",
    "LinearFMChirp",
    "PulseTrain",
    "Waveform",
]
