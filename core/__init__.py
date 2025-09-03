"""
Core utilities and shared functionality for the radar simulation.

This package provides:
- Mathematical utility functions (utils.py)
- Font management and initialization (fonts.py)
"""

from .utils import wrap_angle_deg, lerp, pol2cart, cart2pol, clamp
from .fonts import init_fonts, get_fonts

__all__ = [
    'wrap_angle_deg',
    'lerp', 
    'pol2cart',
    'cart2pol',
    'clamp',
    'init_fonts',
    'get_fonts'
]
