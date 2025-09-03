import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
from typing import Dict, Optional
import config

_fonts: Optional[Dict[str, pygame.font.Font]] = None


def init_fonts() -> Dict[str, pygame.font.Font]:
    """
    Initialize all fonts used in the application.
    
    This function must be called after pygame.init() and before
    using any font objects.
    
    Returns:
        Dictionary containing all initialized font objects
    """
    global _fonts
    
    pygame.init()
    
    _fonts = {
        'default': pygame.font.SysFont("consolas", config.FONT_SIZE_DEFAULT_PT),
        'small': pygame.font.SysFont("consolas", config.FONT_SIZE_SMALL_PT),
        'large': pygame.font.SysFont("consolas", config.FONT_SIZE_LARGE_PT, bold=True)
    }
    
    return _fonts


def get_fonts() -> Dict[str, pygame.font.Font]:
    """
    Get the dictionary of initialized fonts.
    
    Returns:
        Dictionary containing all font objects
        
    Raises:
        RuntimeError: If fonts have not been initialized
    """
    if _fonts is None:
        raise RuntimeError("Fonts not initialized. Call init_fonts() first.")
    
    return _fonts


def get_font(name: str) -> pygame.font.Font:
    """
    Get a specific font by name.
    
    Args:
        name: Font name ('default', 'small', or 'large')
        
    Returns:
        The requested font object
        
    Raises:
        RuntimeError: If fonts have not been initialized
        KeyError: If the requested font name doesn't exist
    """
    fonts = get_fonts()
    return fonts[name]


def get_default_font() -> pygame.font.Font:
    """Get the default font."""
    return get_font('default')


def get_small_font() -> pygame.font.Font:
    """Get the small font."""
    return get_font('small')


def get_large_font() -> pygame.font.Font:
    """Get the large font."""
    return get_font('large')
