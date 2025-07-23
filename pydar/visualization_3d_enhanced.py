"""
Enhanced 3D radar visualization with animated wave propagation.

This module provides real-time visualization of radar waves propagating
through space, reflecting off targets, and returning to the radar.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import time
from dataclasses import dataclass, field
import colorsys
from collections import deque


@dataclass
class WavePulse:
    """Represents a radar pulse propagating through space."""
    
    origin: Tuple[float, float, float]  # Starting position
    direction: Tuple[float, float, float]  # Unit direction vector
    current_distance: float = 0.0  # Current distance from origin
    max_distance: float = 50000.0  # Maximum range
    creation_time: float = field(default_factory=time.time)
    speed: float = 299792458.0  # Speed of light in m/s
    frequency: float = 10e9  # Radar frequency
    power: float = 1.0  # Normalized power
    is_return: bool = False  # Is this a return pulse?
    target_hit: Optional[Dict[str, Any]] = None  # Target information if hit


@dataclass
class EnhancedVisualizationConfig:
    """Configuration for enhanced 3D visualization."""
    
    # Display settings
    figure_width: int = 1400
    figure_height: int = 900
    theme: str = 'plotly_dark'
    
    # Radar display settings
    max_range: float = 50000.0  # meters
    range_rings: int = 5
    
    # Wave visualization
    wave_speed_factor: float = 10000.0  # Speed up factor for visualization
    pulse_width: float = 100.0  # Visual width of pulse in meters
    wave_trail_length: int = 10  # Number of wave positions to show
    wave_opacity: float = 0.6
    
    # Colors
    outgoing_wave_color: str = '#00ff00'  # Green for outgoing
    return_wave_color: str = '#ff0000'  # Red for returns
    target_color: str = '#00ffff'  # Cyan for targets
    detection_color: str = '#ffff00'  # Yellow for detections
    
    # Animation settings
    frame_rate: float = 60.0  # FPS for smooth animation
    show_wave_fronts: bool = True
    show_power_decay: bool = True
    show_doppler_shift: bool = True


class Enhanced3DRadarVisualizer:
    """Enhanced 3D visualization with animated radar wave propagation."""
    
    def __init__(self, config: Optional[EnhancedVisualizationConfig] = None):
        """Initialize enhanced visualizer."""
        self.config = config or EnhancedVisualizationConfig()
        
        # Wave management
        self.active_pulses: List[WavePulse] = []
        self.pulse_history = deque(maxlen=100)
        
        # Target and detection management
        self.targets = {}
        self.detections = deque(maxlen=500)
        self.tracks = {}
        
        # Radar state
        self.radar_position = (0, 0, 0)
        self.current_azimuth = 0.0
        self.current_elevation = 0.0
        self.radar_params = {
            'frequency': 10e9,
            'power': 10000,
            'gain': 35
        }
        
        # Animation state
        self.animation_time = 0.0
        self.last_pulse_time = 0.0
        self.pulse_interval = 0.0005  # 500 microseconds PRI
        
        # Threading
        self._animation_thread = None
        self._stop_event = threading.Event()
        self.is_running = False
        
        # Initialize figure
        self._initialize_figure()
    
    def _initialize_figure(self):
        """Initialize the enhanced plotly figure."""
        # Create figure with 3D scene
        self.fig = go.Figure()
        
        # Add ground plane
        self._add_ground_plane()
        
        # Add range rings
        self._add_range_rings()
        
        # Add radar system
        self._add_radar_system()
        
        # Initialize traces for dynamic elements
        self._initialize_dynamic_traces()
        
        # Configure layout
        self._configure_layout()
    
    def _add_ground_plane(self):
        """Add a ground plane with grid."""
        x = np.linspace(-self.config.max_range, self.config.max_range, 50)
        y = np.linspace(-self.config.max_range, self.config.max_range, 50)
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = np.zeros_like(x_grid)
        
        # Add subtle terrain variation
        z_grid += 100 * np.sin(x_grid/5000) * np.sin(y_grid/5000)
        
        self.fig.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale=[[0, 'rgb(20,20,20)'], [1, 'rgb(40,40,40)']],
                showscale=False,
                opacity=0.7,
                name='Ground'
            )
        )
    
    def _add_range_rings(self):
        """Add 3D range rings."""
        for i in range(1, self.config.range_rings + 1):
            r = self.config.max_range * i / self.config.range_rings
            theta = np.linspace(0, 2*np.pi, 100)
            
            # Ground level rings
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.zeros_like(theta)
            
            self.fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.5)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # Add range labels
            self.fig.add_trace(
                go.Scatter3d(
                    x=[r], y=[0], z=[100],
                    mode='text',
                    text=[f'{r/1000:.0f}km'],
                    textfont=dict(color='white', size=10),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    
    def _add_radar_system(self):
        """Add 3D radar system model."""
        # Radar base/pedestal
        base_r = 50
        base_h = 100
        theta = np.linspace(0, 2*np.pi, 20)
        
        base_x = base_r * np.cos(theta)
        base_y = base_r * np.sin(theta)
        base_z_bottom = np.zeros_like(theta)
        base_z_top = np.full_like(theta, base_h)
        
        # Add radar pedestal
        self.fig.add_trace(
            go.Scatter3d(
                x=np.concatenate([base_x, base_x]),
                y=np.concatenate([base_y, base_y]),
                z=np.concatenate([base_z_bottom, base_z_top]),
                mode='lines',
                line=dict(color='silver', width=3),
                showlegend=False,
                name='Radar Base'
            )
        )
        
        # Radar antenna (will be updated dynamically)
        self.antenna_trace = go.Cone(
            x=[0], y=[0], z=[base_h],
            u=[1], v=[0], w=[0],
            sizemode='absolute',
            sizeref=200,
            colorscale='Blues',
            showscale=False,
            opacity=0.8,
            name='Radar Antenna'
        )
        self.fig.add_trace(self.antenna_trace)
    
    def _initialize_dynamic_traces(self):
        """Initialize traces for dynamic elements."""
        # Outgoing wave pulses
        self.wave_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(
                size=[],
                color=[],
                colorscale='Viridis',
                opacity=0.6,
                symbol='circle'
            ),
            name='Radar Waves',
            showlegend=True
        )
        self.fig.add_trace(self.wave_trace)
        
        # Wave fronts (spherical surfaces)
        self.wavefront_traces = []
        
        # Target markers
        self.target_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers+text',
            marker=dict(
                size=10,
                color=self.config.target_color,
                symbol='diamond'
            ),
            text=[],
            textposition='top center',
            name='Targets',
            showlegend=True
        )
        self.fig.add_trace(self.target_trace)
        
        # Detection flashes
        self.detection_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(
                size=[],
                color=self.config.detection_color,
                opacity=0.8,
                symbol='circle-open'
            ),
            name='Detections',
            showlegend=True
        )
        self.fig.add_trace(self.detection_trace)
        
        # Return signals
        self.return_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(
                size=6,
                color=self.config.return_wave_color,
                opacity=0.7
            ),
            name='Returns',
            showlegend=True
        )
        self.fig.add_trace(self.return_trace)
    
    def _configure_layout(self):
        """Configure the figure layout."""
        self.fig.update_layout(
            template=self.config.theme,
            width=self.config.figure_width,
            height=self.config.figure_height,
            title=dict(
                text='Live 3D Radar Simulation - Wave Propagation',
                font=dict(size=24)
            ),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            scene=dict(
                xaxis=dict(
                    title='East (m)',
                    range=[-self.config.max_range, self.config.max_range],
                    gridcolor='rgba(100,100,100,0.3)',
                    zerolinecolor='rgba(100,100,100,0.5)'
                ),
                yaxis=dict(
                    title='North (m)',
                    range=[-self.config.max_range, self.config.max_range],
                    gridcolor='rgba(100,100,100,0.3)',
                    zerolinecolor='rgba(100,100,100,0.5)'
                ),
                zaxis=dict(
                    title='Height (m)',
                    range=[0, self.config.max_range/2],
                    gridcolor='rgba(100,100,100,0.3)'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5),
                bgcolor='rgb(10,10,10)'
            ),
            paper_bgcolor='rgb(0,0,0)',
            plot_bgcolor='rgb(0,0,0)'
        )
    
    def emit_pulse(self, azimuth: float, elevation: float):
        """Emit a new radar pulse."""
        # Calculate direction vector
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        direction = (
            np.cos(el_rad) * np.sin(az_rad),
            np.cos(el_rad) * np.cos(az_rad),
            np.sin(el_rad)
        )
        
        # Create new pulse
        pulse = WavePulse(
            origin=self.radar_position,
            direction=direction,
            max_distance=self.config.max_range,
            frequency=self.radar_params['frequency'],
            power=1.0
        )
        
        self.active_pulses.append(pulse)
    
    def update_pulse_positions(self, dt: float):
        """Update positions of all active pulses."""
        pulses_to_remove = []
        
        for i, pulse in enumerate(self.active_pulses):
            # Update distance based on speed of light (scaled for visualization)
            pulse.current_distance += pulse.speed * dt / self.config.wave_speed_factor
            
            # Check if pulse has reached max range
            if pulse.current_distance > pulse.max_distance:
                pulses_to_remove.append(i)
                continue
            
            # Calculate current position
            current_pos = (
                pulse.origin[0] + pulse.direction[0] * pulse.current_distance,
                pulse.origin[1] + pulse.direction[1] * pulse.current_distance,
                pulse.origin[2] + pulse.direction[2] * pulse.current_distance
            )
            
            # Check for target intersections
            if not pulse.is_return:
                for target_id, target in self.targets.items():
                    target_pos = target['position']
                    distance_to_target = np.sqrt(
                        (current_pos[0] - target_pos[0])**2 +
                        (current_pos[1] - target_pos[1])**2 +
                        (current_pos[2] - target_pos[2])**2
                    )
                    
                    # Check if pulse hit target (within target size)
                    if distance_to_target < target['size']:
                        # Create return pulse
                        self._create_return_pulse(pulse, target, current_pos)
                        
                        # Create detection flash
                        self._create_detection_flash(target_pos)
                        
                        # Mark original pulse for removal
                        pulses_to_remove.append(i)
                        break
        
        # Remove expired pulses
        for i in reversed(pulses_to_remove):
            self.pulse_history.append(self.active_pulses[i])
            del self.active_pulses[i]
    
    def _create_return_pulse(self, incident_pulse: WavePulse, target: Dict, hit_position: Tuple):
        """Create a return pulse from a target hit."""
        # Calculate return direction (back to radar)
        return_direction = (
            -incident_pulse.direction[0],
            -incident_pulse.direction[1],
            -incident_pulse.direction[2]
        )
        
        # Calculate return power based on radar equation
        incident_power = incident_pulse.power
        target_rcs = target.get('rcs', 10.0)
        range_to_target = incident_pulse.current_distance
        
        # Simplified radar equation for visualization
        return_power = incident_power * target_rcs / (range_to_target**2)
        return_power = min(return_power, 1.0)  # Normalize
        
        # Create return pulse
        return_pulse = WavePulse(
            origin=hit_position,
            direction=return_direction,
            current_distance=0,
            max_distance=range_to_target,
            frequency=incident_pulse.frequency,
            power=return_power,
            is_return=True,
            target_hit=target
        )
        
        self.active_pulses.append(return_pulse)
    
    def _create_detection_flash(self, position: Tuple):
        """Create a visual flash at detection point."""
        self.detections.append({
            'position': position,
            'time': time.time(),
            'intensity': 1.0
        })
    
    def update_visualization(self):
        """Update all visualization elements."""
        # Update antenna pointing
        self._update_antenna()
        
        # Update wave positions
        self._update_wave_display()
        
        # Update targets
        self._update_target_display()
        
        # Update detections
        self._update_detection_display()
        
        # Update wavefronts if enabled
        if self.config.show_wave_fronts:
            self._update_wavefronts()
    
    def _update_antenna(self):
        """Update antenna pointing direction."""
        az_rad = np.radians(self.current_azimuth)
        el_rad = np.radians(self.current_elevation)
        
        # Update cone direction
        u = np.cos(el_rad) * np.sin(az_rad)
        v = np.cos(el_rad) * np.cos(az_rad)
        w = np.sin(el_rad)
        
        antenna_idx = None
        for i, trace in enumerate(self.fig.data):
            if trace.name == 'Radar Antenna':
                antenna_idx = i
                break
        
        if antenna_idx is not None:
            self.fig.data[antenna_idx].update(
                u=[u], v=[v], w=[w]
            )
    
    def _update_wave_display(self):
        """Update wave pulse display."""
        if not self.active_pulses:
            return
        
        # Collect all pulse positions
        x, y, z = [], [], []
        sizes = []
        colors = []
        
        for pulse in self.active_pulses:
            # Calculate current position
            pos_x = pulse.origin[0] + pulse.direction[0] * pulse.current_distance
            pos_y = pulse.origin[1] + pulse.direction[1] * pulse.current_distance
            pos_z = pulse.origin[2] + pulse.direction[2] * pulse.current_distance
            
            # Add multiple points along pulse width for better visualization
            num_points = 5
            for i in range(num_points):
                offset = (i - num_points/2) * self.config.pulse_width / num_points
                x.append(pos_x + pulse.direction[0] * offset)
                y.append(pos_y + pulse.direction[1] * offset)
                z.append(pos_z + pulse.direction[2] * offset)
                
                # Size based on power and distance
                if self.config.show_power_decay:
                    size = 10 * pulse.power / (1 + pulse.current_distance/10000)
                else:
                    size = 8
                sizes.append(size)
                
                # Color based on direction
                if pulse.is_return:
                    colors.append(self.config.return_wave_color)
                else:
                    colors.append(self.config.outgoing_wave_color)
        
        # Update wave trace
        wave_idx = None
        for i, trace in enumerate(self.fig.data):
            if trace.name == 'Radar Waves':
                wave_idx = i
                break
        
        if wave_idx is not None:
            self.fig.data[wave_idx].update(
                x=x, y=y, z=z,
                marker=dict(size=sizes, color=colors)
            )
    
    def _update_target_display(self):
        """Update target positions and labels."""
        if not self.targets:
            return
        
        x, y, z = [], [], []
        texts = []
        
        for target_id, target in self.targets.items():
            pos = target['position']
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
            
            # Create label with target info
            vel = target.get('velocity', [0, 0, 0])
            speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            texts.append(f"T{target_id}<br>{speed:.0f}m/s")
        
        # Update target trace
        target_idx = None
        for i, trace in enumerate(self.fig.data):
            if trace.name == 'Targets':
                target_idx = i
                break
        
        if target_idx is not None:
            self.fig.data[target_idx].update(
                x=x, y=y, z=z, text=texts
            )
    
    def _update_detection_display(self):
        """Update detection flashes."""
        current_time = time.time()
        
        x, y, z = [], [], []
        sizes = []
        
        for detection in self.detections:
            age = current_time - detection['time']
            if age < 1.0:  # Show for 1 second
                pos = detection['position']
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
                
                # Expanding ring effect
                size = 20 + age * 30
                sizes.append(size)
        
        # Update detection trace
        det_idx = None
        for i, trace in enumerate(self.fig.data):
            if trace.name == 'Detections':
                det_idx = i
                break
        
        if det_idx is not None:
            self.fig.data[det_idx].update(
                x=x, y=y, z=z,
                marker=dict(size=sizes)
            )
    
    def _update_wavefronts(self):
        """Update spherical wavefront surfaces."""
        # This would add spherical surfaces for major wavefronts
        # Simplified for performance
        pass
    
    def animation_loop(self):
        """Main animation loop."""
        last_frame_time = time.time()
        frame_interval = 1.0 / self.config.frame_rate
        
        while not self._stop_event.is_set():
            current_time = time.time()
            dt = current_time - last_frame_time
            
            if dt >= frame_interval:
                # Update animation time
                self.animation_time += dt
                
                # Emit new pulse if interval reached
                if current_time - self.last_pulse_time >= self.pulse_interval:
                    self.emit_pulse(self.current_azimuth, self.current_elevation)
                    self.last_pulse_time = current_time
                
                # Update pulse positions
                self.update_pulse_positions(dt)
                
                # Update visualization
                self.update_visualization()
                
                last_frame_time = current_time
            else:
                time.sleep(0.001)
    
    def start(self):
        """Start the animation."""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # Start animation thread
        self._animation_thread = threading.Thread(target=self.animation_loop)
        self._animation_thread.start()
        
        # Show figure
        self.fig.show()
    
    def stop(self):
        """Stop the animation."""
        self.is_running = False
        self._stop_event.set()
        if self._animation_thread:
            self._animation_thread.join()
    
    def set_antenna_position(self, azimuth: float, elevation: float):
        """Set antenna pointing angles."""
        self.current_azimuth = azimuth
        self.current_elevation = elevation
    
    def add_target(self, target_id: str, position: Tuple[float, float, float],
                   velocity: Optional[Tuple[float, float, float]] = None,
                   rcs: float = 10.0, size: float = 100.0):
        """Add or update a target."""
        self.targets[target_id] = {
            'position': position,
            'velocity': velocity or (0, 0, 0),
            'rcs': rcs,
            'size': size,
            'last_hit': 0
        }
    
    def update_target_positions(self, dt: float):
        """Update target positions based on velocity."""
        for target in self.targets.values():
            vel = target['velocity']
            pos = target['position']
            target['position'] = (
                pos[0] + vel[0] * dt,
                pos[1] + vel[1] * dt,
                pos[2] + vel[2] * dt
            )
