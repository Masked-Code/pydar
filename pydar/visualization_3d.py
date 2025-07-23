"""
3D radar visualization module.

This module provides real-time 3D visualization of radar data including
PPI displays, 3D scatter plots, and volumetric rendering.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
from dataclasses import dataclass
import colorsys


@dataclass
class VisualizationConfig:
    """Configuration for 3D visualization."""
    
    # Display settings
    figure_width: int = 1200
    figure_height: int = 800
    theme: str = 'plotly_dark'
    
    # Radar display settings
    max_range: float = 50000.0  # meters
    range_rings: int = 5
    azimuth_lines: int = 12
    elevation_lines: int = 6
    
    # Color settings
    detection_color: str = 'lime'
    track_color: str = 'cyan'
    clutter_color: str = 'orange'
    beam_color: str = 'yellow'
    
    # Update settings
    update_interval: float = 0.1  # seconds
    trail_length: int = 20  # Number of historical points to show
    
    # 3D settings
    camera_distance: float = 2.0
    camera_elevation: float = 30.0
    camera_azimuth: float = 45.0


class Radar3DVisualizer:
    """3D visualization for live radar data."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize 3D visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Figure and layout
        self.fig = None
        self.traces = {}
        
        # Data buffers
        self.detection_buffer = []
        self.track_buffer = {}
        self.beam_positions = []
        self.scan_data = None
        
        # Threading
        self._update_thread = None
        self._stop_event = threading.Event()
        self.is_running = False
        
        # Initialize figure
        self._initialize_figure()
    
    def _initialize_figure(self):
        """Initialize the plotly figure with subplots."""
        # Create subplots with different views
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D View', 'PPI Display', 'Range-Height', 'Track View'),
            specs=[
                [{'type': 'scatter3d', 'rowspan': 2}, {'type': 'polar'}],
                [None, {'type': 'scatter'}]
            ],
            row_heights=[0.6, 0.4],
            column_widths=[0.6, 0.4]
        )
        
        # Initialize empty traces
        self._add_coordinate_system()
        self._add_range_rings()
        self._add_radar_beam()
        self._add_detection_traces()
        self._add_track_traces()
        
        # Update layout
        self._update_layout()
    
    def _add_coordinate_system(self):
        """Add coordinate system to 3D view."""
        # Add ground plane grid
        x_grid = np.linspace(-self.config.max_range, self.config.max_range, 20)
        y_grid = np.linspace(-self.config.max_range, self.config.max_range, 20)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        z_mesh = np.zeros_like(x_mesh)
        
        self.fig.add_trace(
            go.Surface(
                x=x_mesh,
                y=y_mesh,
                z=z_mesh,
                colorscale='Greys',
                showscale=False,
                opacity=0.1,
                name='Ground Plane'
            ),
            row=1, col=1
        )
    
    def _add_range_rings(self):
        """Add range rings to displays."""
        # 3D range rings
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
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    name=f'Range {r/1000:.0f}km'
                ),
                row=1, col=1
            )
        
        # PPI range rings (polar)
        for i in range(1, self.config.range_rings + 1):
            r = self.config.max_range * i / self.config.range_rings
            self.fig.add_trace(
                go.Scatterpolar(
                    r=[r] * 360,
                    theta=list(range(360)),
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    name=f'Range {r/1000:.0f}km'
                ),
                row=1, col=2
            )
    
    def _add_radar_beam(self):
        """Add radar beam visualization."""
        # Initialize with dummy data
        self.traces['beam'] = go.Cone(
            x=[0], y=[0], z=[0],
            u=[1], v=[0], w=[0],
            sizemode='absolute',
            sizeref=self.config.max_range/10,
            colorscale='Viridis',
            showscale=False,
            opacity=0.3,
            name='Radar Beam'
        )
        self.fig.add_trace(self.traces['beam'], row=1, col=1)
    
    def _add_detection_traces(self):
        """Add detection point traces."""
        # 3D detections
        self.traces['detections_3d'] = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(
                color=self.config.detection_color,
                size=6,
                symbol='circle'
            ),
            name='Detections'
        )
        self.fig.add_trace(self.traces['detections_3d'], row=1, col=1)
        
        # PPI detections
        self.traces['detections_ppi'] = go.Scatterpolar(
            r=[], theta=[],
            mode='markers',
            marker=dict(
                color=self.config.detection_color,
                size=6
            ),
            showlegend=False,
            name='Detections'
        )
        self.fig.add_trace(self.traces['detections_ppi'], row=1, col=2)
        
        # Range-Height detections
        self.traces['detections_rh'] = go.Scatter(
            x=[], y=[],
            mode='markers',
            marker=dict(
                color=self.config.detection_color,
                size=6
            ),
            showlegend=False,
            name='Detections'
        )
        self.fig.add_trace(self.traces['detections_rh'], row=2, col=2)
    
    def _add_track_traces(self):
        """Add track visualization traces."""
        # 3D tracks
        self.traces['tracks_3d'] = go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines+markers',
            line=dict(
                color=self.config.track_color,
                width=3
            ),
            marker=dict(
                color=self.config.track_color,
                size=8,
                symbol='diamond'
            ),
            name='Tracks'
        )
        self.fig.add_trace(self.traces['tracks_3d'], row=1, col=1)
    
    def _update_layout(self):
        """Update figure layout."""
        self.fig.update_layout(
            template=self.config.theme,
            width=self.config.figure_width,
            height=self.config.figure_height,
            title='Live Radar 3D Visualization',
            showlegend=True,
            legend=dict(x=0.85, y=0.95)
        )
        
        # 3D scene settings
        self.fig.update_scenes(
            xaxis=dict(
                title='East (m)',
                range=[-self.config.max_range, self.config.max_range],
                gridcolor='gray',
                gridwidth=1
            ),
            yaxis=dict(
                title='North (m)',
                range=[-self.config.max_range, self.config.max_range],
                gridcolor='gray',
                gridwidth=1
            ),
            zaxis=dict(
                title='Height (m)',
                range=[0, self.config.max_range/2],
                gridcolor='gray',
                gridwidth=1
            ),
            camera=dict(
                eye=dict(
                    x=self.config.camera_distance * np.cos(np.radians(self.config.camera_azimuth)),
                    y=self.config.camera_distance * np.sin(np.radians(self.config.camera_azimuth)),
                    z=self.config.camera_distance * np.sin(np.radians(self.config.camera_elevation))
                )
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        )
        
        # Polar subplot settings
        self.fig.update_polars(
            radialaxis=dict(
                range=[0, self.config.max_range],
                title='Range (m)'
            ),
            angularaxis=dict(
                direction='clockwise',
                rotation=90
            )
        )
        
        # Range-Height subplot settings
        self.fig.update_xaxes(
            title_text='Ground Range (m)',
            range=[0, self.config.max_range],
            row=2, col=2
        )
        self.fig.update_yaxes(
            title_text='Height (m)',
            range=[0, self.config.max_range/2],
            row=2, col=2
        )
    
    def update_detections(self, detections: List[Dict[str, Any]]):
        """Update detection displays."""
        if not detections:
            return
        
        # Add to buffer with timestamp
        current_time = time.time()
        for det in detections:
            self.detection_buffer.append({
                **det,
                'timestamp': current_time
            })
        
        # Remove old detections
        cutoff_time = current_time - 5.0  # Keep last 5 seconds
        self.detection_buffer = [d for d in self.detection_buffer 
                                if d['timestamp'] > cutoff_time]
        
        # Convert to coordinates
        x, y, z = [], [], []
        r, theta = [], []
        range_ground, height = [], []
        
        for det in self.detection_buffer:
            # Convert spherical to Cartesian
            r_det = det['range']
            az_rad = np.radians(det['azimuth'])
            el_rad = np.radians(det['elevation'])
            
            x_pos = r_det * np.cos(el_rad) * np.sin(az_rad)
            y_pos = r_det * np.cos(el_rad) * np.cos(az_rad)
            z_pos = r_det * np.sin(el_rad)
            
            x.append(x_pos)
            y.append(y_pos)
            z.append(z_pos)
            
            # PPI coordinates
            r.append(r_det * np.cos(el_rad))
            theta.append(det['azimuth'])
            
            # Range-Height coordinates
            range_ground.append(r_det * np.cos(el_rad))
            height.append(z_pos)
        
        # Update traces
        self.fig.data[self.fig.data.index(self.traces['detections_3d'])].update(
            x=x, y=y, z=z
        )
        self.fig.data[self.fig.data.index(self.traces['detections_ppi'])].update(
            r=r, theta=theta
        )
        self.fig.data[self.fig.data.index(self.traces['detections_rh'])].update(
            x=range_ground, y=height
        )
    
    def update_tracks(self, tracks: Dict[int, Dict[str, Any]]):
        """Update track displays."""
        # Clear track buffer
        self.track_buffer = {}
        
        # Process each track
        all_x, all_y, all_z = [], [], []
        
        for track_id, track in tracks.items():
            if 'detections' not in track or not track['detections']:
                continue
            
            # Get track history
            track_history = track['detections'][-self.config.trail_length:]
            
            # Convert to coordinates
            x, y, z = [], [], []
            for det in track_history:
                r_det = det['range']
                az_rad = np.radians(det['azimuth'])
                el_rad = np.radians(det['elevation'])
                
                x_pos = r_det * np.cos(el_rad) * np.sin(az_rad)
                y_pos = r_det * np.cos(el_rad) * np.cos(az_rad)
                z_pos = r_det * np.sin(el_rad)
                
                x.append(x_pos)
                y.append(y_pos)
                z.append(z_pos)
            
            # Add track separator
            if all_x:
                all_x.append(None)
                all_y.append(None)
                all_z.append(None)
            
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)
            
            self.track_buffer[track_id] = {
                'x': x, 'y': y, 'z': z,
                'state': track.get('state', {})
            }
        
        # Update trace
        self.fig.data[self.fig.data.index(self.traces['tracks_3d'])].update(
            x=all_x, y=all_y, z=all_z
        )
    
    def update_beam_position(self, azimuth: float, elevation: float):
        """Update radar beam position."""
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Beam direction vector
        u = np.cos(el_rad) * np.sin(az_rad)
        v = np.cos(el_rad) * np.cos(az_rad)
        w = np.sin(el_rad)
        
        # Update cone trace
        self.fig.data[self.fig.data.index(self.traces['beam'])].update(
            u=[u], v=[v], w=[w]
        )
        
        # Add beam position to history
        self.beam_positions.append({
            'azimuth': azimuth,
            'elevation': elevation,
            'time': time.time()
        })
        
        # Keep only recent positions
        cutoff = time.time() - 10.0
        self.beam_positions = [p for p in self.beam_positions if p['time'] > cutoff]
    
    def update_scan_data(self, scan_result):
        """Update with full scan data for intensity display."""
        if scan_result is None:
            return
        
        # Store for processing
        self.scan_data = scan_result
        
        # Could add intensity heatmaps, range-Doppler plots, etc.
    
    def start(self):
        """Start visualization updates."""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # Show figure
        self.show()
    
    def stop(self):
        """Stop visualization updates."""
        self.is_running = False
        self._stop_event.set()
    
    def show(self):
        """Display the figure."""
        self.fig.show()
    
    def update_from_simulation(self, simulation):
        """Update visualization from live simulation."""
        # Get current state
        state = simulation.get_current_state()
        
        # Update beam position
        self.update_beam_position(
            state['antenna_position']['azimuth'],
            state['antenna_position']['elevation']
        )
        
        # Get recent detections
        recent_detections = []
        try:
            while not simulation.detection_queue.empty():
                recent_detections.extend(simulation.detection_queue.get_nowait())
        except:
            pass
        
        if recent_detections:
            self.update_detections(recent_detections)
        
        # Update tracks
        if state['tracks']:
            self.update_tracks(state['tracks'])
        
        # Get latest scan
        recent_scans = simulation.get_recent_scans(1)
        if recent_scans:
            self.update_scan_data(recent_scans[0])
    
    def create_animation_frame(self) -> go.Frame:
        """Create animation frame for current state."""
        # Collect all current trace data
        data = []
        
        # Add current detection data
        det_3d_idx = self.fig.data.index(self.traces['detections_3d'])
        data.append(go.Scatter3d(
            x=self.fig.data[det_3d_idx].x,
            y=self.fig.data[det_3d_idx].y,
            z=self.fig.data[det_3d_idx].z
        ))
        
        # Add current track data
        track_3d_idx = self.fig.data.index(self.traces['tracks_3d'])
        data.append(go.Scatter3d(
            x=self.fig.data[track_3d_idx].x,
            y=self.fig.data[track_3d_idx].y,
            z=self.fig.data[track_3d_idx].z
        ))
        
        return go.Frame(data=data, name=str(time.time()))
    
    def save_animation(self, filename: str, frames: List[go.Frame]):
        """Save animation to file."""
        # Add frames to figure
        self.fig.frames = frames
        
        # Add animation buttons
        self.fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100},
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }]
        )
        
        # Save to HTML
        self.fig.write_html(filename)
