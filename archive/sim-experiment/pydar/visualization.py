"""
Unified 3D Radar Visualization Module.

This module consolidates all visualization functionality including:
- Basic 3D visualization
- Animated visualization
- Dash-based live visualization
- Enhanced visualization features
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


@dataclass
class VisualizationConfig:
    """Configuration for radar visualization."""
    # Display settings
    figure_width: int = 1200
    figure_height: int = 800
    show_grid: bool = True
    show_axes: bool = True
    
    # Colors
    radar_color: str = 'red'
    target_color: str = 'blue'
    detection_color: str = 'lime'
    track_color: str = 'cyan'
    beam_color: str = 'yellow'
    wave_color: str = 'rgba(0, 255, 255, 0.3)'
    clutter_color: str = 'rgba(128, 128, 128, 0.3)'
    
    # Range settings
    max_range: float = 50000.0  # meters
    max_altitude: float = 10000.0  # meters
    
    # Animation settings
    animation_speed: float = 1.0
    trail_length: int = 10
    
    # Enhanced features
    show_uncertainty: bool = True
    show_doppler: bool = True
    show_rcs: bool = True
    show_statistics: bool = True
    show_waveforms: bool = False
    show_clutter: bool = False


class Radar3DVisualizer:
    """Unified 3D radar visualization with all features."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualizer with configuration."""
        self.config = config or VisualizationConfig()
        self.figure = None
        
    def create_base_figure(self) -> go.Figure:
        """Create the base 3D figure with layout."""
        fig = go.Figure()
        
        # Configure 3D scene
        fig.update_layout(
            width=self.config.figure_width,
            height=self.config.figure_height,
            scene=dict(
                xaxis=dict(
                    title="East (m)",
                    range=[-self.config.max_range, self.config.max_range],
                    showgrid=self.config.show_grid,
                    visible=self.config.show_axes
                ),
                yaxis=dict(
                    title="North (m)",
                    range=[-self.config.max_range, self.config.max_range],
                    showgrid=self.config.show_grid,
                    visible=self.config.show_axes
                ),
                zaxis=dict(
                    title="Altitude (m)",
                    range=[0, self.config.max_altitude],
                    showgrid=self.config.show_grid,
                    visible=self.config.show_axes
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.3),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            title="3D Radar Visualization",
            showlegend=True,
            template="plotly_dark"
        )
        
        return fig
    
    def add_radar(self, fig: go.Figure, radar_system) -> None:
        """Add radar system to the figure."""
        # Radar position
        fig.add_trace(go.Scatter3d(
            x=[radar_system.position[0]],
            y=[radar_system.position[1]],
            z=[radar_system.position[2]],
            mode='markers+text',
            marker=dict(size=10, color=self.config.radar_color, symbol='diamond'),
            text=['Radar'],
            textposition='top center',
            name='Radar',
            showlegend=True
        ))
        
    def add_beam(self, fig: go.Figure, radar_system, beam_azimuth: float, beam_elevation: float) -> None:
        """Add radar beam visualization."""
        # Calculate beam cone
        beam_range = self.config.max_range * 0.8
        az_width = np.radians(radar_system.antenna.beamwidth_azimuth)
        el_width = np.radians(radar_system.antenna.beamwidth_elevation)
        
        # Create beam cone points
        n_points = 20
        theta = np.linspace(0, 2*np.pi, n_points)
        
        # Beam center direction
        az_rad = np.radians(beam_azimuth)
        el_rad = np.radians(beam_elevation)
        
        # Near and far points of beam cone
        near_points = []
        far_points = []
        
        for t in theta:
            # Angular offset from beam center
            d_az = az_width * np.cos(t) / 2
            d_el = el_width * np.sin(t) / 2
            
            # Calculate actual angles
            az = az_rad + d_az
            el = el_rad + d_el
            
            # Near point (at radar)
            near_points.append(radar_system.position)
            
            # Far point
            x = beam_range * np.cos(el) * np.sin(az)
            y = beam_range * np.cos(el) * np.cos(az)
            z = beam_range * np.sin(el) + radar_system.position[2]
            far_points.append([x, y, z])
        
        # Draw beam cone
        for i in range(n_points):
            j = (i + 1) % n_points
            
            # Side face of cone
            x = [near_points[i][0], far_points[i][0], far_points[j][0], near_points[j][0]]
            y = [near_points[i][1], far_points[i][1], far_points[j][1], near_points[j][1]]
            z = [near_points[i][2], far_points[i][2], far_points[j][2], near_points[j][2]]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                color=self.config.beam_color,
                opacity=0.1,
                showscale=False,
                name='Beam' if i == 0 else None,
                showlegend=(i == 0)
            ))
    
    def add_targets(self, fig: go.Figure, targets: List, show_velocity: bool = True) -> None:
        """Add targets to the figure."""
        if not targets:
            return
            
        x, y, z = [], [], []
        text = []
        
        for i, target in enumerate(targets):
            if hasattr(target, 'position'):
                x.append(target.position[0])
                y.append(target.position[1])
                z.append(target.position[2])
            else:
                # Convert from spherical to Cartesian
                r = target.range
                az = np.radians(target.azimuth)
                el = np.radians(target.elevation)
                x.append(r * np.cos(el) * np.sin(az))
                y.append(r * np.cos(el) * np.cos(az))
                z.append(r * np.sin(el))
            
            # Create hover text
            info = f"Target {i}"
            if self.config.show_rcs and hasattr(target, 'rcs'):
                info += f"<br>RCS: {target.rcs:.1f} m²"
            if self.config.show_doppler and hasattr(target, 'velocity'):
                # Handle velocity as either scalar or vector
                if isinstance(target.velocity, (list, np.ndarray)):
                    # Calculate magnitude of velocity vector
                    velocity_magnitude = np.linalg.norm(target.velocity)
                    info += f"<br>Velocity: {velocity_magnitude:.1f} m/s"
                else:
                    info += f"<br>Velocity: {target.velocity:.1f} m/s"
            text.append(info)
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(size=8, color=self.config.target_color),
            text=text,
            textposition='top center',
            name='Targets',
            hoverinfo='text'
        ))
        
        # Add velocity vectors if requested
        if show_velocity and hasattr(targets[0], 'velocity_3d'):
            for i, target in enumerate(targets):
                if hasattr(target, 'position') and hasattr(target, 'velocity_3d'):
                    vel_scale = 100  # Scale factor for visibility
                    fig.add_trace(go.Scatter3d(
                        x=[target.position[0], target.position[0] + vel_scale * target.velocity_3d[0]],
                        y=[target.position[1], target.position[1] + vel_scale * target.velocity_3d[1]],
                        z=[target.position[2], target.position[2] + vel_scale * target.velocity_3d[2]],
                        mode='lines',
                        line=dict(color='orange', width=3),
                        name='Velocity' if i == 0 else None,
                        showlegend=(i == 0)
                    ))
    
    def add_detections(self, fig: go.Figure, detections: List[Dict]) -> None:
        """Add detection points to the figure."""
        if not detections:
            return
            
        x, y, z = [], [], []
        text = []
        
        for det in detections:
            r = det['range']
            az = np.radians(det['azimuth'])
            el = np.radians(det['elevation'])
            
            x.append(r * np.cos(el) * np.sin(az))
            y.append(r * np.cos(el) * np.cos(az))
            z.append(r * np.sin(el))
            
            info = f"Detection<br>Range: {r:.0f} m"
            if 'snr' in det:
                info += f"<br>SNR: {det['snr']:.1f} dB"
            text.append(info)
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=6, color=self.config.detection_color, symbol='x'),
            text=text,
            name='Detections',
            hoverinfo='text'
        ))
    
    def add_tracks(self, fig: go.Figure, tracks: List[Dict]) -> None:
        """Add track trails to the figure."""
        for i, track in enumerate(tracks):
            if not track.get('detections'):
                continue
                
            x, y, z = [], [], []
            for det in track['detections'][-self.config.trail_length:]:
                r = det['range']
                az = np.radians(det['azimuth'])
                el = np.radians(det['elevation'])
                
                x.append(r * np.cos(el) * np.sin(az))
                y.append(r * np.cos(el) * np.cos(az))
                z.append(r * np.sin(el))
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                line=dict(color=self.config.track_color, width=3),
                marker=dict(size=4),
                name=f'Track {track["id"]}',
                showlegend=(i == 0)
            ))
    
    def add_waveform_visualization(self, fig: go.Figure, waveform_data: Dict) -> None:
        """Add waveform visualization as a subplot."""
        if not self.config.show_waveforms or not waveform_data:
            return
            
        # This would be implemented as a separate subplot
        # For now, we'll skip the detailed implementation
        pass
    
    def add_statistics_panel(self, fig: go.Figure, stats: Dict) -> None:
        """Add statistics as annotations."""
        if not self.config.show_statistics or not stats:
            return
            
        stats_text = []
        if 'num_detections' in stats:
            stats_text.append(f"Detections: {stats['num_detections']}")
        if 'num_tracks' in stats:
            stats_text.append(f"Active Tracks: {stats['num_tracks']}")
        if 'scan_time' in stats:
            stats_text.append(f"Scan Time: {stats['scan_time']:.1f} s")
        
        fig.add_annotation(
            text="<br>".join(stats_text),
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
            font=dict(color="white", size=12),
            align="left"
        )
    
    def create_figure_from_state(self, state: Dict) -> go.Figure:
        """Create a complete figure from a state snapshot."""
        fig = self.create_base_figure()
        
        # Add radar
        if 'radar' in state:
            # Create a simple radar object for visualization
            class SimpleRadar:
                def __init__(self, state):
                    self.position = state['position']
                    self.antenna = type('obj', (object,), {
                        'beamwidth_azimuth': state.get('beamwidth_azimuth', 3.0),
                        'beamwidth_elevation': state.get('beamwidth_elevation', 3.0)
                    })()
            
            radar = SimpleRadar(state['radar'])
            self.add_radar(fig, radar)
            
            # Add beam
            if 'antenna_azimuth' in state['radar'] and 'antenna_elevation' in state['radar']:
                self.add_beam(fig, radar, state['radar']['antenna_azimuth'], 
                            state['radar']['antenna_elevation'])
        
        # Add targets
        if 'targets' in state and state['targets']:
            # Convert target dictionaries to objects
            targets = []
            for t in state['targets']:
                target = type('obj', (object,), t)()
                if 'position' in t:
                    target.position = t['position']
                if 'velocity' in t:
                    target.velocity_3d = t['velocity']
                targets.append(target)
            self.add_targets(fig, targets)
        
        # Add detections
        if 'detections' in state:
            self.add_detections(fig, state['detections'])
        
        # Add tracks
        if 'tracks' in state:
            self.add_tracks(fig, state['tracks'])
        
        # Add statistics
        if self.config.show_statistics:
            stats = {
                'num_detections': len(state.get('detections', [])),
                'num_tracks': len(state.get('tracks', [])),
                'scan_time': state.get('time', 0)
            }
            self.add_statistics_panel(fig, stats)
        
        return fig
    
    def visualize(self, radar_system, environment, scan_result=None, 
                  detections=None, tracks=None) -> go.Figure:
        """Create a complete visualization from components."""
        fig = self.create_base_figure()
        
        # Add radar
        self.add_radar(fig, radar_system)
        
        # Add current beam position
        if hasattr(radar_system, 'antenna_azimuth') and hasattr(radar_system, 'antenna_elevation'):
            self.add_beam(fig, radar_system, radar_system.antenna_azimuth, 
                         radar_system.antenna_elevation)
        
        # Add targets
        if environment and hasattr(environment, 'targets') and environment.targets:
            self.add_targets(fig, environment.targets.targets)
        
        # Add detections
        if detections:
            self.add_detections(fig, detections)
        
        # Add tracks
        if tracks:
            self.add_tracks(fig, tracks)
        
        return fig
    
    def show(self, fig: Optional[go.Figure] = None) -> None:
        """Display the figure."""
        if fig is None:
            fig = self.figure
        if fig:
            fig.show()


class DashRadarVisualizer:
    """Dash-based live radar visualization."""
    
    def __init__(self, simulation, config: Optional[VisualizationConfig] = None):
        """Initialize with a simulation instance."""
        self.simulation = simulation
        self.config = config or VisualizationConfig()
        self.visualizer = Radar3DVisualizer(config)
        self.app = None
        
    def create_app(self) -> dash.Dash:
        """Create the Dash application."""
        app = dash.Dash(__name__)
        
        # Store for maintaining state between updates
        app.simulation = self.simulation
        app.visualizer = self.visualizer
        app.last_state = None
        app.last_figure = None
        
        # Layout
        app.layout = html.Div([
            html.H1("Live 3D Radar Simulation", style={'text-align': 'center'}),
            html.Div([
                html.Div([
                    html.H3("Simulation Info"),
                    html.Div(id='info-display', style={'font-family': 'monospace'})
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    dcc.Graph(id='radar-plot', style={'height': '800px'})
                ], style={'width': '70%', 'display': 'inline-block'})
            ]),
            dcc.Interval(
                id='interval-component',
                interval=50,  # Update every 50ms
                n_intervals=0
            )
        ])
        
        # Callback for updates
        @app.callback(
            [Output('radar-plot', 'figure'),
             Output('info-display', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_visualization(n):
            # Get latest state from simulation
            state = None
            if hasattr(app.simulation, 'data_queue'):
                try:
                    import queue
                    while not app.simulation.data_queue.empty():
                        state = app.simulation.data_queue.get_nowait()
                        app.last_state = state
                except queue.Empty:
                    pass
            
            # Use last known state if no new data
            if state is None:
                state = app.last_state
                
            if state is None:
                return go.Figure(), "Waiting for data..."
            
            # Create figure from state
            figure = app.visualizer.create_figure_from_state(state)
            app.last_figure = figure
            
            # Create info text
            info_text = [
                f"Time: {state.get('time', 0):.1f} s",
                f"Targets: {len(state.get('targets', []))}",
                f"Detections: {len(state.get('detections', []))}",
                f"Active Tracks: {len(state.get('tracks', []))}"
            ]
            
            if 'radar' in state:
                info_text.extend([
                    f"Azimuth: {state['radar'].get('antenna_azimuth', 0):.1f}°",
                    f"Elevation: {state['radar'].get('antenna_elevation', 0):.1f}°"
                ])
            
            return figure, html.Pre('\n'.join(info_text))
        
        self.app = app
        return app
    
    def run(self, debug=False, host='127.0.0.1', port=8050):
        """Run the Dash application."""
        if self.app is None:
            self.create_app()
        
        import webbrowser
        webbrowser.open(f'http://{host}:{port}/')
        self.app.run(debug=debug, host=host, port=port)
