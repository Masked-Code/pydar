"""
Live 3D Radar Simulation with Real-time Visualization.

This example demonstrates a live radar simulation where:
- Targets move along randomly generated linear paths
- The radar scans and detects targets in real-time
- The visualization updates continuously to show current state
- Everything happens dynamically (no pre-recorded frames)
"""

import numpy as np
import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import webbrowser

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydar import (
    RadarSystem, Antenna, Environment, Target, TargetCollection
)
from pydar.waveforms import LinearFMChirp
from pydar.visualization import (
    DashRadarVisualizer, VisualizationConfig, Radar3DVisualizer
)


@dataclass
class LinearTarget:
    """Target moving along a linear path."""
    id: int
    start_position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    rcs: float  # Radar cross section in m²
    
    def get_position(self, time: float) -> np.ndarray:
        """Get target position at given time."""
        return self.start_position + self.velocity * time


class LiveRadarSimulation:
    """Live radar simulation with real-time target tracking."""
    
    def __init__(self):
        # Initialize waveform first (needed by radar)
        self.waveform = self._create_waveform()
        # Initialize radar
        self.radar = self._create_radar()
        self.environment = Environment()
        
        # Simulation parameters
        self.simulation_time = 0.0
        self.update_rate = 20.0  # Hz - reasonable update rate
        self.scan_rate = 5.0  # Hz - faster but not too fast
        
        # Target management
        self.targets: List[LinearTarget] = []
        self.target_collection = TargetCollection()
        
        # Scanning parameters
        self.azimuth_limits = (-60, 60)  # degrees
        self.elevation_limits = (0, 30)  # degrees
        self.azimuth_step = 5.0  # degrees - balanced step size
        self.elevation_step = 5.0  # degrees - balanced step size
        self.current_azimuth = self.azimuth_limits[0]
        self.current_elevation = self.elevation_limits[0]
        
        # Detection storage
        self.detections: List[Dict] = []
        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 1
        
        # Threading
        self.is_running = False
        self.simulation_thread = None
        self.data_queue = queue.Queue(maxsize=10)
        
        # Visualization config
        self.viz_config = VisualizationConfig(
            figure_width=1200,
            figure_height=800,
            detection_color='lime',
            track_color='cyan',
            beam_color='yellow',
            max_range=30000.0  # 30 km
        )
        # Create base visualizer for figure creation
        self.visualizer = Radar3DVisualizer(self.viz_config)
    
    def _create_radar(self) -> RadarSystem:
        """Create radar system."""
        antenna = Antenna(
            gain=40.0,
            beamwidth_azimuth=2.0,
            beamwidth_elevation=2.0,
            sidelobe_level=-25.0
        )
        
        return RadarSystem(
            antenna=antenna,
            waveform=self.waveform,
            position=(0, 0, 10),  # 10m height
            transmit_power=1e6,  # 1 MW
            noise_figure=3.0,
            losses=2.0
        )
    
    def _create_waveform(self) -> LinearFMChirp:
        """Create radar waveform."""
        return LinearFMChirp(
            duration=10e-6,  # 10 microsecond pulse
            sample_rate=10e6,  # 10 MHz sample rate
            bandwidth=5e6,  # 5 MHz bandwidth
            center_frequency=10e9  # 10 GHz (X-band)
        )
    
    def generate_random_targets(self, num_targets: int = 5):
        """Generate random targets with linear paths."""
        self.targets.clear()
        self.target_collection = TargetCollection()
        
        for i in range(num_targets):
            # Random starting position
            range_start = np.random.uniform(5000, 20000)  # 5-20 km
            azimuth_start = np.random.uniform(-45, 45)  # degrees
            elevation_start = np.random.uniform(0, 20)  # degrees
            
            # Convert to Cartesian
            az_rad = np.radians(azimuth_start)
            el_rad = np.radians(elevation_start)
            x = range_start * np.cos(el_rad) * np.sin(az_rad)
            y = range_start * np.cos(el_rad) * np.cos(az_rad)
            z = range_start * np.sin(el_rad) + np.random.uniform(100, 5000)
            
            # Random velocity (50-200 m/s)
            speed = np.random.uniform(50, 200)
            heading = np.random.uniform(0, 2*np.pi)
            climb_angle = np.random.uniform(-0.1, 0.1)  # Small climb/descent
            
            vx = speed * np.cos(climb_angle) * np.sin(heading)
            vy = speed * np.cos(climb_angle) * np.cos(heading)
            vz = speed * np.sin(climb_angle)
            
            # Random RCS (1-50 m²)
            rcs = np.random.uniform(1, 50)
            
            target = LinearTarget(
                id=i,
                start_position=np.array([x, y, z]),
                velocity=np.array([vx, vy, vz]),
                rcs=rcs
            )
            self.targets.append(target)
            
            # Add to target collection for radar simulation
            # Convert to spherical coordinates for Target class
            x, y, z = target.start_position
            range_to_target = np.sqrt(x**2 + y**2 + z**2)
            azimuth = np.degrees(np.arctan2(x, y))
            elevation = np.degrees(np.arcsin(z / range_to_target))
            
            # Calculate radial velocity
            radial_velocity = np.dot(target.velocity, target.start_position) / range_to_target
            
            radar_target = Target(
                range=range_to_target,
                velocity=radial_velocity,
                rcs=rcs,
                azimuth=azimuth,
                elevation=elevation
            )
            # Store the full 3D position and velocity for updates
            radar_target.position_3d = target.start_position
            radar_target.velocity_3d = target.velocity
            self.target_collection.add_target(radar_target)
        
        print(f"Generated {num_targets} random targets")
    
    def update_target_positions(self):
        """Update all target positions based on current time."""
        for i, linear_target in enumerate(self.targets):
            new_position = linear_target.get_position(self.simulation_time)
            
            # Update in target collection
            if i < len(self.target_collection.targets):
                radar_target = self.target_collection.targets[i]
                
                # Convert new position to spherical coordinates
                x, y, z = new_position
                range_to_target = np.sqrt(x**2 + y**2 + z**2)
                azimuth = np.degrees(np.arctan2(x, y))
                elevation = np.degrees(np.arcsin(z / range_to_target))
                
                # Update radar target properties
                radar_target.range = range_to_target
                radar_target.azimuth = azimuth
                radar_target.elevation = elevation
                
                # Calculate new radial velocity
                radar_target.velocity = np.dot(linear_target.velocity, new_position) / range_to_target
                
                # Update stored 3D position
                radar_target.position_3d = new_position
    
    def perform_scan(self) -> Dict:
        """Perform a radar scan and process detections."""
        # Set antenna pointing
        self.radar.antenna_azimuth = self.current_azimuth
        self.radar.antenna_elevation = self.current_elevation
        
        # Update environment with current targets
        self.environment.targets = self.target_collection
        
        # Perform scan
        scan_result = self.radar.scan(self.environment)
        
        # Process detections
        detections = self.process_detections(scan_result)
        
        # Update tracks
        if detections:
            self.update_tracks(detections)
        
        # Create state snapshot
        state = self.create_state_snapshot()
        
        # Update antenna position for next scan
        self.update_antenna_position()
        
        return state
    
    def process_detections(self, scan_result) -> List[Dict]:
        """Process scan result to extract detections."""
        detections = []
        
        # SimpleScanResult contains a list of DetectionReturn objects
        for detection_return in scan_result.returns:
            # Calculate SNR from power (simplified)
            snr_db = 10 * np.log10(detection_return.power / self.radar.noise_power) if detection_return.power > 0 else -20
            
            detection = {
                'range': detection_return.range,
                'azimuth': detection_return.azimuth,
                'elevation': detection_return.elevation,
                'doppler': detection_return.doppler_shift,
                'power': detection_return.power,
                'snr': snr_db,
                'time': self.simulation_time,
                'target_id': detection_return.target_id
            }
            detections.append(detection)
            self.detections.append(detection)  # Store in history
        
        # Keep only recent detections (last 5 seconds)
        cutoff_time = self.simulation_time - 5.0
        self.detections = [d for d in self.detections if d['time'] > cutoff_time]
        
        return detections
    
    def update_tracks(self, new_detections: List[Dict]):
        """Update target tracks based on detections."""
        # Simple nearest-neighbor tracking
        for detection in new_detections:
            # Find nearest track
            min_distance = float('inf')
            nearest_track_id = None
            
            for track_id, track in self.tracks.items():
                if track['detections']:
                    last_det = track['detections'][-1]
                    # Simple distance metric
                    distance = np.sqrt(
                        (detection['range'] - last_det['range'])**2 +
                        (detection['azimuth'] - last_det['azimuth'])**2 * 100 +
                        (detection['elevation'] - last_det['elevation'])**2 * 100
                    )
                    
                    time_diff = detection['time'] - last_det['time']
                    if time_diff < 5.0 and distance < 1000 and distance < min_distance:
                        min_distance = distance
                        nearest_track_id = track_id
            
            if nearest_track_id is not None:
                # Update existing track
                self.tracks[nearest_track_id]['detections'].append(detection)
                self.tracks[nearest_track_id]['last_update'] = self.simulation_time
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    'id': track_id,
                    'detections': [detection],
                    'first_detection': self.simulation_time,
                    'last_update': self.simulation_time
                }
        
        # Remove stale tracks
        stale_time = 5.0
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if self.simulation_time - track['last_update'] > stale_time:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def update_antenna_position(self):
        """Update antenna pointing for next scan."""
        self.current_azimuth += self.azimuth_step
        
        if self.current_azimuth > self.azimuth_limits[1]:
            self.current_azimuth = self.azimuth_limits[0]
            self.current_elevation += self.elevation_step
            
            if self.current_elevation > self.elevation_limits[1]:
                self.current_elevation = self.elevation_limits[0]
    
    def create_state_snapshot(self) -> Dict:
        """Create current state snapshot for visualization."""
        # Radar state
        radar_state = {
            'position': self.radar.position.tolist(),
            'antenna_azimuth': self.current_azimuth,
            'antenna_elevation': self.current_elevation,
            'frequency': self.radar.frequency,
            'beamwidth_azimuth': self.radar.antenna.beamwidth_azimuth,
            'beamwidth_elevation': self.radar.antenna.beamwidth_elevation
        }
        
        # Recent detections (last 2 seconds for visibility)
        recent_time = self.simulation_time - 2.0
        recent_detections = [d for d in self.detections if d['time'] > recent_time]
        
        # Active tracks with trails
        active_tracks = []
        for track in self.tracks.values():
            if track['detections']:
                # Get trail of recent detections
                trail_detections = [d for d in track['detections'] 
                                  if self.simulation_time - d['time'] < 10.0]
                if trail_detections:
                    active_tracks.append({
                        'id': track['id'],
                        'detections': trail_detections
                    })
        
        # Current target positions
        target_states = []
        for target in self.targets:
            pos = target.get_position(self.simulation_time)
            target_states.append({
                'id': target.id,
                'position': pos.tolist(),
                'velocity': target.velocity.tolist(),
                'rcs': target.rcs
            })
        
        return {
            'time': self.simulation_time,
            'radar': radar_state,
            'detections': recent_detections,
            'tracks': active_tracks,
            'targets': target_states
        }
    
    def simulation_loop(self):
        """Main simulation loop running in separate thread."""
        last_update = time.time()
        last_scan = time.time()
        last_visualization_update = time.time()
        update_interval = 1.0 / self.update_rate
        scan_interval = 1.0 / self.scan_rate
        visualization_interval = 0.05  # Send visualization updates every 50ms
        
        while self.is_running:
            current_time = time.time()
            dt = current_time - last_update
            
            if dt >= update_interval:
                # Update simulation time
                self.simulation_time += dt
                
                # Update target positions
                self.update_target_positions()
                
                # Perform scan if needed
                if current_time - last_scan >= scan_interval:
                    self.perform_scan()
                    last_scan = current_time
                
                # Send state updates for visualization more frequently
                if current_time - last_visualization_update >= visualization_interval:
                    state = self.create_state_snapshot()
                    
                    # Send state to visualization
                    try:
                        # Remove old data if queue is full
                        if self.data_queue.full():
                            self.data_queue.get_nowait()
                        self.data_queue.put_nowait(state)
                    except queue.Empty:
                        pass
                    
                    last_visualization_update = current_time
                
                last_update = current_time
            else:
                time.sleep(0.001)
    
    def start(self):
        """Start the simulation."""
        if self.is_running:
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        print("Simulation started")
    
    def stop(self):
        """Stop the simulation."""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        print("Simulation stopped")


def create_dash_app(simulation: LiveRadarSimulation):
    """Create Dash app for live visualization."""
    app = dash.Dash(__name__)
    
    # Store for maintaining state between updates
    app.last_state = None
    app.last_figure = None
    
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
            interval=50,  # Update every 50ms for smooth performance
            n_intervals=0
        )
    ])
    
    @app.callback(
        [Output('radar-plot', 'figure'),
         Output('info-display', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_visualization(n):
        # Get latest state from queue
        state = None
        try:
            # Get all available states and use the latest
            while not simulation.data_queue.empty():
                state = simulation.data_queue.get_nowait()
                app.last_state = state  # Store the latest state
        except queue.Empty:
            pass
        
        # Use the last known state if no new data
        if state is None:
            state = app.last_state
            
        if state is None:
            # Only return empty on first call when no data exists yet
            return go.Figure(), "Waiting for data..."
        
        # Create figure from state
        figure = simulation.visualizer.create_figure_from_state(state)
        app.last_figure = figure
        
        # Create info text
        info_text = [
            f"Time: {state['time']:.1f} s",
            f"Azimuth: {state['radar']['antenna_azimuth']:.1f}°",
            f"Elevation: {state['radar']['antenna_elevation']:.1f}°",
            f"Detections: {len(state['detections'])}",
            f"Active Tracks: {len(state['tracks'])}",
            f"Targets: {len(state['targets'])}"
        ]
        
        return figure, html.Pre('\n'.join(info_text))
    
    return app


def main():
    """Run live radar simulation with visualization."""
    print("=== Live 3D Radar Simulation ===")
    
    # Create simulation
    simulation = LiveRadarSimulation()
    
    # Generate random targets
    num_targets = 7
    simulation.generate_random_targets(num_targets)
    
    # Start simulation
    simulation.start()
    
    # Create and run Dash app
    app = create_dash_app(simulation)
    
    # Open browser
    webbrowser.open('http://127.0.0.1:8050/')
    
    try:
        # Run app
        app.run(debug=False, host='127.0.0.1', port=8050)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        simulation.stop()


if __name__ == "__main__":
    main()
