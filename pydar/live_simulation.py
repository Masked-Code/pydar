"""
Live radar simulation controller.

This module provides real-time radar simulation with continuous scanning
and live updates for 3D visualization.
"""

import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
import numpy as np

from .radar import RadarSystem
from .environment import Environment
from .waveforms import Waveform
from .scan_result import ScanResult


@dataclass
class SimulationConfig:
    """Configuration for live simulation."""
    
    update_rate: float = 30.0  # Hz (frames per second)
    scan_rate: float = 1.0  # Hz (scans per second)
    azimuth_start: float = -180.0  # degrees
    azimuth_end: float = 180.0  # degrees
    azimuth_step: float = 1.0  # degrees
    elevation_start: float = -10.0  # degrees
    elevation_end: float = 30.0  # degrees
    elevation_step: float = 5.0  # degrees
    max_range: float = 50000.0  # meters
    buffer_size: int = 100  # Number of scans to keep in memory
    enable_doppler: bool = True
    enable_tracking: bool = True


class LiveRadarSimulation:
    """Live radar simulation with continuous scanning."""
    
    def __init__(self, radar: RadarSystem, environment: Environment, 
                 waveform: Waveform, config: Optional[SimulationConfig] = None):
        """
        Initialize live simulation.
        
        Args:
            radar: Radar system to simulate
            environment: Environment to scan
            waveform: Waveform to use
            config: Simulation configuration
        """
        self.radar = radar
        self.environment = environment
        self.waveform = waveform
        self.config = config or SimulationConfig()
        
        # Simulation state
        self.is_running = False
        self.simulation_time = 0.0
        self.current_azimuth = self.config.azimuth_start
        self.current_elevation = self.config.elevation_start
        self.scan_count = 0
        
        # Threading
        self._simulation_thread = None
        self._stop_event = threading.Event()
        
        # Data queues
        self.scan_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.detection_queue = queue.Queue(maxsize=self.config.buffer_size)
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'on_scan_complete': [],
            'on_detection': [],
            'on_track_update': [],
            'on_simulation_step': []
        }
        
        # Performance metrics
        self.metrics = {
            'scan_time': 0.0,
            'processing_time': 0.0,
            'frame_rate': 0.0,
            'cpu_usage': 0.0
        }
        
        # Track management
        self.tracks = {}
        self.next_track_id = 1
    
    def start(self):
        """Start the live simulation."""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        self._simulation_thread = threading.Thread(target=self._simulation_loop)
        self._simulation_thread.start()
    
    def stop(self):
        """Stop the live simulation."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        if self._simulation_thread:
            self._simulation_thread.join()
    
    def _simulation_loop(self):
        """Main simulation loop running in separate thread."""
        last_update = time.time()
        update_interval = 1.0 / self.config.update_rate
        scan_interval = 1.0 / self.config.scan_rate
        last_scan = time.time()
        
        while not self._stop_event.is_set():
            current_time = time.time()
            dt = current_time - last_update
            
            if dt >= update_interval:
                self.simulation_time += dt
                
                self.environment.targets.update_all(self.simulation_time)
                
                if current_time - last_scan >= scan_interval:
                    self._perform_scan()
                    last_scan = current_time
                
                if self.config.enable_tracking:
                    self._update_tracks()
                
                self._update_metrics(dt)
                
                self._trigger_callbacks('on_simulation_step', self.simulation_time)
                
                last_update = current_time
            else:
                time.sleep(0.001)
    
    def _perform_scan(self):
        """Perform a radar scan at current antenna position."""
        start_time = time.time()
        
        self.radar.antenna_azimuth = self.current_azimuth
        self.radar.antenna_elevation = self.current_elevation
        
        # Perform scan
        scan_result = self.radar.scan(self.environment, self.waveform)
        scan_result.azimuth = self.current_azimuth
        scan_result.elevation = self.current_elevation
        scan_result.scan_time = self.simulation_time
        
        detections = self._process_detections(scan_result)
        
        self._update_antenna_position()
        
        try:
            self.scan_queue.put_nowait(scan_result)
            if detections:
                self.detection_queue.put_nowait(detections)
        except queue.Full:
            self.scan_queue.get()
            self.scan_queue.put(scan_result)
            if detections:
                self.detection_queue.get()
                self.detection_queue.put(detections)
        
        self.metrics['scan_time'] = time.time() - start_time
        self.scan_count += 1
        
        self._trigger_callbacks('on_scan_complete', scan_result)
        if detections:
            self._trigger_callbacks('on_detection', detections)
    
    def _process_detections(self, scan_result: ScanResult) -> List[Dict[str, Any]]:
        """Process scan result to extract detections."""
        detections = []
        
        ranges, amplitude = scan_result.range_profile()
        noise_floor = np.median(amplitude)
        threshold = noise_floor * 10  
        
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(amplitude, height=threshold, distance=10)
        
        for peak_idx in peaks:
            detection = {
                'range': ranges[peak_idx],
                'azimuth': self.current_azimuth,
                'elevation': self.current_elevation,
                'amplitude': amplitude[peak_idx],
                'snr': 20 * np.log10(amplitude[peak_idx] / noise_floor),
                'time': self.simulation_time
            }
            
            if self.config.enable_doppler:
                detection['doppler'] = 0.0 
                detection['velocity'] = 0.0 
            
            detections.append(detection)
        
        return detections
    
    def _update_antenna_position(self):
        """Update antenna pointing angles for next scan."""
        # Simple raster scan pattern
        self.current_azimuth += self.config.azimuth_step
        
        if self.current_azimuth > self.config.azimuth_end:
            self.current_azimuth = self.config.azimuth_start
            self.current_elevation += self.config.elevation_step
            
            if self.current_elevation > self.config.elevation_end:
                self.current_elevation = self.config.elevation_start
    
    def _update_tracks(self):
        """Update target tracks based on recent detections."""
        recent_detections = []
        try:
            while not self.detection_queue.empty():
                recent_detections.extend(self.detection_queue.get_nowait())
        except queue.Empty:
            pass
        
        # Associate detections with existing tracks
        for detection in recent_detections:
            matched_track = self._find_nearest_track(detection)
            
            if matched_track:
                track_id = matched_track['id']
                self.tracks[track_id]['detections'].append(detection)
                self.tracks[track_id]['last_update'] = self.simulation_time
                self._update_track_state(track_id)
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    'id': track_id,
                    'detections': [detection],
                    'first_detection': self.simulation_time,
                    'last_update': self.simulation_time,
                    'state': self._initialize_track_state(detection)
                }
            
            self._trigger_callbacks('on_track_update', self.tracks[track_id])
        
        stale_time = 5.0 
        current_tracks = list(self.tracks.keys())
        for track_id in current_tracks:
            if self.simulation_time - self.tracks[track_id]['last_update'] > stale_time:
                del self.tracks[track_id]
    
    def _find_nearest_track(self, detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find nearest existing track to a detection."""
        min_distance = float('inf')
        nearest_track = None
        
        # Simple distance metric in range-azimuth-elevation space
        for track in self.tracks.values():
            if not track['detections']:
                continue
            
            last_det = track['detections'][-1]
            distance = np.sqrt(
                (detection['range'] - last_det['range'])**2 +
                (detection['azimuth'] - last_det['azimuth'])**2 * 100 + 
                (detection['elevation'] - last_det['elevation'])**2 * 100
            )
            
            if distance < 500 and distance < min_distance: 
                min_distance = distance
                nearest_track = track
        
        return nearest_track
    
    def _initialize_track_state(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize track state from first detection."""
        return {
            'position': [detection['range'], detection['azimuth'], detection['elevation']],
            'velocity': [0.0, 0.0, 0.0],
            'confidence': 0.5
        }
    
    def _update_track_state(self, track_id: int):
        """Update track state based on detections."""
        track = self.tracks[track_id]
        detections = track['detections']
        
        if len(detections) < 2:
            return
        
        recent_dets = detections[-2:]
        dt = recent_dets[1]['time'] - recent_dets[0]['time']
        
        if dt > 0:
            state = track['state']
            state['velocity'][0] = (recent_dets[1]['range'] - recent_dets[0]['range']) / dt
            state['confidence'] = min(1.0, state['confidence'] + 0.1)
    
    def _update_metrics(self, dt: float):
        """Update performance metrics."""
        self.metrics['frame_rate'] = 1.0 / dt if dt > 0 else 0.0
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for simulation events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger all callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Error in callback for {event}: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            'simulation_time': self.simulation_time,
            'antenna_position': {
                'azimuth': self.current_azimuth,
                'elevation': self.current_elevation
            },
            'scan_count': self.scan_count,
            'tracks': dict(self.tracks),
            'metrics': dict(self.metrics),
            'is_running': self.is_running
        }
    
    def get_recent_scans(self, n: int = 10) -> List[ScanResult]:
        """Get n most recent scan results."""
        scans = []
        temp_queue = queue.Queue()
        
        try:
            while not self.scan_queue.empty() and len(scans) < n:
                scan = self.scan_queue.get_nowait()
                scans.append(scan)
                temp_queue.put(scan)
        except queue.Empty:
            pass
        
        while not temp_queue.empty():
            self.scan_queue.put(temp_queue.get())
        
        return scans
