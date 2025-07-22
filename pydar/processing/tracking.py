"""
Simple target tracking algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
import uuid


@dataclass
class Track:
    """Represents a single target track."""
    
    track_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))  # [x, vx, y, vy]
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4))
    measurements: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    confirmed: bool = False
    tentative_count: int = 0
    missed_count: int = 0
    
    def predict(self, dt: float, process_noise: np.ndarray):
        """
        Predict next state using constant velocity model.
        
        Args:
            dt: Time step
            process_noise: Process noise covariance matrix
        """
        # State transition matrix
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + process_noise
    
    def update(self, measurement: np.ndarray, measurement_noise: np.ndarray):
        """
        Update track with new measurement using Kalman filter.
        
        Args:
            measurement: New measurement [x, y]
            measurement_noise: Measurement noise covariance
        """
        # Measurement matrix (observe position only)
        H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        # Innovation
        y = measurement - H @ self.state
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + measurement_noise
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.covariance = (I - K @ H) @ self.covariance
        
        # Store measurement
        self.measurements.append(measurement)
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return np.array([self.state[0], self.state[2]])
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return np.array([self.state[1], self.state[3]])


class SimpleTracker:
    """Simple multi-target tracker using nearest neighbor association."""
    
    def __init__(self, 
                 max_distance: float = 100.0,
                 max_missed: int = 3,
                 min_hits: int = 3,
                 process_noise_std: float = 1.0,
                 measurement_noise_std: float = 10.0):
        """
        Initialize tracker.
        
        Args:
            max_distance: Maximum association distance
            max_missed: Maximum missed detections before track deletion
            min_hits: Minimum detections to confirm track
            process_noise_std: Process noise standard deviation
            measurement_noise_std: Measurement noise standard deviation
        """
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_hits = min_hits
        
        # Noise covariances
        self.Q = np.diag([process_noise_std**2] * 4)  # Process noise
        self.R = np.diag([measurement_noise_std**2] * 2)  # Measurement noise
        
        # Active tracks
        self.tracks: List[Track] = []
        self.next_track_id = 0
    
    def update(self, detections: List[np.ndarray], timestamp: float = 0.0) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection positions [x, y]
            timestamp: Current timestamp
            
        Returns:
            List of active tracks
        """
        # Predict existing tracks
        if hasattr(self, 'last_timestamp'):
            dt = timestamp - self.last_timestamp
        else:
            dt = 0.1  # Default time step
        
        for track in self.tracks:
            track.predict(dt, self.Q)
        
        # Associate detections to tracks
        if self.tracks and detections:
            associations = self._associate(detections)
            
            # Update matched tracks
            for track_idx, det_idx in associations:
                self.tracks[track_idx].update(detections[det_idx], self.R)
                self.tracks[track_idx].missed_count = 0
                self.tracks[track_idx].tentative_count += 1
                
                # Confirm track if enough hits
                if self.tracks[track_idx].tentative_count >= self.min_hits:
                    self.tracks[track_idx].confirmed = True
            
            # Create new tracks for unmatched detections
            unmatched_dets = set(range(len(detections))) - set([d for _, d in associations])
            for det_idx in unmatched_dets:
                self._create_track(detections[det_idx])
            
            # Increment missed count for unmatched tracks
            unmatched_tracks = set(range(len(self.tracks))) - set([t for t, _ in associations])
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].missed_count += 1
        
        elif detections:
            # No existing tracks, create new ones
            for detection in detections:
                self._create_track(detection)
        
        else:
            # No detections, increment all missed counts
            for track in self.tracks:
                track.missed_count += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.missed_count < self.max_missed]
        
        # Store timestamp
        self.last_timestamp = timestamp
        
        return self.get_confirmed_tracks()
    
    def _associate(self, detections: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Associate detections to tracks using Hungarian algorithm.
        
        Args:
            detections: List of detections
            
        Returns:
            List of (track_idx, detection_idx) pairs
        """
        # Build cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            track_pos = track.get_position()
            for j, detection in enumerate(detections):
                distance = np.linalg.norm(track_pos - detection)
                cost_matrix[i, j] = distance
        
        # Solve assignment problem
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter by maximum distance
        associations = []
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] < self.max_distance:
                associations.append((track_idx, det_idx))
        
        return associations
    
    def _create_track(self, detection: np.ndarray):
        """Create new track from detection."""
        track = Track()
        
        # Initialize state (position with zero velocity)
        track.state = np.array([detection[0], 0, detection[1], 0])
        
        # Initialize covariance
        track.covariance = np.diag([100, 10, 100, 10])  # Large initial uncertainty
        
        # Store first measurement
        track.measurements.append(detection)
        
        # Initialize tentative count since we have first detection
        track.tentative_count = 1
        
        self.tracks.append(track)
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get list of confirmed tracks."""
        return [t for t in self.tracks if t.confirmed]
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks including tentative."""
        return self.tracks


class KalmanFilter:
    """Standard Kalman filter for target tracking."""
    
    def __init__(self, dim_x: int, dim_z: int):
        """
        Initialize Kalman filter.
        
        Args:
            dim_x: State dimension
            dim_z: Measurement dimension
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State estimate
        self.x = np.zeros(dim_x)
        
        # Error covariance
        self.P = np.eye(dim_x)
        
        # State transition matrix
        self.F = np.eye(dim_x)
        
        # Measurement function
        self.H = np.zeros((dim_z, dim_x))
        
        # Process noise covariance
        self.Q = np.eye(dim_x)
        
        # Measurement noise covariance
        self.R = np.eye(dim_z)
    
    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z: np.ndarray):
        """
        Update with measurement.
        
        Args:
            z: Measurement vector
        """
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
