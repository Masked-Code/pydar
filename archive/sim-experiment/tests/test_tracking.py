"""
Tests for target tracking algorithms.
"""

import pytest
import numpy as np

from pydar.processing.tracking import Track, SimpleTracker, KalmanFilter


class TestTrack:
    """Test Track class."""
    
    def test_track_initialization(self):
        """Test track initialization."""
        track = Track()
        
        assert track.track_id is not None
        assert track.state.shape == (4,)
        assert track.covariance.shape == (4, 4)
        assert track.confirmed == False
        assert track.tentative_count == 0
        assert track.missed_count == 0
    
    def test_track_predict(self):
        """Test track prediction."""
        track = Track()
        track.state = np.array([100, 10, 200, -5])  # x=100, vx=10, y=200, vy=-5
        
        # Predict 1 second ahead
        dt = 1.0
        Q = np.eye(4) * 0.1
        track.predict(dt, Q)
        
        # Position should update based on velocity
        assert track.state[0] == pytest.approx(110)  # x + vx*dt
        assert track.state[2] == pytest.approx(195)  # y + vy*dt
    
    def test_track_update(self):
        """Test track update with measurement."""
        track = Track()
        track.state = np.array([100, 10, 200, -5])
        
        # Measurement at slightly different position
        measurement = np.array([105, 198])
        R = np.eye(2) * 1.0
        
        track.update(measurement, R)
        
        # State should move towards measurement
        assert 100 < track.state[0] < 105
        assert 198 < track.state[2] < 200
        assert len(track.measurements) == 1
    
    def test_track_get_methods(self):
        """Test position and velocity getters."""
        track = Track()
        track.state = np.array([100, 10, 200, -5])
        
        pos = track.get_position()
        assert np.array_equal(pos, [100, 200])
        
        vel = track.get_velocity()
        assert np.array_equal(vel, [10, -5])


class TestSimpleTracker:
    """Test SimpleTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = SimpleTracker(
            max_distance=50.0,
            max_missed=3,
            min_hits=2
        )
        
        assert tracker.max_distance == 50.0
        assert tracker.max_missed == 3
        assert tracker.min_hits == 2
        assert len(tracker.tracks) == 0
    
    def test_single_detection(self):
        """Test tracking with single detection."""
        tracker = SimpleTracker()
        
        detections = [np.array([100, 200])]
        tracks = tracker.update(detections, timestamp=0.0)
        
        assert len(tracker.tracks) == 1
        assert len(tracks) == 0  # Not confirmed yet
    
    def test_track_confirmation(self):
        """Test track confirmation after multiple detections."""
        tracker = SimpleTracker(min_hits=3)
        
        # Add detections over time
        for t in range(5):
            detection = np.array([100 + t*10, 200])
            tracker.update([detection], timestamp=t)
        
        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) >= 1
        assert confirmed[0].confirmed == True
    
    def test_track_association(self):
        """Test correct association of detections to tracks."""
        tracker = SimpleTracker(max_distance=30.0)
        
        # First detection
        tracker.update([np.array([100, 200])], timestamp=0.0)
        
        # Second detection close to first
        tracker.update([np.array([110, 205])], timestamp=1.0)
        
        assert len(tracker.tracks) == 1  # Should associate to same track
        assert tracker.tracks[0].tentative_count == 2
    
    def test_multiple_targets(self):
        """Test tracking multiple targets."""
        tracker = SimpleTracker(max_distance=30.0)
        
        # Two well-separated detections
        detections = [
            np.array([100, 200]),
            np.array([300, 400])
        ]
        tracker.update(detections, timestamp=0.0)
        
        assert len(tracker.tracks) == 2
    
    def test_track_deletion(self):
        """Test track deletion after missed detections."""
        tracker = SimpleTracker(max_missed=2)
        
        # Create a track
        tracker.update([np.array([100, 200])], timestamp=0.0)
        assert len(tracker.tracks) == 1
        
        # Miss detections
        for t in range(3):
            tracker.update([], timestamp=t+1)
        
        assert len(tracker.tracks) == 0  # Track should be deleted
    
    def test_no_detections_handling(self):
        """Test handling of no detections."""
        tracker = SimpleTracker()
        
        # Update with empty detection list
        tracks = tracker.update([], timestamp=0.0)
        assert len(tracks) == 0


class TestKalmanFilter:
    """Test KalmanFilter class."""
    
    def test_kalman_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        assert kf.dim_x == 4
        assert kf.dim_z == 2
        assert kf.x.shape == (4,)
        assert kf.P.shape == (4, 4)
        assert kf.F.shape == (4, 4)
        assert kf.H.shape == (2, 4)
    
    def test_kalman_predict(self):
        """Test Kalman filter prediction."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # Simple constant velocity model
        kf.F = np.array([[1, 1], [0, 1]])  # x' = x + v, v' = v
        kf.x = np.array([0, 1])  # Initial position 0, velocity 1
        kf.Q = np.eye(2) * 0.1
        
        kf.predict()
        
        assert kf.x[0] == 1  # Position should increase
        assert kf.x[1] == 1  # Velocity unchanged
    
    def test_kalman_update(self):
        """Test Kalman filter update."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # Measurement of position only
        kf.H = np.array([[1, 0]])
        kf.x = np.array([0, 1])
        kf.R = np.array([[1]])
        
        # Measure position at 0.5
        z = np.array([0.5])
        kf.update(z)
        
        # State should move towards measurement
        assert 0 < kf.x[0] < 0.5
