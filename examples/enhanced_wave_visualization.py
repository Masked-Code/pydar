"""
Enhanced radar wave propagation visualization example.

This example shows:
- Animated radar pulses traveling through space
- Pulses hitting targets and reflecting back
- Visual representation of the radar scanning process
- Real-time detection visualization
"""

import numpy as np
import time
import threading
from pydar.visualization_3d_enhanced import Enhanced3DRadarVisualizer, EnhancedVisualizationConfig


def create_scanning_pattern(visualizer, scan_speed=10.0):
    """Create a continuous scanning pattern for the radar."""
    azimuth = 0.0
    elevation = 10.0
    az_direction = 1
    el_direction = 1
    
    while visualizer.is_running:
        # Update azimuth
        azimuth += az_direction * scan_speed * 0.1
        if azimuth > 45 or azimuth < -45:
            az_direction *= -1
            # Step elevation
            elevation += el_direction * 5
            if elevation > 30 or elevation < 0:
                el_direction *= -1
        
        visualizer.set_antenna_position(azimuth, elevation)
        time.sleep(0.1)


def update_moving_targets(visualizer):
    """Update moving target positions."""
    dt = 0.1  # Update interval
    
    while visualizer.is_running:
        visualizer.update_target_positions(dt)
        time.sleep(dt)


def main():
    """Run enhanced wave propagation visualization."""
    
    print("=== Enhanced 3D Radar Wave Visualization ===")
    print("This visualization shows:")
    print("- Green pulses: Outgoing radar waves")
    print("- Red pulses: Return signals from targets")
    print("- Yellow flashes: Detection events")
    print("- Cyan diamonds: Targets")
    print()
    
    # Configure visualization
    config = EnhancedVisualizationConfig(
        figure_width=1600,
        figure_height=900,
        max_range=30000,  # 30 km
        wave_speed_factor=5000,  # Speed up waves for visibility
        pulse_width=200,  # Visual width of pulses
        frame_rate=60.0,  # 60 FPS for smooth animation
        show_power_decay=True
    )
    
    # Create visualizer
    print("Initializing enhanced visualizer...")
    visualizer = Enhanced3DRadarVisualizer(config)
    
    # Add various targets at different positions
    print("Adding targets...")
    
    # Aircraft at high altitude
    visualizer.add_target(
        target_id="AC1",
        position=(15000, 10000, 5000),
        velocity=(-150, -50, 0),  # Moving southwest
        rcs=50.0,
        size=200
    )
    
    # Helicopter hovering
    visualizer.add_target(
        target_id="HEL1",
        position=(5000, -3000, 1000),
        velocity=(20, 10, 0),  # Slow movement
        rcs=20.0,
        size=150
    )
    
    # Drone swarm
    for i in range(3):
        angle = i * 120  # Spread around
        dist = 8000
        visualizer.add_target(
            target_id=f"UAV{i}",
            position=(
                dist * np.cos(np.radians(angle)),
                dist * np.sin(np.radians(angle)),
                500 + i * 200
            ),
            velocity=(
                -20 * np.sin(np.radians(angle)),
                20 * np.cos(np.radians(angle)),
                0
            ),
            rcs=0.5,
            size=50
        )
    
    # Ground vehicle
    visualizer.add_target(
        target_id="GV1",
        position=(3000, 2000, 0),
        velocity=(15, -10, 0),
        rcs=10.0,
        size=100
    )
    
    # Stationary tower
    visualizer.add_target(
        target_id="TOWER",
        position=(-5000, 5000, 200),
        velocity=(0, 0, 0),
        rcs=100.0,
        size=150
    )
    
    # Start the visualization
    print("Starting enhanced visualization...")
    visualizer.start()
    
    # Start scanning pattern in separate thread
    scan_thread = threading.Thread(
        target=create_scanning_pattern,
        args=(visualizer,)
    )
    scan_thread.daemon = True
    scan_thread.start()
    
    # Start target movement updates
    target_thread = threading.Thread(
        target=update_moving_targets,
        args=(visualizer,)
    )
    target_thread.daemon = True
    target_thread.start()
    
    print("\nVisualization running!")
    print("You should see:")
    print("1. Radar antenna (blue cone) scanning back and forth")
    print("2. Green pulses emanating from the radar")
    print("3. Pulses traveling outward at the speed of light")
    print("4. Yellow flashes when pulses hit targets")
    print("5. Red return pulses traveling back to the radar")
    print("6. Targets moving according to their velocities")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            # Could add status updates here
            
    except KeyboardInterrupt:
        print("\n\nStopping visualization...")
    
    # Stop everything
    visualizer.stop()
    print("Visualization stopped.")


if __name__ == "__main__":
    main()
