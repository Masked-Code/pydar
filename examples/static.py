"""
Enhanced 3D radar visualization example.

This example shows:
- 3D visualization of radar system and targets
- Multiple target types at different positions
- Radar beam visualization
- Detection results
"""

import numpy as np
from pydar.visualization import Radar3DVisualizer, VisualizationConfig
from pydar import RadarSystem, Target, Environment, LinearFMChirp, Antenna, TargetCollection


def create_target_from_position(position, velocity, rcs, target_id):
    """Create a Target from Cartesian position and velocity."""
    # Calculate spherical coordinates
    x, y, z = position
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z / r)) if r > 0 else 0
    
    # Calculate radial velocity (velocity along line of sight)
    if r > 0:
        los_vec = np.array([x, y, z]) / r
        radial_velocity = np.dot(velocity, los_vec)
    else:
        radial_velocity = 0
    
    target = Target(
        range=r,
        velocity=radial_velocity,
        rcs=rcs,
        azimuth=azimuth,
        elevation=elevation
    )
    target.target_id = target_id
    target.position = np.array(position)  # Store for visualization
    target.velocity_3d = np.array(velocity)  # Store for visualization
    
    return target


def main():
    """Run enhanced 3D visualization."""
    
    print("=== Enhanced 3D Radar Visualization ===")
    print("This visualization shows:")
    print("- 3D view of radar and targets")
    print("- Radar beam visualization")
    print("- Different target types and positions")
    print()
    
    # Configure visualization
    config = VisualizationConfig(
        figure_width=1600,
        figure_height=900,
        max_range=30000,  # 30 km
        show_statistics=True,
        show_rcs=True,
        show_doppler=True
    )
    
    # Create radar system
    print("Creating radar system...")
    antenna = Antenna(
        gain=35,
        beamwidth_azimuth=2.0,
        beamwidth_elevation=2.0,
        sidelobe_level=-25
    )
    
    waveform = LinearFMChirp(
        duration=10e-6,
        sample_rate=1e9,
        bandwidth=100e6,
        center_frequency=10e9
    )
    
    radar = RadarSystem(
        antenna=antenna,
        waveform=waveform,
        position=(0, 0, 100),  # 100m elevation
        velocity=(0, 0, 0),
        transmit_power=5000,
        noise_figure=3.0,
        losses=4.0
    )
    
    # Create environment with targets
    print("Creating environment with targets...")
    environment = Environment()
    targets = TargetCollection()
    
    # Aircraft at high altitude
    targets.add_target(create_target_from_position(
        position=(15000, 10000, 5000),
        velocity=(-150, -50, 0),  # Moving southwest
        rcs=50.0,
        target_id="AC1"
    ))
    
    # Helicopter hovering
    targets.add_target(create_target_from_position(
        position=(5000, -3000, 1000),
        velocity=(20, 10, 0),  # Slow movement
        rcs=20.0,
        target_id="HEL1"
    ))
    
    # Drone swarm
    for i in range(3):
        angle = i * 120  # Spread around
        dist = 8000
        targets.add_target(create_target_from_position(
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
            target_id=f"UAV{i}"
        ))
    
    # Ground vehicle
    targets.add_target(create_target_from_position(
        position=(3000, 2000, 0),
        velocity=(15, -10, 0),
        rcs=10.0,
        target_id="GV1"
    ))
    
    # Stationary tower
    targets.add_target(create_target_from_position(
        position=(-5000, 5000, 200),
        velocity=(0, 0, 0),
        rcs=100.0,
        target_id="TOWER"
    ))
    
    environment.targets = targets
    
    # Perform a scan
    print("\nPerforming radar scan...")
    radar.antenna_azimuth = 0
    radar.antenna_elevation = 10
    scan_result = radar.scan(environment)
    
    print(f"Detected {len(scan_result.returns)} targets")
    
    # Create visualization
    print("\nCreating 3D visualization...")
    visualizer = Radar3DVisualizer(config)
    
    # Create the figure
    fig = visualizer.create_base_figure()
    
    # Add radar
    visualizer.add_radar(fig, radar)
    
    # Add radar beam
    visualizer.add_beam(fig, radar, radar.antenna_azimuth, radar.antenna_elevation)
    
    # Add all targets
    visualizer.add_targets(fig, targets.targets)
    
    # Add detections
    if scan_result.returns:
        detections = []
        for ret in scan_result.returns:
            detections.append({
                'range': ret.range,
                'azimuth': ret.azimuth,
                'elevation': ret.elevation,
                'doppler': ret.doppler_shift,
                'power': ret.power,
                'target_id': ret.target_id
            })
        visualizer.add_detections(fig, detections)
    
    # Add statistics
    if config.show_statistics:
        stats = {
            'num_targets': len(targets.targets),
            'num_detections': len(scan_result.returns),
            'antenna_az': radar.antenna_azimuth,
            'antenna_el': radar.antenna_elevation
        }
        visualizer.add_statistics_panel(fig, stats)
    
    # Show the visualization
    print("\nDisplaying visualization...")
    print("You should see:")
    print("1. Red diamond: Radar position")
    print("2. Yellow cone: Radar beam")
    print("3. Blue markers: Target positions")
    print("4. Green markers: Detected targets")
    print("5. Statistics panel (if enabled)")
    
    fig.show()
    
    print("\nVisualization complete!")
    print("Close the browser tab to exit.")


if __name__ == "__main__":
    main()
