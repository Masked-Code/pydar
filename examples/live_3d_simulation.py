"""
Live 3D radar simulation example.

This example demonstrates:
- Real-time radar simulation with moving targets
- Continuous scanning with antenna movement
- Live 3D visualization with multiple views
- Target tracking and display
"""

import numpy as np
import time
from pydar import RadarSystem, Target, Environment, LinearFMChirp
from pydar.live_simulation import LiveRadarSimulation, SimulationConfig
from pydar.visualization_3d import Radar3DVisualizer, VisualizationConfig
import threading


def create_dynamic_scenario():
    """Create a dynamic scenario with moving targets."""
    env = Environment()
    
    # Add various types of targets
    
    # Aircraft - high altitude, fast moving
    aircraft1 = Target(
        range=20000,
        velocity=150,  # ~540 km/h
        rcs=50.0,
        azimuth=30,
        elevation=15,
        acceleration=-2.0  # Slowing down
    )
    env.add_target(aircraft1)
    
    # Helicopter - medium altitude, slow moving
    helicopter = Target(
        range=8000,
        velocity=40,  # ~144 km/h
        rcs=20.0,
        azimuth=-45,
        elevation=10,
        acceleration=0.5
    )
    env.add_target(helicopter)
    
    # Drone swarm - low altitude, various speeds
    for i in range(5):
        drone = Target(
            range=3000 + i * 500,
            velocity=15 + i * 5,
            rcs=0.1,
            azimuth=-60 + i * 30,
            elevation=5,
            acceleration=np.random.uniform(-1, 1)
        )
        env.add_target(drone)
    
    # Ground vehicles
    for i in range(3):
        vehicle = Target(
            range=5000 + i * 1000,
            velocity=20 - i * 5,  # Different speeds
            rcs=5.0,
            azimuth=90 + i * 10,
            elevation=0,  # Ground level
            acceleration=0
        )
        env.add_target(vehicle)
    
    # Add some clutter patches
    env.add_land_clutter('urban', (10000, 60), resolution=100)
    
    return env


def live_visualization_loop(simulation, visualizer):
    """Run visualization updates in a loop."""
    while simulation.is_running:
        try:
            # Update visualization from simulation
            visualizer.update_from_simulation(simulation)
            
            # Small delay to control update rate
            time.sleep(0.1)
        except Exception as e:
            print(f"Visualization error: {e}")
            break


def main():
    """Run live 3D radar simulation."""
    
    print("=== Live 3D Radar Simulation ===")
    print("Setting up radar system...")
    
    # Create high-performance radar system
    radar = RadarSystem(
        frequency=10e9,      # X-band
        power=10000,         # 10 kW peak power
        antenna_gain=35,     # 35 dB antenna gain
        system_loss=3,
        noise_figure=3
    )
    
    # Configure waveform for good range and Doppler resolution
    waveform = LinearFMChirp(
        duration=20e-6,      # 20 microsecond chirp
        sample_rate=200e6,   # 200 MHz sampling
        bandwidth=100e6      # 100 MHz bandwidth
    )
    
    print(f"Range resolution: {radar.range_resolution(waveform.bandwidth):.1f} m")
    
    # Create dynamic scenario
    print("Creating dynamic scenario...")
    environment = create_dynamic_scenario()
    
    # Configure simulation
    sim_config = SimulationConfig(
        update_rate=30.0,        # 30 FPS
        scan_rate=2.0,           # 2 scans per second
        azimuth_start=-90,
        azimuth_end=90,
        azimuth_step=2.0,
        elevation_start=0,
        elevation_end=30,
        elevation_step=5.0,
        max_range=30000,
        enable_doppler=True,
        enable_tracking=True
    )
    
    # Create live simulation
    print("Initializing live simulation...")
    simulation = LiveRadarSimulation(
        radar=radar,
        environment=environment,
        waveform=waveform,
        config=sim_config
    )
    
    # Configure visualization
    vis_config = VisualizationConfig(
        figure_width=1400,
        figure_height=900,
        theme='plotly_dark',
        max_range=30000,
        range_rings=6,
        trail_length=30,
        update_interval=0.1
    )
    
    # Create 3D visualizer
    print("Creating 3D visualization...")
    visualizer = Radar3DVisualizer(config=vis_config)
    
    # Register callbacks for events
    def on_detection(detections):
        print(f"New detections: {len(detections)} targets")
    
    def on_track_update(track):
        if track['state']['confidence'] > 0.8:
            print(f"Track {track['id']}: Range={track['state']['position'][0]:.0f}m, "
                  f"Velocity={track['state']['velocity'][0]:.1f}m/s")
    
    simulation.register_callback('on_detection', on_detection)
    simulation.register_callback('on_track_update', on_track_update)
    
    # Start simulation
    print("Starting simulation...")
    simulation.start()
    
    # Start visualization
    print("Starting 3D visualization...")
    visualizer.start()
    
    # Run visualization updates in separate thread
    vis_thread = threading.Thread(
        target=live_visualization_loop,
        args=(simulation, visualizer)
    )
    vis_thread.start()
    
    # Run for specified duration
    print("\nSimulation running. Press Ctrl+C to stop.")
    print("You should see a 3D visualization window with:")
    print("- Live radar beam scanning")
    print("- Real-time target detections")
    print("- Target tracks with trails")
    print("- Multiple view angles (3D, PPI, Range-Height)")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
            
            # Print periodic status
            state = simulation.get_current_state()
            print(f"\rTime: {state['simulation_time']:.1f}s, "
                  f"Scans: {state['scan_count']}, "
                  f"Tracks: {len(state['tracks'])}, "
                  f"FPS: {state['metrics']['frame_rate']:.1f}", end='')
            
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
    
    # Stop everything
    simulation.stop()
    visualizer.stop()
    vis_thread.join()
    
    print("Simulation complete!")
    
    # Optionally save animation
    save_animation = input("\nSave animation to HTML? (y/n): ")
    if save_animation.lower() == 'y':
        print("Collecting animation frames...")
        frames = []
        for _ in range(100):  # Collect 100 frames
            frames.append(visualizer.create_animation_frame())
            time.sleep(0.1)
        
        print("Saving animation...")
        visualizer.save_animation('radar_simulation.html', frames)
        print("Animation saved to radar_simulation.html")


if __name__ == "__main__":
    main()
