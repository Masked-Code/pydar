"""
Integrated radar simulation with enhanced wave visualization.

This example combines the physics-based radar simulation with
the enhanced 3D visualization to show realistic wave propagation.
"""

import numpy as np
import time
import threading
from pydar import RadarSystem, Target, Environment, LinearFMChirp
from pydar.live_simulation import LiveRadarSimulation, SimulationConfig
from pydar.visualization_3d_enhanced import Enhanced3DRadarVisualizer, EnhancedVisualizationConfig


class IntegratedRadarVisualizer:
    """Integrates live radar simulation with enhanced visualization."""
    
    def __init__(self, radar_system, environment, waveform):
        """Initialize integrated visualizer."""
        self.radar = radar_system
        self.environment = environment
        self.waveform = waveform
        
        # Configure simulation for visualization
        self.sim_config = SimulationConfig(
            update_rate=60.0,  # Match visualization frame rate
            scan_rate=10.0,    # 10 scans per second for visibility
            azimuth_start=-60,
            azimuth_end=60,
            azimuth_step=5.0,
            elevation_start=0,
            elevation_end=30,
            elevation_step=10.0,
            max_range=30000
        )
        
        # Create live simulation
        self.simulation = LiveRadarSimulation(
            self.radar, 
            self.environment, 
            self.waveform,
            self.sim_config
        )
        
        # Configure enhanced visualization
        self.vis_config = EnhancedVisualizationConfig(
            figure_width=1600,
            figure_height=900,
            max_range=30000,
            wave_speed_factor=5000,
            pulse_width=300,
            frame_rate=60.0,
            show_power_decay=True
        )
        
        # Create visualizer
        self.visualizer = Enhanced3DRadarVisualizer(self.vis_config)
        
        # Setup callbacks
        self.simulation.register_callback('on_scan_complete', self._on_scan)
        self.simulation.register_callback('on_detection', self._on_detection)
    
    def _on_scan(self, scan_result):
        """Handle scan completion."""
        # Update antenna position in visualizer
        self.visualizer.set_antenna_position(
            scan_result.azimuth,
            scan_result.elevation
        )
        
        # Emit pulse in visualization
        self.visualizer.emit_pulse(
            scan_result.azimuth,
            scan_result.elevation
        )
    
    def _on_detection(self, detections):
        """Handle new detections."""
        # Detections are already visualized by wave hits
        pass
    
    def _sync_targets(self):
        """Synchronize targets between simulation and visualization."""
        for i, target in enumerate(self.environment.targets):
            # Convert to Cartesian coordinates
            x, y, z = target.to_cartesian()
            
            # Add to visualizer
            self.visualizer.add_target(
                target_id=str(i),
                position=(x, y, z),
                velocity=(
                    target.velocity * np.sin(np.radians(target.azimuth)),
                    target.velocity * np.cos(np.radians(target.azimuth)),
                    0
                ),
                rcs=target.rcs,
                size=100 + 10 * np.log10(target.rcs)  # Size based on RCS
            )
    
    def run(self):
        """Run integrated simulation and visualization."""
        # Initial target sync
        self._sync_targets()
        
        # Start simulation
        print("Starting integrated simulation...")
        self.simulation.start()
        
        # Start visualization
        print("Starting enhanced visualization...")
        self.visualizer.start()
        
        # Update loop
        try:
            while True:
                # Update target positions in visualizer
                dt = 0.1
                self.visualizer.update_target_positions(dt)
                
                # Re-sync targets periodically
                if int(time.time()) % 5 == 0:
                    self._sync_targets()
                
                time.sleep(dt)
                
        except KeyboardInterrupt:
            print("\nStopping...")
            self.simulation.stop()
            self.visualizer.stop()


def create_realistic_scenario():
    """Create a realistic radar scenario."""
    env = Environment()
    
    # Commercial aircraft
    aircraft = Target(
        range=25000,
        velocity=250,  # ~900 km/h
        rcs=100.0,  # Large RCS
        azimuth=30,
        elevation=10,
        acceleration=-5.0  # Descending
    )
    env.add_target(aircraft)
    
    # Military fighter
    fighter = Target(
        range=15000,
        velocity=400,  # ~1440 km/h
        rcs=5.0,  # Small RCS (stealth)
        azimuth=-20,
        elevation=15,
        acceleration=10.0  # Climbing
    )
    env.add_target(fighter)
    
    # Helicopter
    helicopter = Target(
        range=8000,
        velocity=50,  # ~180 km/h
        rcs=20.0,
        azimuth=0,
        elevation=5,
        acceleration=0
    )
    env.add_target(helicopter)
    
    # Drone formation
    for i in range(5):
        drone = Target(
            range=5000 + i * 1000,
            velocity=30,  # ~108 km/h
            rcs=0.1,  # Very small
            azimuth=-40 + i * 20,
            elevation=8,
            acceleration=0
        )
        env.add_target(drone)
    
    # Add ground clutter
    env.add_land_clutter('urban', (15000, 90), resolution=500)
    
    return env


def main():
    """Run integrated radar simulation with wave visualization."""
    
    print("=== Integrated Radar Simulation with Wave Visualization ===")
    print()
    
    # Create advanced radar system
    radar = RadarSystem(
        frequency=10e9,      # X-band
        power=100000,        # 100 kW peak power
        antenna_gain=40,     # 40 dB antenna gain
        system_loss=3,
        noise_figure=2
    )
    
    # High-resolution waveform
    waveform = LinearFMChirp(
        duration=50e-6,      # 50 microsecond chirp
        sample_rate=500e6,   # 500 MHz sampling
        bandwidth=150e6      # 150 MHz bandwidth
    )
    
    print(f"Radar Configuration:")
    print(f"  Frequency: {radar.frequency/1e9:.1f} GHz")
    print(f"  Power: {radar.power/1000:.1f} kW")
    print(f"  Range Resolution: {radar.range_resolution(waveform.bandwidth):.1f} m")
    print()
    
    # Create scenario
    print("Creating realistic scenario...")
    environment = create_realistic_scenario()
    print(f"  Targets: {len(list(environment.targets))}")
    print()
    
    # Create and run integrated visualizer
    integrated = IntegratedRadarVisualizer(radar, environment, waveform)
    
    print("Controls:")
    print("- Rotate: Click and drag")
    print("- Zoom: Scroll wheel")
    print("- Pan: Right click and drag")
    print()
    print("Legend:")
    print("- Blue cone: Radar antenna direction")
    print("- Green dots: Outgoing radar pulses")
    print("- Red dots: Return echoes")
    print("- Yellow rings: Detection events")
    print("- Cyan diamonds: Tracked targets")
    print()
    
    integrated.run()


if __name__ == "__main__":
    main()
