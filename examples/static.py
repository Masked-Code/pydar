"""
Static Radar Simulation Example

This example demonstrates:
- Single and multi-frame radar simulations
- Signal processing and analysis
- Detection algorithms
- Performance metrics
- Static 3D visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Import pydar components
from pydar import (
    RadarSystem, Antenna, LinearFMChirp,
    Target, TargetCollection,
    Environment, Atmosphere,
    Radar3DVisualizer, VisualizationConfig
)
from pydar.processing import CFARDetector, RangeDopplerProcessor
import numpy as np
import scipy.constants as const


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


def create_simple_scenario():
    """Create a simple radar scenario for analysis."""
    # Create antenna
    antenna = Antenna(
        beamwidth_azimuth=2.0,
        beamwidth_elevation=2.0,
        gain=35.0,
        sidelobe_level=-25.0
    )
    
    # Create waveform
    waveform = LinearFMChirp(
        duration=10e-6,  # 10 microseconds
        sample_rate=1e9,  # 1 GHz sample rate
        bandwidth=100e6,  # 100 MHz
        center_frequency=10e9  # X-band
    )
    
    # Create radar
    radar = RadarSystem(
        antenna=antenna,
        waveform=waveform,
        position=(0, 0, 0),
        velocity=(0, 0, 0),
        transmit_power=5000,  # 5 kW
        noise_figure=3.0,
        losses=4.0
    )
    
    # Create targets
    targets = TargetCollection()
    
    # Target 1: Approaching aircraft
    targets.add_target(create_target_from_position(
        position=[10000, 5000, 3000],
        velocity=[-200, -100, 0],  # Approaching at ~224 m/s
        rcs=10.0,
        target_id="AIRCRAFT_1"
    ))
    
    # Target 2: Crossing target
    targets.add_target(create_target_from_position(
        position=[-8000, 12000, 5000],
        velocity=[150, 0, 0],  # Crossing at 150 m/s
        rcs=5.0,
        target_id="AIRCRAFT_2"
    ))
    
    # Target 3: Small drone
    targets.add_target(create_target_from_position(
        position=[3000, 2000, 500],
        velocity=[20, 10, 5],
        rcs=0.01,  # Very small RCS
        target_id="DRONE_1"
    ))
    
    # Create environment
    environment = Environment()
    environment.targets = targets
    
    return radar, environment


def perform_single_scan(radar: RadarSystem, environment: Environment, 
                       azimuth: float, elevation: float) -> Dict:
    """Perform a single radar scan and analyze results."""
    print(f"\nScanning at Az: {azimuth:.1f}°, El: {elevation:.1f}°")
    
    # Point antenna
    radar.antenna_azimuth = azimuth
    radar.antenna_elevation = elevation
    
    # Perform scan
    scan_result = radar.scan(environment)
    
    # Process returns
    detections = []
    if scan_result.returns:
        print(f"  Found {len(scan_result.returns)} returns")
        
        for ret in scan_result.returns:
            # Calculate SNR
            snr = calculate_snr(
                ret.power,
                radar.noise_power,
                radar.waveform.bandwidth
            )
            
            # Detection threshold (simple threshold detector)
            if snr > 13.0:  # ~13 dB for Pd=0.9, Pfa=1e-6
                detection = {
                    'range': ret.range,
                    'azimuth': ret.azimuth,
                    'elevation': ret.elevation,
                    'doppler': ret.doppler_shift,
                    'snr': snr,
                    'target_id': ret.target_id
                }
                detections.append(detection)
                print(f"    Detection: {ret.target_id} at {ret.range:.0f}m, SNR: {snr:.1f} dB")
    
    return {
        'azimuth': azimuth,
        'elevation': elevation,
        'detections': detections,
        'scan_result': scan_result
    }


def perform_volume_scan(radar: RadarSystem, environment: Environment) -> List[Dict]:
    """Perform a volume scan over a sector."""
    results = []
    
    # Define scan volume
    az_start, az_end, az_step = -45, 45, 5
    el_start, el_end, el_step = 0, 30, 5
    
    print(f"\nPerforming volume scan:")
    print(f"  Azimuth: {az_start}° to {az_end}° in {az_step}° steps")
    print(f"  Elevation: {el_start}° to {el_end}° in {el_step}° steps")
    
    # Scan pattern
    for el in np.arange(el_start, el_end + el_step, el_step):
        for az in np.arange(az_start, az_end + az_step, az_step):
            result = perform_single_scan(radar, environment, az, el)
            results.append(result)
    
    # Summary
    total_detections = sum(len(r['detections']) for r in results)
    print(f"\nVolume scan complete: {total_detections} total detections")
    
    return results


def analyze_range_doppler(radar: RadarSystem, environment: Environment, 
                         azimuth: float, elevation: float):
    """Perform range-Doppler analysis."""
    print(f"\nRange-Doppler Analysis at Az: {azimuth:.1f}°, El: {elevation:.1f}°")
    
    # Point antenna
    radar.antenna_azimuth = azimuth
    radar.antenna_elevation = elevation
    
    # Get returns
    scan_result = radar.scan(environment)
    
    if not scan_result.returns:
        print("  No returns detected")
        return
    
    # Create range-Doppler processor
    processor = RangeDopplerProcessor(
        waveform=radar.waveform,
        num_pulses=128,
        range_bins=256,
        doppler_bins=256
    )
    
    # Process returns (simplified)
    for ret in scan_result.returns:
        print(f"\n  Target: {ret.target_id}")
        print(f"    Range: {ret.range:.0f} m")
        print(f"    Radial velocity: {ret.doppler_shift * 3e8 / (2 * radar.waveform.center_frequency):.1f} m/s")
        print(f"    Power: {10*np.log10(ret.power):.1f} dBm")


def demonstrate_cfar(radar: RadarSystem, environment: Environment):
    """Demonstrate CFAR detection."""
    print("\nCFAR Detection Demonstration")
    
    # Create CFAR detector (using CA-CFAR)
    from pydar.processing.cfar import CellAveragingCFAR
    cfar = CellAveragingCFAR(
        guard_cells=4,
        training_cells=16,
        pfa=1e-6  # Probability of false alarm
    )
    
    # Perform scan
    scan_result = radar.scan(environment)
    
    if scan_result.returns:
        # Create range profile
        max_range = 50000
        range_bins = 1000
        ranges = np.linspace(0, max_range, range_bins)
        range_profile = np.zeros(range_bins)
        
        # Add returns to range profile
        for ret in scan_result.returns:
            idx = int(ret.range / max_range * range_bins)
            if 0 <= idx < range_bins:
                range_profile[idx] += ret.power
        
        # Add noise
        noise_power = radar.noise_power
        range_profile += np.random.rayleigh(np.sqrt(noise_power), range_bins)
        
        # Apply CFAR
        detections = cfar.detect(range_profile)
        
        print(f"  CFAR detected {len(detections)} targets")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(ranges/1000, 10*np.log10(range_profile), 'b-', label='Range Profile')
        
        for det_idx in detections:
            plt.axvline(ranges[det_idx]/1000, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Range (km)')
        plt.ylabel('Power (dB)')
        plt.title('CFAR Detection Results')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def visualize_scenario(radar: RadarSystem, environment: Environment, 
                      scan_results: List[Dict]):
    """Create 3D visualization of the scenario."""
    # Create visualizer
    config = VisualizationConfig(
        figure_width=1000,
        figure_height=800,
        show_statistics=True,
        trail_length=0  # No trails for static view
    )
    visualizer = Radar3DVisualizer(config)
    
    # Create figure
    fig = visualizer.create_base_figure()
    
    # Add radar
    visualizer.add_radar(fig, radar)
    
    # Add targets
    visualizer.add_targets(fig, environment.targets.targets)
    
    # Add all detections
    all_detections = []
    for result in scan_results:
        all_detections.extend(result['detections'])
    
    if all_detections:
        visualizer.add_detections(fig, all_detections)
    
    # Show
    fig.show()


def calculate_snr(power_received, noise_power, bandwidth):
    """Calculate Signal-to-Noise Ratio."""
    return 10 * np.log10(power_received / (noise_power * bandwidth))


def detection_probability(snr_db, pfa=1e-6):
    """Calculate the detection probability given SNR and probability of false alarm."""
    # Placeholder calculation
    return min(max(snr_db / 20, 0.1), 0.9)


def main():
    """Run static simulation examples."""
    print("=== Static Radar Simulation ===")
    
    # Create scenario
    radar, environment = create_simple_scenario()
    
    # Example 1: Single beam position
    print("\n1. Single Beam Position Analysis")
    result = perform_single_scan(radar, environment, azimuth=20.0, elevation=10.0)
    
    # Example 2: Volume scan
    print("\n2. Volume Scan")
    scan_results = perform_volume_scan(radar, environment)
    
    # Example 3: Range-Doppler analysis
    print("\n3. Range-Doppler Analysis")
    analyze_range_doppler(radar, environment, azimuth=20.0, elevation=10.0)
    
    # Example 4: CFAR detection
    print("\n4. CFAR Detection")
    radar.antenna_azimuth = 20.0
    radar.antenna_elevation = 10.0
    demonstrate_cfar(radar, environment)
    
    # Example 5: 3D Visualization
    print("\n5. 3D Visualization")
    visualize_scenario(radar, environment, scan_results)
    
    # Performance metrics
    print("\n=== Performance Metrics ===")
    for target in environment.targets.targets:
        # Calculate detection probability
        r = np.linalg.norm(target.position)
        snr = radar.calculate_snr(r, target.rcs)
        pd = detection_probability(snr, pfa=1e-6)
        
        print(f"\nTarget: {target.target_id}")
        print(f"  Range: {r:.0f} m")
        print(f"  RCS: {target.rcs:.2f} m²")
        print(f"  SNR: {snr:.1f} dB")
        print(f"  Pd: {pd:.3f}")
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
