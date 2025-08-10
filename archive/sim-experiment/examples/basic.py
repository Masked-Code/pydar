"""
Basic radar simulation example.

This example demonstrates:
- Creating a radar system
- Defining targets
- Running a simulation
- Visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt

from pydar import RadarSystem, Target, Environment, LinearFMChirp, Antenna


def main():
    """Run basic radar simulation."""
    
    # Create radar system
    print("Creating radar system...")
    antenna = Antenna(gain=30, beamwidth_azimuth=2.0, beamwidth_elevation=2.0)  # Define an example antenna
    waveform = LinearFMChirp(duration=10e-6, sample_rate=100e6, bandwidth=50e6, center_frequency=10e9)  # Define an example waveform
    radar = RadarSystem(
        antenna=antenna,      # Use defined antenna
        waveform=waveform,    # Use defined waveform
        position=(0, 0, 0),   # Default position
        velocity=(0, 0, 0),   # Default velocity
        transmit_power=1000,  # 1 kW transmit power
        noise_figure=3,       # 3 dB noise figure
        losses=3              # 3 dB system losses
    )
    
    # Print radar parameters
    print(f"Radar frequency: {radar.frequency/1e9:.1f} GHz")
    print(f"Wavelength: {radar.wavelength*100:.1f} cm")
    print(f"Transmit power: {radar.transmit_power} W")
    
    print(f"Waveform duration: {waveform.duration*1e6:.1f} μs")
    print(f"Bandwidth: {waveform.bandwidth/1e6:.1f} MHz")
    print(f"Range resolution: {radar.range_resolution(waveform.bandwidth):.1f} m")
    
    # Create environment with targets
    print("\nSetting up environment...")
    env = Environment()
    
    # Add some targets
    targets = [
        Target(range=2000, velocity=50, rcs=1.0, azimuth=10),    # Small drone
        Target(range=5000, velocity=-30, rcs=10.0, azimuth=-5),  # Car
        Target(range=10000, velocity=100, rcs=100.0, azimuth=0), # Aircraft
    ]
    
    for i, target in enumerate(targets):
        env.add_target(target)
        print(f"Target {i+1}: Range={target.range}m, Velocity={target.velocity}m/s, RCS={target.rcs}m²")
    
    # Run simulation - scan across azimuth angles
    print("\nRunning radar scan...")
    all_detections = []
    
    # Scan from -15 to +15 degrees in steps
    scan_azimuths = np.arange(-15, 16, 1)  # 1 degree steps
    
    for az in scan_azimuths:
        radar.antenna_azimuth = az
        radar.antenna_elevation = 0
        result = radar.scan(env)
        
        if result.returns:
            print(f"  Az {az:+3.0f}°: {len(result.returns)} detections")
            all_detections.extend(result.returns)
    
    # Print all detections
    print(f"\nTotal detections: {len(all_detections)}")
    for i, detection in enumerate(all_detections):
        print(f"\nDetection {i+1}:")
        print(f"  Range: {detection.range:.1f} m")
        print(f"  Azimuth: {detection.azimuth:.1f}°")
        print(f"  Elevation: {detection.elevation:.1f}°")
        print(f"  Doppler: {detection.doppler_shift:.1f} Hz")
        print(f"  Power: {10*np.log10(detection.power):.1f} dBm")
    
    # Simple visualization of detection results
    if all_detections:
        print("\nGenerating detection plot...")
        
        plt.figure(figsize=(10, 8))
        
        # Plot 1: Range vs Azimuth
        plt.subplot(2, 1, 1)
        ranges = [det.range for det in all_detections]
        azimuths = [det.azimuth for det in all_detections]
        powers_db = [10*np.log10(det.power) for det in all_detections]
        
        scatter = plt.scatter(azimuths, ranges, c=powers_db, s=100, cmap='hot')
        plt.colorbar(scatter, label='Power (dBm)')
        plt.xlabel('Azimuth (degrees)')
        plt.ylabel('Range (m)')
        plt.title('Detected Targets')
        plt.grid(True)
        
        # Plot 2: Range vs Doppler
        plt.subplot(2, 1, 2)
        dopplers = [det.doppler_shift for det in all_detections]
        velocities = [dop * radar.wavelength / 2 for dop in dopplers]  # Convert to velocity
        
        scatter2 = plt.scatter(velocities, ranges, c=powers_db, s=100, cmap='hot')
        plt.colorbar(scatter2, label='Power (dBm)')
        plt.xlabel('Radial Velocity (m/s)')
        plt.ylabel('Range (m)')
        plt.title('Range-Doppler Map')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('basic_simulation_detections.png', dpi=150)
        plt.show()
        
        # Calculate and display SNR for each unique target
        print("\nDetection Analysis:")
        unique_targets = {}
        for detection in all_detections:
            if detection.target_id not in unique_targets:
                unique_targets[detection.target_id] = detection
        
        for i, (tid, detection) in enumerate(unique_targets.items()):
            # Find matching target by range
            matching_target = None
            for t in targets:
                if abs(t.range - detection.range) < 10:  # Within 10m
                    matching_target = t
                    break
            
            if matching_target:
                snr = radar.snr(detection.range, matching_target.rcs)
                
                # Probability of detection (simplified)
                if snr > 13:  # ~13 dB for Pd=0.9, Pfa=1e-6
                    pd = 0.9
                elif snr > 10:
                    pd = 0.5
                else:
                    pd = 0.1
                    
                print(f"\nTarget {i+1} (ID: {detection.target_id}):")
                print(f"  SNR: {snr:.1f} dB")
                print(f"  Estimated Pd: {pd:.1f}")
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
