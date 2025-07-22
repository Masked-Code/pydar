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

from pydar import RadarSystem, Target, Environment, LinearFMChirp


def main():
    """Run basic radar simulation."""
    
    # Create radar system
    print("Creating radar system...")
    radar = RadarSystem(
        frequency=10e9,      # 10 GHz (X-band)
        power=1000,          # 1 kW transmit power
        antenna_gain=30,     # 30 dB antenna gain
        system_loss=3,       # 3 dB system losses
        noise_figure=3       # 3 dB noise figure
    )
    
    # Print radar parameters
    print(f"Radar frequency: {radar.frequency/1e9:.1f} GHz")
    print(f"Wavelength: {radar.wavelength*100:.1f} cm")
    print(f"Transmit power: {radar.power} W")
    
    # Create waveform
    print("\nCreating waveform...")
    waveform = LinearFMChirp(
        duration=10e-6,      # 10 microsecond chirp
        sample_rate=100e6,   # 100 MHz sampling rate
        bandwidth=50e6       # 50 MHz bandwidth
    )
    
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
    
    # Run simulation
    print("\nRunning radar scan...")
    result = radar.scan(env, waveform)
    
    # Print results summary
    print("\n" + result.summary())
    
    # Visualization
    print("\nGenerating plots...")
    
    # Plot transmitted and received signals
    plt.figure(figsize=(12, 8))
    
    # Time axis in microseconds
    t = np.arange(len(result.tx_signal)) / result.sample_rate * 1e6
    
    # Transmitted signal
    plt.subplot(3, 1, 1)
    plt.plot(t, np.real(result.tx_signal))
    plt.title('Transmitted Signal (Real Part)')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Received signal
    plt.subplot(3, 1, 2)
    plt.plot(t, np.real(result.rx_signal))
    plt.title('Received Signal (Real Part)')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Matched filter output
    mf_output = result.matched_filter()
    ranges, amplitude = result.range_profile()
    
    plt.subplot(3, 1, 3)
    plt.plot(ranges/1000, 20*np.log10(amplitude + 1e-10))
    plt.title('Range Profile (Matched Filter Output)')
    plt.xlabel('Range (km)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)
    
    # Mark target locations
    for info in result.target_info:
        target = info['target']
        plt.axvline(x=target.range/1000, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('basic_simulation_signals.png', dpi=150)
    plt.show()
    
    # Plot range profile separately
    result.plot_range_profile(max_range=15000, db_scale=True)
    
    # Calculate and display detection metrics
    print("\nDetection Analysis:")
    for i, info in enumerate(result.target_info):
        target = info['target']
        snr = info['snr']
        
        # Probability of detection (simplified)
        if snr > 13:  # ~13 dB for Pd=0.9, Pfa=1e-6
            pd = 0.9
        elif snr > 10:
            pd = 0.5
        else:
            pd = 0.1
            
        print(f"Target {i+1}: SNR = {snr:.1f} dB, Estimated Pd = {pd:.1f}")
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
