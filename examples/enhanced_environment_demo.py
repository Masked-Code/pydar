"""
Enhanced Environment Demonstration

This example demonstrates the advanced environmental modeling capabilities
of PyDar including:
- Sea and land clutter models
- Advanced propagation effects
- Terrain shadowing
- Atmospheric ducting
- Comprehensive environment scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from pydar import (
    RadarSystem, Antenna, LinearFMChirp, Target,
    EnhancedEnvironment, Atmosphere, SeaClutterModel, LandClutterModel
)


def demonstrate_clutter_models():
    """Demonstrate different clutter models."""
    print("=== Clutter Models Demonstration ===")
    
    # Create range and azimuth bins
    range_bins = np.linspace(1000, 20000, 40)
    azimuth_bins = np.linspace(-30, 30, 61)
    frequency = 10e9  # X-band
    
    # Sea clutter model
    print("\n1. Sea Clutter Model")
    sea_model = SeaClutterModel(
        wind_speed=15.0,
        wind_direction=270,  # Wind from west
        sea_state=6,         # Rough seas
        polarization='VV'
    )
    
    sea_clutter = sea_model.generate_clutter_rcs(range_bins, azimuth_bins, frequency)
    print(f"   Mean sea clutter RCS: {np.mean(sea_clutter):.3f} m²")
    print(f"   Max sea clutter RCS: {np.max(sea_clutter):.3f} m²")
    
    # Land clutter models for different terrain types
    terrain_types = ['urban', 'rural', 'forest', 'desert', 'mountains']
    land_clutters = {}
    
    print("\n2. Land Clutter Models")
    for terrain in terrain_types:
        land_model = LandClutterModel(
            terrain_type=terrain,
            vegetation_density=0.6 if terrain in ['forest', 'rural'] else 0.2,
            terrain_roughness=2.0 if terrain == 'mountains' else 1.0,
            soil_moisture=0.4
        )
        
        land_clutter = land_model.generate_clutter_rcs(range_bins, azimuth_bins, frequency)
        land_clutters[terrain] = land_clutter
        
        print(f"   {terrain.capitalize()} clutter - Mean: {np.mean(land_clutter):.3f} m², "
              f"Max: {np.max(land_clutter):.3f} m²")
    
    # Visualize clutter maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Radar Clutter Models', fontsize=16)
    
    # Plot sea clutter
    im1 = axes[0, 0].imshow(10*np.log10(sea_clutter), aspect='auto', 
                           extent=[azimuth_bins[0], azimuth_bins[-1], 
                                  range_bins[-1]/1000, range_bins[0]/1000],
                           cmap='viridis')
    axes[0, 0].set_title('Sea Clutter (dBsm)')
    axes[0, 0].set_xlabel('Azimuth (degrees)')
    axes[0, 0].set_ylabel('Range (km)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot land clutter for different terrains
    terrain_plots = [('urban', (0, 1)), ('rural', (0, 2)), ('forest', (1, 0)), 
                    ('desert', (1, 1)), ('mountains', (1, 2))]
    
    for terrain, (row, col) in terrain_plots:
        clutter_db = 10*np.log10(land_clutters[terrain])
        im = axes[row, col].imshow(clutter_db, aspect='auto',
                                  extent=[azimuth_bins[0], azimuth_bins[-1], 
                                         range_bins[-1]/1000, range_bins[0]/1000],
                                  cmap='viridis')
        axes[row, col].set_title(f'{terrain.capitalize()} Clutter (dBsm)')
        axes[row, col].set_xlabel('Azimuth (degrees)')
        axes[row, col].set_ylabel('Range (km)')
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    plt.savefig('clutter_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return sea_model, land_clutters


def demonstrate_propagation_effects():
    """Demonstrate advanced propagation effects."""
    print("\n=== Propagation Effects Demonstration ===")
    
    # Create different atmospheric conditions
    atmospheres = {
        'Clear': Atmosphere(temperature=288, humidity=0.3, rain_rate=0.0),
        'Humid': Atmosphere(temperature=295, humidity=0.8, rain_rate=0.0),
        'Rainy': Atmosphere(temperature=285, humidity=0.9, rain_rate=5.0),
        'Storm': Atmosphere(temperature=280, humidity=0.95, rain_rate=20.0)
    }
    
    ranges = np.linspace(1000, 50000, 100)
    frequency = 10e9
    
    # Calculate propagation loss for different conditions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    print("\n1. Atmospheric Propagation Effects")
    for name, atmos in atmospheres.items():
        env = EnhancedEnvironment(atmos)
        
        losses = []
        for r in ranges:
            loss = env.calculate_enhanced_propagation_loss(
                frequency=frequency,
                target_range=r,
                radar_height=100,
                target_height=1000
            )
            losses.append(loss)
        
        # Convert to dB
        losses_db = -10 * np.log10(losses)
        
        ax1.plot(ranges/1000, losses_db, label=name, linewidth=2)
        
        print(f"   {name}: Loss at 10km = {losses_db[np.argmin(abs(ranges-10000))]:.1f} dB")
    
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Propagation Loss (dB)')
    ax1.set_title('Atmospheric Propagation Effects')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Demonstrate terrain shadowing
    print("\n2. Terrain Shadowing Effects")
    
    # Create terrain profiles
    terrain_ranges = np.linspace(0, 20000, 100)
    terrain_profiles = {
        'Flat': np.zeros_like(terrain_ranges),
        'Hills': 200 * np.sin(terrain_ranges / 3000) + 100,
        'Mountains': 500 * np.sin(terrain_ranges / 5000) + 300,
        'Valley': -100 * np.cos(terrain_ranges / 4000) + 200
    }
    
    env = EnhancedEnvironment()
    target_range = 15000
    
    for name, heights in terrain_profiles.items():
        # Set terrain profile
        env.set_terrain_profile(terrain_ranges, heights)
        
        # Calculate shadowing factor
        shadow_factor = env.propagation_model.terrain_shadowing(
            range_profile=terrain_ranges,
            height_profile=heights,
            radar_height=50,
            target_height=100,
            target_range=target_range
        )
        
        # Calculate enhanced propagation loss
        loss = env.calculate_enhanced_propagation_loss(
            frequency=frequency,
            target_range=target_range,
            radar_height=50,
            target_height=100
        )
        
        loss_db = -10 * np.log10(loss)
        shadow_db = -10 * np.log10(max(shadow_factor, 1e-6))
        
        print(f"   {name}: Shadowing = {shadow_db:.1f} dB, Total loss = {loss_db:.1f} dB")
        
        # Plot terrain profile
        ax2.plot(terrain_ranges/1000, heights, label=f'{name} Terrain', linewidth=2)
    
    ax2.axhline(y=50, color='red', linestyle='--', label='Radar Height')
    ax2.axhline(y=100, color='blue', linestyle='--', label='Target Height')
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Terrain Profiles for Shadowing Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('propagation_effects.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_comprehensive_scenario():
    """Demonstrate a comprehensive radar scenario with all environmental effects."""
    print("\n=== Comprehensive Radar Scenario ===")
    
    # Create enhanced environment
    atmosphere = Atmosphere(temperature=290, humidity=0.7, rain_rate=1.0)
    env = EnhancedEnvironment(atmosphere)
    
    # Add mixed clutter (coastal scenario)
    print("\n1. Setting up coastal environment...")
    
    # Sea clutter (offshore)
    sea_model = SeaClutterModel(
        wind_speed=12.0,
        wind_direction=270,  # Wind from west
        sea_state=4,
        polarization='VV'
    )
    
    # Land clutter (coastal urban area)
    land_model = LandClutterModel(
        terrain_type='urban',
        vegetation_density=0.3,
        terrain_roughness=1.2,
        soil_moisture=0.5
    )
    
    env.add_clutter_model(sea_model)
    env.add_clutter_model(land_model)
    
    # Set coastal terrain profile
    ranges = np.linspace(0, 30000, 150)
    # Simulate coastline: sea level to 0-10km, then rising land
    heights = np.where(ranges < 10000, 0, 50 * (ranges - 10000) / 20000)
    env.set_terrain_profile(ranges, heights)
    
    # Create radar system
    print("2. Creating radar system...")
    antenna = Antenna(gain=35, beamwidth_azimuth=1.5, beamwidth_elevation=1.5)
    waveform = LinearFMChirp(
        duration=50e-6,
        sample_rate=200e6,
        bandwidth=100e6,
        center_frequency=10e9
    )
    
    radar = RadarSystem(
        antenna=antenna,
        waveform=waveform,
        position=(0, 0, 50),  # 50m above sea level
        transmit_power=10000,  # 10 kW
        noise_figure=3.0,
        losses=4.0
    )
    
    # Add targets
    print("3. Adding targets...")
    targets = [
        Target(range=5000, velocity=0, rcs=1000, azimuth=0, elevation=2),    # Ship
        Target(range=8000, velocity=150, rcs=50, azimuth=5, elevation=10),   # Aircraft
        Target(range=15000, velocity=-80, rcs=20, azimuth=-10, elevation=5), # Helicopter
        Target(range=25000, velocity=200, rcs=100, azimuth=15, elevation=15) # Jet
    ]
    
    for i, target in enumerate(targets):
        env.add_target(target)
        print(f"   Target {i+1}: Range={target.range}m, RCS={target.rcs}m², "
              f"Az={target.azimuth}°, Vel={target.velocity}m/s")
    
    # Generate clutter map
    print("\n4. Generating clutter map...")
    range_bins = np.linspace(1000, 30000, 60)
    azimuth_bins = np.linspace(-20, 20, 41)
    
    clutter_map = env.generate_clutter_map(range_bins, azimuth_bins, radar.frequency)
    
    # Calculate target detectability
    print("\n5. Analyzing target detectability...")
    detection_results = []
    
    for i, target in enumerate(targets):
        # Calculate enhanced propagation loss
        prop_loss = env.calculate_enhanced_propagation_loss(
            frequency=radar.frequency,
            target_range=target.range,
            radar_height=50,
            target_height=target.range * np.sin(np.radians(target.elevation)),
            azimuth=target.azimuth
        )
        
        # Calculate SNR with enhanced propagation
        received_power = radar.radar_equation(target.range, target.rcs) * prop_loss
        snr_linear = received_power / radar.noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        # Estimate clutter power at target location
        range_idx = np.argmin(abs(range_bins - target.range))
        az_idx = np.argmin(abs(azimuth_bins - target.azimuth))
        clutter_rcs = clutter_map[range_idx, az_idx]
        clutter_power = radar.radar_equation(target.range, clutter_rcs) * prop_loss
        
        # Signal-to-clutter ratio
        scr_db = 10 * np.log10(received_power / clutter_power) if clutter_power > 0 else float('inf')
        
        detection_results.append({
            'target': i+1,
            'snr_db': snr_db,
            'scr_db': scr_db,
            'prop_loss_db': -10*np.log10(prop_loss),
            'detectable': snr_db > 13 and scr_db > 10  # Detection thresholds
        })
        
        print(f"   Target {i+1}: SNR={snr_db:.1f}dB, SCR={scr_db:.1f}dB, "
              f"PropLoss={-10*np.log10(prop_loss):.1f}dB, "
              f"Detectable={'Yes' if snr_db > 13 and scr_db > 10 else 'No'}")
    
    # Visualize the scenario
    print("\n6. Creating visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Coastal Radar Scenario', fontsize=16)
    
    # Plot 1: Clutter map
    clutter_db = 10 * np.log10(np.maximum(clutter_map, 1e-6))
    im1 = ax1.imshow(clutter_db, aspect='auto', origin='lower',
                     extent=[azimuth_bins[0], azimuth_bins[-1], 
                            range_bins[0]/1000, range_bins[-1]/1000],
                     cmap='viridis')
    
    # Overlay targets
    for i, target in enumerate(targets):
        ax1.scatter(target.azimuth, target.range/1000, 
                   s=100, c='red', marker='*', 
                   label=f'Target {i+1}' if i == 0 else "")
    
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Range (km)')
    ax1.set_title('Clutter Map with Targets')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='Clutter RCS (dBsm)')
    
    # Plot 2: Terrain profile
    ax2.plot(ranges/1000, heights, 'b-', linewidth=2, label='Terrain')
    ax2.axhline(y=50, color='red', linestyle='--', label='Radar Height')
    ax2.fill_between([0, 10], [0, 0], [50, 50], alpha=0.3, color='blue', label='Sea')
    ax2.fill_between([10, 30], [0, 50], alpha=0.3, color='brown', label='Land')
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Coastal Terrain Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: SNR vs Range
    target_ranges = [t.range for t in targets]
    target_snrs = [r['snr_db'] for r in detection_results]
    target_detectable = [r['detectable'] for r in detection_results]
    
    colors = ['green' if det else 'red' for det in target_detectable]
    ax3.scatter([r/1000 for r in target_ranges], target_snrs, c=colors, s=100)
    ax3.axhline(y=13, color='red', linestyle='--', label='Detection Threshold')
    ax3.set_xlabel('Range (km)')
    ax3.set_ylabel('SNR (dB)')
    ax3.set_title('Target Detectability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (r, snr) in enumerate(zip(target_ranges, target_snrs)):
        ax3.annotate(f'T{i+1}', (r/1000, snr), xytext=(5, 5), 
                    textcoords='offset points')
    
    # Plot 4: Signal-to-Clutter Ratio
    target_scrs = [r['scr_db'] for r in detection_results]
    ax4.scatter([r/1000 for r in target_ranges], target_scrs, c=colors, s=100)
    ax4.axhline(y=10, color='red', linestyle='--', label='SCR Threshold')
    ax4.set_xlabel('Range (km)')
    ax4.set_ylabel('SCR (dB)')
    ax4.set_title('Signal-to-Clutter Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (r, scr) in enumerate(zip(target_ranges, target_scrs)):
        ax4.annotate(f'T{i+1}', (r/1000, scr), xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('comprehensive_radar_scenario.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary
    detectable_count = sum(r['detectable'] for r in detection_results)
    print(f"\n=== Scenario Summary ===")
    print(f"Total targets: {len(targets)}")
    print(f"Detectable targets: {detectable_count}")
    print(f"Detection rate: {detectable_count/len(targets)*100:.1f}%")
    print(f"Mean clutter level: {np.mean(clutter_db):.1f} dBsm")
    print(f"Max clutter level: {np.max(clutter_db):.1f} dBsm")


def main():
    """Run all demonstrations."""
    print("PyDar Enhanced Environment Demonstration")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Demonstrate clutter models
        sea_model, land_clutters = demonstrate_clutter_models()
        
        # Demonstrate propagation effects
        demonstrate_propagation_effects()
        
        # Demonstrate comprehensive scenario
        demonstrate_comprehensive_scenario()
        
        print("\n" + "=" * 50)
        print("Enhanced Environment Demonstration Complete!")
        print("Generated plots:")
        print("- clutter_models_comparison.png")
        print("- propagation_effects.png") 
        print("- comprehensive_radar_scenario.png")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
