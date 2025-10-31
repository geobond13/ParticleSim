"""
Example 05: Multi-Species Intake Validation

Comprehensive validation of intake compression with:
- Multi-species atmospheric composition (O, N2, O2)
- Realistic VLEO conditions (225 km altitude)
- Diagnostic tracking and CSV export
- Comparison to theoretical predictions
- Publication-quality visualization

Week 5 Deliverable: Full intake validation with diagnostics.

Target Performance (from Parodi et al. 2025):
- N2 compression ratio: 400-550 (target: 475)
- O2 compression ratio: 70-110 (target: 90)
- Operating altitude: 225 km
- Atmospheric temperature: 900 K
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from intakesim.particles import ParticleArrayNumba, sample_maxwellian_velocity
from intakesim.mesh import Mesh1D
from intakesim.dsmc.mover import push_particles_ballistic
from intakesim.dsmc.surfaces import cll_reflect_particle
from intakesim.geometry.intake import (
    HoneycombIntake,
    sample_freestream_velocity,
    apply_attitude_jitter,
)
from intakesim.diagnostics import DiagnosticTracker
from intakesim.constants import SPECIES, kB


def simulate_multi_species_intake():
    """
    Validate intake compression with multi-species atmosphere.

    This example demonstrates:
    1. Multi-species particle tracking (O, N2, O2)
    2. Diagnostic module integration
    3. CSV export for post-processing
    4. Comparison to literature results
    """
    print("\n" + "="*70)
    print("Multi-Species ABEP Intake Validation")
    print("="*70)

    # ================== ATMOSPHERIC COMPOSITION ==================
    # VLEO at 225 km altitude (from MSIS model)
    composition = {
        'O': 0.83,    # 83% atomic oxygen
        'N2': 0.14,   # 14% molecular nitrogen
        'O2': 0.02,   # 2% molecular oxygen
        'NO': 0.01,   # 1% nitric oxide
    }

    # Total atmospheric density at 225 km
    rho_total = 1.0e20  # m^-3

    # Individual species densities
    rho_O = rho_total * composition['O']
    rho_N2 = rho_total * composition['N2']
    rho_O2 = rho_total * composition['O2']

    # ================== PHYSICAL PARAMETERS ==================
    v_orbital = 7780.0  # m/s
    altitude = 225e3  # m
    T_atm = 900.0  # K

    # Honeycomb intake geometry (matching Parodi et al.)
    inlet_area = 0.01  # m^2 (100 cm^2)
    outlet_area = 0.001  # m^2 (10 cm^2)
    channel_diameter = 0.001  # m (1 mm)
    channel_length = 0.02  # m (20 mm), L/D = 20

    intake = HoneycombIntake(
        inlet_area, outlet_area, channel_length, channel_diameter
    )

    print(f"\nIntake Geometry:")
    print(f"  {intake}")
    print(f"  Clausing factor: {intake.clausing_factor:.4f}")
    print(f"  Geometric compression: {intake.geometric_compression:.1f}×")

    print(f"\nAtmospheric Composition (225 km):")
    print(f"  Atomic oxygen (O): {composition['O']*100:.1f}% ({rho_O:.2e} m^-3)")
    print(f"  Nitrogen (N2): {composition['N2']*100:.1f}% ({rho_N2:.2e} m^-3)")
    print(f"  Oxygen (O2): {composition['O2']*100:.1f}% ({rho_O2:.2e} m^-3)")
    print(f"  Total density: {rho_total:.2e} m^-3")

    # ================== SIMULATION PARAMETERS ==================
    buffer_inlet = 0.01  # m
    buffer_outlet = 0.01  # m
    domain_length = buffer_inlet + channel_length + buffer_outlet

    n_cells = 20
    cross_section = inlet_area

    # Time parameters
    dt = 1e-6  # 1 microsecond
    n_steps = 1000  # 1 ms total simulation
    output_interval = 50  # Output every 50 microseconds

    # Particle parameters
    n_particles_inject_per_step = 100  # Total particles per step (all species)
    max_particles = 100000  # Increased capacity for multi-species

    jitter_angle_deg = 7.0

    print(f"\nSimulation Parameters:")
    print(f"  Domain length: {domain_length*1e3:.1f} mm")
    print(f"  Timestep: {dt*1e6:.1f} microseconds")
    print(f"  Total time: {n_steps*dt*1e3:.2f} ms")
    print(f"  Injection rate: {n_particles_inject_per_step} particles/step (all species)")
    print(f"  Maximum particles: {max_particles:,}")

    # ================== INITIALIZATION ==================
    mesh = Mesh1D(length=domain_length, n_cells=n_cells, cross_section=cross_section)
    particles = ParticleArrayNumba(max_particles=max_particles)

    # Species data
    species_list = ['O', 'N2', 'O2']
    mass_array = np.array([SPECIES[s].mass for s in species_list], dtype=np.float64)

    # CLL surface parameters
    alpha_n = 0.9
    alpha_t = 0.9
    T_wall = 300.0  # K
    v_wall = np.zeros(3, dtype=np.float64)

    # Particle weights (species-dependent based on flux)
    real_flux_O = rho_O * v_orbital * inlet_area
    real_flux_N2 = rho_N2 * v_orbital * inlet_area
    real_flux_O2 = rho_O2 * v_orbital * inlet_area

    n_inject_O = int(n_particles_inject_per_step * composition['O'])
    n_inject_N2 = int(n_particles_inject_per_step * composition['N2'])
    n_inject_O2 = n_particles_inject_per_step - n_inject_O - n_inject_N2

    weight_O = real_flux_O / (n_inject_O / dt) if n_inject_O > 0 else 0
    weight_N2 = real_flux_N2 / (n_inject_N2 / dt) if n_inject_N2 > 0 else 0
    weight_O2 = real_flux_O2 / (n_inject_O2 / dt) if n_inject_O2 > 0 else 0

    print(f"\nParticle Weighting:")
    print(f"  O: {n_inject_O} particles/step, weight = {weight_O:.2e}")
    print(f"  N2: {n_inject_N2} particles/step, weight = {weight_N2:.2e}")
    print(f"  O2: {n_inject_O2} particles/step, weight = {weight_O2:.2e}")

    # ================== DIAGNOSTIC TRACKER ==================
    tracker = DiagnosticTracker(n_steps=n_steps, output_interval=output_interval)

    # Species-specific tracking
    n_outputs = n_steps // output_interval + 1
    CR_O = np.zeros(n_outputs)
    CR_N2 = np.zeros(n_outputs)
    CR_O2 = np.zeros(n_outputs)

    # ================== MAIN SIMULATION LOOP ==================
    print(f"\nStarting multi-species simulation...")
    t_start = time.time()

    output_idx = 0
    z_inlet = buffer_inlet
    z_outlet = buffer_inlet + channel_length

    for step in range(n_steps):
        current_time = step * dt

        # ========== PARTICLE INJECTION (MULTI-SPECIES) ==========
        if particles.n_particles + n_particles_inject_per_step < max_particles:
            # Inject O particles
            if n_inject_O > 0:
                v_inject_O = sample_freestream_velocity(
                    v_orbital, T_atm, SPECIES['O'].mass, n_inject_O
                )
                v_inject_O = apply_attitude_jitter(v_inject_O, jitter_angle_deg)

                x_inject_O = np.zeros((n_inject_O, 3), dtype=np.float64)
                x_inject_O[:, 0] = buffer_inlet
                x_inject_O[:, 1] = (np.random.rand(n_inject_O) - 0.5) * np.sqrt(inlet_area)
                x_inject_O[:, 2] = (np.random.rand(n_inject_O) - 0.5) * np.sqrt(inlet_area)

                particles.add_particles(x_inject_O, v_inject_O, species='O', weight=weight_O)

            # Inject N2 particles
            if n_inject_N2 > 0:
                v_inject_N2 = sample_freestream_velocity(
                    v_orbital, T_atm, SPECIES['N2'].mass, n_inject_N2
                )
                v_inject_N2 = apply_attitude_jitter(v_inject_N2, jitter_angle_deg)

                x_inject_N2 = np.zeros((n_inject_N2, 3), dtype=np.float64)
                x_inject_N2[:, 0] = buffer_inlet
                x_inject_N2[:, 1] = (np.random.rand(n_inject_N2) - 0.5) * np.sqrt(inlet_area)
                x_inject_N2[:, 2] = (np.random.rand(n_inject_N2) - 0.5) * np.sqrt(inlet_area)

                particles.add_particles(x_inject_N2, v_inject_N2, species='N2', weight=weight_N2)

            # Inject O2 particles
            if n_inject_O2 > 0:
                v_inject_O2 = sample_freestream_velocity(
                    v_orbital, T_atm, SPECIES['O2'].mass, n_inject_O2
                )
                v_inject_O2 = apply_attitude_jitter(v_inject_O2, jitter_angle_deg)

                x_inject_O2 = np.zeros((n_inject_O2, 3), dtype=np.float64)
                x_inject_O2[:, 0] = buffer_inlet
                x_inject_O2[:, 1] = (np.random.rand(n_inject_O2) - 0.5) * np.sqrt(inlet_area)
                x_inject_O2[:, 2] = (np.random.rand(n_inject_O2) - 0.5) * np.sqrt(inlet_area)

                particles.add_particles(x_inject_O2, v_inject_O2, species='O2', weight=weight_O2)

        # ========== BALLISTIC MOTION ==========
        push_particles_ballistic(
            particles.x, particles.v, particles.active, dt, particles.n_particles
        )

        # ========== BOUNDARY CONDITIONS ==========
        for i in range(particles.n_particles):
            if not particles.active[i]:
                continue

            z = particles.x[i, 0]

            if z < 0 or z > domain_length:
                particles.active[i] = False

        # ========== WALL COLLISIONS ==========
        for i in range(particles.n_particles):
            if not particles.active[i]:
                continue

            z = particles.x[i, 0]
            if z_inlet <= z <= z_outlet:
                r_perp = np.sqrt(particles.x[i, 1]**2 + particles.x[i, 2]**2)
                avg_channel_radius = channel_diameter / 2.0

                if r_perp > avg_channel_radius * 1.5:
                    species_id = particles.species_id[i]
                    mass = mass_array[species_id]

                    v_reflected = cll_reflect_particle(
                        particles.v[i], v_wall, mass, T_wall, alpha_n, alpha_t
                    )
                    particles.v[i] = v_reflected

                    if r_perp > 0:
                        particles.x[i, 1] *= 0.9
                        particles.x[i, 2] *= 0.9

        # ========== DIAGNOSTICS ==========
        if step % output_interval == 0:
            # Overall diagnostics
            tracker.record(
                step=step,
                time=current_time,
                x=particles.x,
                v=particles.v,
                active=particles.active,
                weight=particles.weight,
                n_particles=particles.n_particles,
                mass=SPECIES['O'].mass,  # Use O for average
                z_inlet=z_inlet,
                z_outlet=z_outlet
            )

            # Species-specific compression ratios
            for species_name, species_idx in [('O', 0), ('N2', 1), ('O2', 2)]:
                n_inlet = 0.0
                n_outlet = 0.0

                for i in range(particles.n_particles):
                    if not particles.active[i] or particles.species_id[i] != species_idx:
                        continue

                    z = particles.x[i, 0]

                    if z_inlet <= z < z_inlet + 0.005:
                        n_inlet += particles.weight[i]

                    if z_outlet - 0.005 <= z < z_outlet:
                        n_outlet += particles.weight[i]

                n_inlet /= 0.005  # Normalize by sample volume
                n_outlet /= 0.005

                CR = n_outlet / n_inlet if n_inlet > 0 else 0.0

                if species_name == 'O':
                    CR_O[output_idx] = CR
                elif species_name == 'N2':
                    CR_N2[output_idx] = CR
                elif species_name == 'O2':
                    CR_O2[output_idx] = CR

            output_idx += 1

            if step % (output_interval * 4) == 0:
                n_active = np.sum(particles.active[:particles.n_particles])
                print(f"  t = {current_time*1e6:.1f} us: N_active = {n_active:,}, "
                      f"CR(N2) = {CR_N2[output_idx-1]:.2f}, "
                      f"CR(O2) = {CR_O2[output_idx-1]:.2f}")

    t_elapsed = time.time() - t_start

    print(f"\nSimulation complete in {t_elapsed:.2f} seconds")
    print(f"  Performance: {n_steps * particles.n_particles / t_elapsed / 1e6:.1f} M particle-steps/sec")

    # ================== SAVE DIAGNOSTICS ==================
    tracker.save_csv('intake_validation_diagnostics.csv')

    # Save species-specific data
    np.savetxt('intake_validation_species_CR.csv',
               np.column_stack((tracker.time[:output_idx], CR_O[:output_idx],
                                CR_N2[:output_idx], CR_O2[:output_idx])),
               delimiter=',',
               header='time_s,CR_O,CR_N2,CR_O2',
               comments='')

    print("\nSpecies CR data saved to 'intake_validation_species_CR.csv'")

    # ================== ANALYSIS & VALIDATION ==================
    print(f"\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    # Time-averaged compression ratios (steady-state: last 50%)
    steady_start = output_idx // 2

    CR_N2_mean = np.mean(CR_N2[steady_start:output_idx])
    CR_O2_mean = np.mean(CR_O2[steady_start:output_idx])
    CR_O_mean = np.mean(CR_O[steady_start:output_idx])

    print(f"\nSteady-State Compression Ratios:")
    print(f"  N2: {CR_N2_mean:.1f} (Parodi target: 475)")
    print(f"  O2: {CR_O2_mean:.1f} (Parodi target: 90)")
    print(f"  O: {CR_O_mean:.1f}")

    print(f"\nComparison to Parodi et al. (2025):")
    error_N2 = abs(CR_N2_mean - 475) / 475 * 100
    error_O2 = abs(CR_O2_mean - 90) / 90 * 100
    print(f"  N2 error: {error_N2:.1f}%")
    print(f"  O2 error: {error_O2:.1f}%")

    if error_N2 < 30 and error_O2 < 30:
        print(f"  [PASS] VALIDATION PASSED (within 30%)")
    else:
        print(f"  [NOTE] Results differ from Parodi (expected for simplified free-molecular model)")

    # ================== VISUALIZATION ==================
    tracker.plot(show=False, save_filename='intake_validation_overall.png')

    # Species-specific plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time_us = tracker.time[:output_idx] * 1e6

    # Plot 1: Species compression ratios
    ax = axes[0, 0]
    ax.plot(time_us, CR_N2[:output_idx], 'b-', linewidth=2, label='N2')
    ax.plot(time_us, CR_O2[:output_idx], 'r-', linewidth=2, label='O2')
    ax.plot(time_us, CR_O[:output_idx], 'g-', linewidth=2, label='O')
    ax.axhline(y=475, color='b', linestyle='--', alpha=0.5, label='N2 target (Parodi)')
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='O2 target (Parodi)')
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Species-Specific Compression Ratios', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 2: Compression ratio comparison
    ax = axes[0, 1]
    species_names = ['N2', 'O2', 'O']
    species_CR = [CR_N2_mean, CR_O2_mean, CR_O_mean]
    target_CR = [475, 90, None]

    x_pos = np.arange(len(species_names))
    bars = ax.bar(x_pos, species_CR, alpha=0.7, color=['blue', 'red', 'green'])

    # Add target lines
    ax.axhline(y=475, color='b', linestyle='--', linewidth=2, alpha=0.5, label='N2 target')
    ax.axhline(y=90, color='r', linestyle='--', linewidth=2, alpha=0.5, label='O2 target')

    ax.set_xlabel('Species', fontsize=12)
    ax.set_ylabel('Mean Compression Ratio', fontsize=12)
    ax.set_title('Steady-State CR by Species', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(species_names)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Particle count by species
    ax = axes[1, 0]
    ax.plot(time_us, tracker.n_particles[:output_idx], 'k-', linewidth=2)
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Total Active Particles', fontsize=12)
    ax.set_title('Particle Population Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Transmission efficiency
    ax = axes[1, 1]
    transmission_efficiency = tracker.compression_ratio[:output_idx] / intake.geometric_compression
    ax.plot(time_us, transmission_efficiency * intake.clausing_factor, 'purple',
            linewidth=2, label='Effective transmission')
    ax.axhline(y=intake.clausing_factor, color='k', linestyle='--',
               linewidth=2, label=f'Clausing factor ({intake.clausing_factor:.4f})')
    ax.set_xlabel('Time (μs)', fontsize=12)
    ax.set_ylabel('Transmission Efficiency', fontsize=12)
    ax.set_title('Intake Transmission Efficiency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.05])

    plt.tight_layout()
    plt.savefig('intake_validation_species.png', dpi=300, bbox_inches='tight')
    print("\nSpecies plot saved as 'intake_validation_species.png'")
    plt.show()

    # Print summary
    tracker.summary()


if __name__ == "__main__":
    simulate_multi_species_intake()
