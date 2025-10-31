"""
Example 07: Parodi et al. (2025) Intake Compression Reproduction

Demonstrates:
- Multi-species atmospheric intake (O, N2, O2)
- 200 km altitude conditions
- Diffuse CLL surface interactions
- Species-specific compression ratio tracking

Reference: Parodi et al. (2025) "Particle-based Simulation of an Air-Breathing Electric Propulsion System"

NOTE: This calculates LOCAL CR (outlet/inlet), not Parodi's SYSTEM CR (chamber/freestream).
See validation/README.md for detailed explanation of the difference.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intakesim.particles import ParticleArrayNumba
from intakesim.mesh import Mesh1D
from intakesim.dsmc.mover import push_particles_ballistic
from intakesim.dsmc.surfaces import cll_reflect_particle
from intakesim.geometry.intake import HoneycombIntake, sample_freestream_velocity, apply_attitude_jitter
from intakesim.constants import SPECIES
import time


def run_parodi_intake_simulation(n_steps=1000, n_particles_per_step=50, verbose=True):
    """
    Run intake compression simulation with Parodi et al. (2025) configuration.

    Args:
        n_steps: Number of simulation timesteps
        n_particles_per_step: Particles injected per step
        verbose: Print progress

    Returns:
        Dictionary with results
    """

    # =========================================================================
    # Configuration (Parodi et al. 2025)
    # =========================================================================

    # Atmospheric conditions at 200 km
    altitude_km = 200.0
    rho_atm = 4.2e17  # m^-3
    T_atm = 900.0  # K
    v_orbital = 7780.0  # m/s

    # Species composition (200 km atmosphere)
    composition = {
        'O': 0.83,    # 83% atomic oxygen
        'N2': 0.14,   # 14% molecular nitrogen
        'O2': 0.02,   # 2% molecular oxygen
    }

    # Diffuse surface (CLL parameters)
    sigma_n = 1.0  # Fully diffuse normal accommodation
    sigma_t = 0.9  # Nearly diffuse tangential accommodation
    T_wall = 300.0  # K

    # Intake geometry
    inlet_area = 0.01  # m^2
    outlet_area = 0.001  # m^2
    channel_diameter = 0.001  # m (1 mm)
    L_over_D = 20.0
    channel_length = L_over_D * channel_diameter

    # =========================================================================
    # Setup
    # =========================================================================

    if verbose:
        print(f"\n{'='*70}")
        print("Example 07: Parodi et al. (2025) Intake Compression")
        print(f"{'='*70}\n")
        print(f"Atmospheric conditions (h = {altitude_km:.0f} km):")
        print(f"  Density: {rho_atm:.2e} m^-3")
        print(f"  Temperature: {T_atm:.0f} K")
        print(f"  Orbital velocity: {v_orbital:.0f} m/s")
        print(f"  Species: O ({composition['O']*100:.0f}%), " +
              f"N2 ({composition['N2']*100:.0f}%), " +
              f"O2 ({composition['O2']*100:.0f}%)")
        print(f"\nSurface model:")
        print(f"  CLL diffuse (sigma_n={sigma_n}, sigma_t={sigma_t})")
        print(f"  Wall temperature: {T_wall:.0f} K")

    # Create intake geometry - Phase II: Multi-channel honeycomb
    intake = HoneycombIntake(inlet_area, outlet_area, channel_length, channel_diameter,
                             use_multichannel=True)

    if verbose:
        print(f"\nIntake geometry:")
        print(f"  Inlet area: {inlet_area} m^2")
        print(f"  Outlet area: {outlet_area} m^2")
        print(f"  Channel length: {channel_length*1e3:.1f} mm")
        print(f"  Channel diameter: {channel_diameter*1e3:.1f} mm")
        print(f"  L/D ratio: {L_over_D:.1f}")
        print(f"  Geometric compression: {intake.geometric_compression:.1f}")
        print(f"  Clausing factor: {intake.clausing_factor:.4f}")
        print(f"  Number of channels: {intake.n_channels:,}")

    # Simulation domain
    buffer_inlet = 0.01  # 1 cm buffer before inlet
    buffer_outlet = 0.01  # 1 cm buffer after outlet
    domain_length = buffer_inlet + channel_length + buffer_outlet

    n_cells = 20
    dt = 1e-6  # 1 microsecond
    max_particles = 30000

    # Initialize
    mesh = Mesh1D(length=domain_length, n_cells=n_cells, cross_section=inlet_area)
    particles = ParticleArrayNumba(max_particles=max_particles)

    # CLL parameters
    v_wall = np.zeros(3, dtype=np.float64)

    # Particle injection
    total_flux = rho_atm * v_orbital * inlet_area  # particles/s
    total_sim_flux = n_particles_per_step / dt
    particle_weight = total_flux / total_sim_flux

    # Species-specific injection
    n_inject_O = int(n_particles_per_step * composition['O'])
    n_inject_N2 = int(n_particles_per_step * composition['N2'])
    n_inject_O2 = int(n_particles_per_step * composition['O2'])

    # Species masses
    mass_O = SPECIES['O'].mass
    mass_N2 = SPECIES['N2'].mass
    mass_O2 = SPECIES['O2'].mass

    # Diagnostics boundaries
    z_inlet = buffer_inlet
    z_outlet = buffer_inlet + channel_length

    # Storage
    CR_O_list = []
    CR_N2_list = []
    CR_O2_list = []
    time_history = []

    if verbose:
        print(f"\nSimulation parameters:")
        print(f"  Timesteps: {n_steps}")
        print(f"  Timestep: {dt*1e6:.2f} us")
        print(f"  Particles/step: {n_particles_per_step}")
        print(f"  Particle weight: {particle_weight:.2e}")
        print(f"\n{'='*70}")
        print("Running simulation...")
        print(f"{'='*70}")

    # =========================================================================
    # Main simulation loop
    # =========================================================================

    t_start = time.time()

    for step in range(n_steps):
        # Progress
        if verbose and step % 200 == 0:
            print(f"  Step {step}/{n_steps} ({step/n_steps*100:.0f}%), " +
                  f"n_particles = {particles.n_particles}")

        # Inject particles
        if particles.n_particles + n_particles_per_step < max_particles:
            # Oxygen
            if n_inject_O > 0:
                v_inject_O = sample_freestream_velocity(v_orbital, T_atm, mass_O, n_inject_O)
                v_inject_O = apply_attitude_jitter(v_inject_O, 7.0)

                x_inject_O = np.zeros((n_inject_O, 3), dtype=np.float64)
                x_inject_O[:, 0] = buffer_inlet
                x_inject_O[:, 1] = (np.random.rand(n_inject_O) - 0.5) * np.sqrt(inlet_area)
                x_inject_O[:, 2] = (np.random.rand(n_inject_O) - 0.5) * np.sqrt(inlet_area)

                particles.add_particles(x_inject_O, v_inject_O, species='O', weight=particle_weight)

            # Nitrogen
            if n_inject_N2 > 0:
                v_inject_N2 = sample_freestream_velocity(v_orbital, T_atm, mass_N2, n_inject_N2)
                v_inject_N2 = apply_attitude_jitter(v_inject_N2, 7.0)

                x_inject_N2 = np.zeros((n_inject_N2, 3), dtype=np.float64)
                x_inject_N2[:, 0] = buffer_inlet
                x_inject_N2[:, 1] = (np.random.rand(n_inject_N2) - 0.5) * np.sqrt(inlet_area)
                x_inject_N2[:, 2] = (np.random.rand(n_inject_N2) - 0.5) * np.sqrt(inlet_area)

                particles.add_particles(x_inject_N2, v_inject_N2, species='N2', weight=particle_weight)

            # Oxygen molecules
            if n_inject_O2 > 0:
                v_inject_O2 = sample_freestream_velocity(v_orbital, T_atm, mass_O2, n_inject_O2)
                v_inject_O2 = apply_attitude_jitter(v_inject_O2, 7.0)

                x_inject_O2 = np.zeros((n_inject_O2, 3), dtype=np.float64)
                x_inject_O2[:, 0] = buffer_inlet
                x_inject_O2[:, 1] = (np.random.rand(n_inject_O2) - 0.5) * np.sqrt(inlet_area)
                x_inject_O2[:, 2] = (np.random.rand(n_inject_O2) - 0.5) * np.sqrt(inlet_area)

                particles.add_particles(x_inject_O2, v_inject_O2, species='O2', weight=particle_weight)

        # Move particles
        push_particles_ballistic(
            particles.x, particles.v, particles.active, dt, particles.n_particles
        )

        # Outflow boundaries
        for i in range(particles.n_particles):
            if not particles.active[i]:
                continue
            z = particles.x[i, 0]
            if z < 0 or z > domain_length:
                particles.active[i] = False

        # Wall collisions - MULTI-CHANNEL HONEYCOMB GEOMETRY (Phase II)
        for i in range(particles.n_particles):
            if not particles.active[i]:
                continue

            z = particles.x[i, 0]
            if z_inlet <= z <= z_outlet:
                # Determine which channel particle is in
                channel_id = intake.get_channel_id(particles.x[i, 1], particles.x[i, 2])

                if channel_id < 0:
                    particles.active[i] = False
                    continue

                # Get radial distance from channel centerline
                r_perp = intake.get_radial_distance(particles.x[i], channel_id)

                if r_perp > intake.channel_radius:
                    # Get wall normal for this channel
                    wall_normal = intake.get_wall_normal(particles.x[i], channel_id)

                    # Get particle mass
                    species_idx = particles.species_id[i]
                    if species_idx == 0:  # O
                        mass = mass_O
                    elif species_idx == 1:  # N2
                        mass = mass_N2
                    else:  # O2
                        mass = mass_O2

                    # Generalized CLL reflection
                    from intakesim.dsmc.surfaces import cll_reflect_particle_general
                    v_reflected = cll_reflect_particle_general(
                        particles.v[i], v_wall, mass, T_wall, sigma_n, sigma_t, wall_normal
                    )
                    particles.v[i] = v_reflected

                    # Radial pushback
                    cy, cz = intake.channel_centers[channel_id]
                    dy = particles.x[i, 1] - cy
                    dz = particles.x[i, 2] - cz
                    if r_perp > 0:
                        scale = (intake.channel_radius * 0.95) / r_perp
                        particles.x[i, 1] = cy + dy * scale
                        particles.x[i, 2] = cz + dz * scale

        # Measure CR (steady-state only)
        if step > n_steps // 2 and step % 50 == 0:
            for species_name, species_idx in [('O', 0), ('N2', 1), ('O2', 2)]:
                n_inlet = 0.0
                n_outlet = 0.0

                for i in range(particles.n_particles):
                    if not particles.active[i]:
                        continue
                    if particles.species_id[i] != species_idx:
                        continue

                    z = particles.x[i, 0]

                    if z_inlet <= z < z_inlet + 0.005:
                        n_inlet += particles.weight[i]

                    if z_outlet - 0.005 <= z < z_outlet:
                        n_outlet += particles.weight[i]

                # Volume normalization
                inlet_volume = 0.005 * inlet_area
                outlet_volume = 0.005 * outlet_area

                n_density_inlet = n_inlet / inlet_volume if inlet_volume > 0 else 0.0
                n_density_outlet = n_outlet / outlet_volume if outlet_volume > 0 else 0.0

                CR = n_density_outlet / n_density_inlet if n_density_inlet > 0 else 0.0

                if species_name == 'O':
                    pass  # O not measured in Parodi
                elif species_name == 'N2':
                    CR_N2_list.append(CR)
                elif species_name == 'O2':
                    CR_O2_list.append(CR)

            time_history.append(step * dt * 1e3)  # Convert to ms

    t_elapsed = time.time() - t_start

    # =========================================================================
    # Results
    # =========================================================================

    CR_N2_mean = np.mean(CR_N2_list) if len(CR_N2_list) > 0 else 0.0
    CR_N2_std = np.std(CR_N2_list) if len(CR_N2_list) > 0 else 0.0

    CR_O2_mean = np.mean(CR_O2_list) if len(CR_O2_list) > 0 else 0.0
    CR_O2_std = np.std(CR_O2_list) if len(CR_O2_list) > 0 else 0.0

    if verbose:
        print(f"\n{'='*70}")
        print("Simulation complete!")
        print(f"{'='*70}")
        print(f"  Compute time: {t_elapsed:.1f} s")
        print(f"  Final particle count: {particles.n_particles}")
        print(f"\n  Results (LOCAL CR = outlet/inlet):")
        print(f"    CR(N2) = {CR_N2_mean:.1f} +/- {CR_N2_std:.1f}")
        print(f"    CR(O2) = {CR_O2_mean:.1f} +/- {CR_O2_std:.1f}")
        print(f"\n  NOTE: Parodi's SYSTEM CR (chamber/freestream) = 475 for N2")
        print(f"        This is NOT directly comparable to our LOCAL CR.")
        print(f"        See validation/README.md for explanation.")
        print(f"{'='*70}\n")

    return {
        'CR_N2_mean': CR_N2_mean,
        'CR_N2_std': CR_N2_std,
        'CR_O2_mean': CR_O2_mean,
        'CR_O2_std': CR_O2_std,
        'CR_N2_history': np.array(CR_N2_list),
        'CR_O2_history': np.array(CR_O2_list),
        'time_history': np.array(time_history),
        'compute_time_s': t_elapsed,
        'n_particles_final': particles.n_particles,
    }


def plot_results(results):
    """
    Plot compression ratio time history.

    Args:
        results: Dictionary from run_parodi_intake_simulation()
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # N2 compression ratio
    ax1.plot(results['time_history'], results['CR_N2_history'], 'o-',
             color='blue', markersize=4, alpha=0.7, label='N2 LOCAL CR')
    ax1.axhline(results['CR_N2_mean'], color='blue', linestyle='--',
                linewidth=2, label=f'Mean = {results["CR_N2_mean"]:.1f}')
    ax1.fill_between([results['time_history'][0], results['time_history'][-1]],
                      results['CR_N2_mean'] - results['CR_N2_std'],
                      results['CR_N2_mean'] + results['CR_N2_std'],
                      color='blue', alpha=0.2, label=f'±1σ = {results["CR_N2_std"]:.1f}')
    ax1.set_ylabel('N2 Compression Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Example 07: Parodi et al. (2025) Intake Compression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # O2 compression ratio
    ax2.plot(results['time_history'], results['CR_O2_history'], 's-',
             color='red', markersize=4, alpha=0.7, label='O2 LOCAL CR')
    ax2.axhline(results['CR_O2_mean'], color='red', linestyle='--',
                linewidth=2, label=f'Mean = {results["CR_O2_mean"]:.1f}')
    ax2.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('O2 Compression Ratio', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('example_07_parodi_intake.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'example_07_parodi_intake.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("IntakeSIM Example 07: Parodi et al. (2025) Intake Compression")
    print("="*70)

    # Run simulation
    results = run_parodi_intake_simulation(n_steps=1000, n_particles_per_step=50, verbose=True)

    # Plot results
    plot_results(results)

    print("\nExample complete! See 'example_07_parodi_intake.png' for visualization.")
