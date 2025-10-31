"""
Example 08: Romano et al. (2021) Diffuse Intake Benchmark

Demonstrates:
- Compression efficiency measurement (eta_c)
- Diffuse surface model validation
- Altitude sweep (150-250 km)

Reference: Romano et al. (2021) "Intake Design for an Atmospheric Breathing Electric Propulsion System"

Compression efficiency: eta_c = CR_measured / CR_geometric
- eta_c < 1: Diffuse walls reduce compression (expected)
- eta_c = 1: Perfect geometric compression
- eta_c > 1: Super-geometric compression (unusual, requires investigation)
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


def run_romano_case(altitude_km, n_steps=1500, n_particles_per_step=50, verbose=False):
    """
    Run single altitude case for Romano benchmark.

    Args:
        altitude_km: Altitude (150, 175, 200, 225, 250)
        n_steps: Simulation timesteps
        n_particles_per_step: Particles per step
        verbose: Print progress

    Returns:
        Dictionary with results
    """

    # Atmospheric data (NRLMSISE-00 model)
    atm_data = {
        150: {'rho': 2.1e18, 'T': 600, 'composition': {'N2': 0.78, 'O2': 0.21, 'O': 0.01}},
        175: {'rho': 3.8e17, 'T': 700, 'composition': {'N2': 0.60, 'O2': 0.15, 'O': 0.25}},
        200: {'rho': 4.2e17, 'T': 900, 'composition': {'N2': 0.14, 'O2': 0.02, 'O': 0.83}},
        225: {'rho': 1.8e17, 'T': 1000, 'composition': {'N2': 0.05, 'O2': 0.005, 'O': 0.945}},
        250: {'rho': 7.5e16, 'T': 1100, 'composition': {'N2': 0.02, 'O2': 0.001, 'O': 0.979}},
    }

    atm = atm_data[altitude_km]
    rho_atm = atm['rho']
    T_atm = atm['T']
    composition = atm['composition']
    v_orbital = 7780.0  # m/s

    # Diffuse surface
    sigma_n = 1.0
    sigma_t = 0.9
    T_wall = 300.0

    # Geometry
    inlet_area = 0.01  # m^2
    outlet_area = 0.001  # m^2
    channel_diameter = 0.001  # m
    L_over_D = 20.0
    channel_length = L_over_D * channel_diameter

    # Phase II: Enable multi-channel honeycomb geometry
    intake = HoneycombIntake(inlet_area, outlet_area, channel_length, channel_diameter,
                             use_multichannel=True)

    # Domain
    buffer_inlet = 0.01
    buffer_outlet = 0.01
    domain_length = buffer_inlet + channel_length + buffer_outlet

    n_cells = 20
    dt = 1e-6
    max_particles = 50000

    # Initialize
    mesh = Mesh1D(length=domain_length, n_cells=n_cells, cross_section=inlet_area)
    particles = ParticleArrayNumba(max_particles=max_particles)

    v_wall = np.zeros(3, dtype=np.float64)

    # Injection
    total_flux = rho_atm * v_orbital * inlet_area
    total_sim_flux = n_particles_per_step / dt
    particle_weight = total_flux / total_sim_flux

    species_list = list(composition.keys())
    n_inject = {sp: int(n_particles_per_step * composition[sp]) for sp in species_list}
    species_mass = {sp: SPECIES[sp].mass for sp in species_list}

    # Diagnostics
    z_inlet = buffer_inlet
    z_outlet = buffer_inlet + channel_length

    CR_list = []

    # Main loop
    t_start = time.time()

    for step in range(n_steps):
        if verbose and step % 300 == 0:
            print(f"    Step {step}/{n_steps} ({step/n_steps*100:.0f}%)")

        # Inject
        if particles.n_particles + n_particles_per_step < max_particles:
            for species_name in species_list:
                n_sp = n_inject[species_name]
                if n_sp > 0:
                    mass_sp = species_mass[species_name]

                    v_inject = sample_freestream_velocity(v_orbital, T_atm, mass_sp, n_sp)
                    v_inject = apply_attitude_jitter(v_inject, 7.0)

                    x_inject = np.zeros((n_sp, 3), dtype=np.float64)
                    x_inject[:, 0] = buffer_inlet
                    x_inject[:, 1] = (np.random.rand(n_sp) - 0.5) * np.sqrt(inlet_area)
                    x_inject[:, 2] = (np.random.rand(n_sp) - 0.5) * np.sqrt(inlet_area)

                    particles.add_particles(x_inject, v_inject, species=species_name, weight=particle_weight)

        # Move
        push_particles_ballistic(
            particles.x, particles.v, particles.active, dt, particles.n_particles
        )

        # Outflow
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

                    species_idx = particles.species_id[i]
                    mass = list(species_mass.values())[species_idx]

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

        # Measure CR
        if step > n_steps // 2 and step % 50 == 0:
            n_inlet = 0.0
            n_outlet = 0.0

            for i in range(particles.n_particles):
                if not particles.active[i]:
                    continue

                z = particles.x[i, 0]

                if z_inlet <= z < z_inlet + 0.005:
                    n_inlet += particles.weight[i]

                if z_outlet - 0.005 <= z < z_outlet:
                    n_outlet += particles.weight[i]

            inlet_volume = 0.005 * inlet_area
            outlet_volume = 0.005 * outlet_area

            n_density_inlet = n_inlet / inlet_volume if inlet_volume > 0 else 0.0
            n_density_outlet = n_outlet / outlet_volume if outlet_volume > 0 else 0.0

            CR = n_density_outlet / n_density_inlet if n_density_inlet > 0 else 0.0

            if CR > 0:
                CR_list.append(CR)

    t_elapsed = time.time() - t_start

    # Results
    CR_mean = np.mean(CR_list) if len(CR_list) > 0 else 0.0
    CR_std = np.std(CR_list) if len(CR_list) > 0 else 0.0

    CR_geometric = intake.geometric_compression
    eta_c = CR_mean / CR_geometric if CR_geometric > 0 else 0.0

    return {
        'altitude_km': altitude_km,
        'CR_measured': CR_mean,
        'CR_std': CR_std,
        'CR_geometric': CR_geometric,
        'eta_c': eta_c,
        'compute_time_s': t_elapsed,
    }


def run_altitude_sweep(altitudes=[150, 175, 200, 225, 250], n_steps=1500):
    """
    Run Romano benchmark at multiple altitudes.

    Args:
        altitudes: List of altitudes [km]
        n_steps: Timesteps per case

    Returns:
        List of results dictionaries
    """
    print(f"\n{'='*70}")
    print("Example 08: Romano et al. (2021) Diffuse Intake Benchmark")
    print(f"{'='*70}\n")
    print("Running altitude sweep...")
    print(f"  Altitudes: {altitudes} km")
    print(f"  Timesteps per case: {n_steps}")
    print(f"  Total cases: {len(altitudes)}\n")

    # Romano reference data (Table 8)
    romano_eta_c = {
        150: 0.458,
        175: 0.412,
        200: 0.381,
        225: 0.358,
        250: 0.342,
    }

    results_list = []
    t_start_total = time.time()

    for i, alt in enumerate(altitudes):
        print(f"  [{i+1}/{len(altitudes)}] Running h = {alt} km...")
        result = run_romano_case(alt, n_steps=n_steps, n_particles_per_step=50, verbose=False)
        result['romano_eta_c'] = romano_eta_c.get(alt, 0.0)
        results_list.append(result)

        print(f"      -> eta_c = {result['eta_c']:.3f} " +
              f"(Romano: {result['romano_eta_c']:.3f}, " +
              f"error: {(result['eta_c']/result['romano_eta_c']-1)*100:+.0f}%)")
        print(f"      -> Compute time: {result['compute_time_s']:.1f} s\n")

    t_total = time.time() - t_start_total

    print(f"{'='*70}")
    print(f"Altitude sweep complete!")
    print(f"  Total compute time: {t_total:.1f} s ({t_total/60:.1f} min)")
    print(f"{'='*70}\n")

    return results_list


def plot_altitude_sweep(results_list):
    """
    Plot altitude sweep results.

    Args:
        results_list: List of dictionaries from run_altitude_sweep()
    """
    altitudes = [r['altitude_km'] for r in results_list]
    eta_c_sim = [r['eta_c'] for r in results_list]
    eta_c_romano = [r['romano_eta_c'] for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Compression efficiency vs altitude
    ax1.plot(altitudes, eta_c_romano, 'o-', linewidth=2, markersize=8,
             color='blue', label='Romano et al. (2021)', alpha=0.7)
    ax1.plot(altitudes, eta_c_sim, 's--', linewidth=2, markersize=8,
             color='red', label='IntakeSIM', alpha=0.7)
    ax1.set_xlabel('Altitude (km)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Compression Efficiency (eta_c)', fontsize=12, fontweight='bold')
    ax1.set_title('Diffuse Intake Performance vs Altitude', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.2)

    # Right plot: Error vs altitude
    errors = [(sim/ref - 1) * 100 for sim, ref in zip(eta_c_sim, eta_c_romano)]
    ax2.bar(altitudes, errors, width=15, color='red', alpha=0.6, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(30, color='orange', linestyle='--', linewidth=1, label='±30% tolerance')
    ax2.axhline(-30, color='orange', linestyle='--', linewidth=1)
    ax2.set_xlabel('Altitude (km)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
    ax2.set_title('IntakeSIM Error vs Romano Reference', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('example_08_romano_benchmark.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'example_08_romano_benchmark.png'")
    plt.show()


def print_summary_table(results_list):
    """
    Print summary table of results.

    Args:
        results_list: List of dictionaries from run_altitude_sweep()
    """
    print("\n" + "="*85)
    print("Summary Table: Romano Diffuse Intake Benchmark")
    print("="*85)
    print(f"{'Altitude':>10} {'CR_meas':>10} {'CR_geom':>10} {'eta_c_sim':>10} " +
          f"{'eta_c_ref':>10} {'Error':>10} {'Status':>10}")
    print("-"*85)

    for r in results_list:
        error_pct = (r['eta_c'] / r['romano_eta_c'] - 1) * 100 if r['romano_eta_c'] > 0 else 0
        status = "PASS" if abs(error_pct) <= 30 else "FAIL"
        print(f"{r['altitude_km']:>10.0f} {r['CR_measured']:>10.2f} {r['CR_geometric']:>10.2f} " +
              f"{r['eta_c']:>10.3f} {r['romano_eta_c']:>10.3f} {error_pct:>9.0f}% " +
              f"{status:>10}")

    print("="*85 + "\n")


if __name__ == "__main__":
    # Run altitude sweep
    results = run_altitude_sweep(altitudes=[150, 175, 200, 225, 250], n_steps=1500)

    # Print summary
    print_summary_table(results)

    # Plot results
    plot_altitude_sweep(results)

    print("\nExample complete! See 'example_08_romano_benchmark.png' for visualization.")
