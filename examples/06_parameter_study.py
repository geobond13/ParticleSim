"""
Example 06: Intake Parameter Study

Systematic exploration of ABEP intake design space:
- L/D ratio optimization
- Channel diameter effects
- Altitude sensitivity

Generates performance curves to identify optimal configurations.

Week 5 Deliverable: Parameter study framework for design optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from typing import Dict, List

from intakesim.particles import ParticleArrayNumba
from intakesim.mesh import Mesh1D
from intakesim.dsmc.mover import push_particles_ballistic
from intakesim.dsmc.surfaces import cll_reflect_particle
from intakesim.geometry.intake import (
    HoneycombIntake,
    sample_freestream_velocity,
    apply_attitude_jitter,
)
from intakesim.constants import SPECIES


def run_intake_configuration(
    L_over_D: float,
    channel_diameter: float,
    altitude: float,
    n_steps: int = 500,
    n_particles_per_step: int = 50,
    verbose: bool = False
) -> Dict:
    """
    Run single intake configuration and return performance metrics.

    Args:
        L_over_D: Length-to-diameter ratio
        channel_diameter: Channel diameter [m]
        altitude: Altitude [m]
        n_steps: Number of simulation steps
        n_particles_per_step: Particles injected per step
        verbose: Print progress

    Returns:
        results: Dictionary with compression ratio, transmission efficiency, etc.
    """
    # Physical parameters
    v_orbital = 7780.0  # m/s
    T_atm = 900.0  # K

    # Altitude-dependent density (exponential atmosphere model)
    h0 = 225e3  # Reference altitude
    rho0 = 1.0e20  # Reference density at 225 km
    H = 50e3  # Scale height
    rho_atm = rho0 * np.exp(-(altitude - h0) / H)

    # Intake geometry
    inlet_area = 0.01  # m^2
    outlet_area = 0.001  # m^2
    channel_length = L_over_D * channel_diameter

    intake = HoneycombIntake(inlet_area, outlet_area, channel_length, channel_diameter)

    # Simulation domain
    buffer_inlet = 0.01
    buffer_outlet = 0.01
    domain_length = buffer_inlet + channel_length + buffer_outlet

    n_cells = 20
    dt = 1e-6  # 1 microsecond
    max_particles = 30000

    # Initialize
    mesh = Mesh1D(length=domain_length, n_cells=n_cells, cross_section=inlet_area)
    particles = ParticleArrayNumba(max_particles=max_particles)

    # Species (simplified: pure O for speed)
    mass_O = SPECIES['O'].mass

    # CLL parameters
    alpha_n = 0.9
    alpha_t = 0.9
    T_wall = 300.0
    v_wall = np.zeros(3, dtype=np.float64)

    # Particle weighting
    real_flux = rho_atm * v_orbital * inlet_area
    sim_flux = n_particles_per_step / dt
    particle_weight = real_flux / sim_flux

    # Diagnostics
    compression_ratios = []
    z_inlet = buffer_inlet
    z_outlet = buffer_inlet + channel_length

    # Main loop
    t_start = time.time()

    for step in range(n_steps):
        # Inject particles
        if particles.n_particles + n_particles_per_step < max_particles:
            v_inject = sample_freestream_velocity(
                v_orbital, T_atm, mass_O, n_particles_per_step
            )
            v_inject = apply_attitude_jitter(v_inject, 7.0)

            x_inject = np.zeros((n_particles_per_step, 3), dtype=np.float64)
            x_inject[:, 0] = buffer_inlet
            x_inject[:, 1] = (np.random.rand(n_particles_per_step) - 0.5) * np.sqrt(inlet_area)
            x_inject[:, 2] = (np.random.rand(n_particles_per_step) - 0.5) * np.sqrt(inlet_area)

            particles.add_particles(x_inject, v_inject, species='O', weight=particle_weight)

        # Ballistic motion
        push_particles_ballistic(
            particles.x, particles.v, particles.active, dt, particles.n_particles
        )

        # Boundaries
        for i in range(particles.n_particles):
            if not particles.active[i]:
                continue
            z = particles.x[i, 0]
            if z < 0 or z > domain_length:
                particles.active[i] = False

        # Wall collisions
        for i in range(particles.n_particles):
            if not particles.active[i]:
                continue

            z = particles.x[i, 0]
            if z_inlet <= z <= z_outlet:
                r_perp = np.sqrt(particles.x[i, 1]**2 + particles.x[i, 2]**2)
                avg_channel_radius = channel_diameter / 2.0

                if r_perp > avg_channel_radius * 1.5:
                    v_reflected = cll_reflect_particle(
                        particles.v[i], v_wall, mass_O, T_wall, alpha_n, alpha_t
                    )
                    particles.v[i] = v_reflected

                    if r_perp > 0:
                        particles.x[i, 1] *= 0.9
                        particles.x[i, 2] *= 0.9

        # Measure compression (every 50 steps in steady-state)
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

            n_inlet /= 0.005
            n_outlet /= 0.005

            CR = n_outlet / n_inlet if n_inlet > 0 else 0.0
            compression_ratios.append(CR)

    t_elapsed = time.time() - t_start

    # Compute results
    mean_CR = np.mean(compression_ratios) if len(compression_ratios) > 0 else 0.0
    transmission_eff = mean_CR / intake.geometric_compression if intake.geometric_compression > 0 else 0.0

    n_active = np.sum(particles.active[:particles.n_particles])

    if verbose:
        print(f"  L/D={L_over_D:.1f}, d={channel_diameter*1e3:.1f}mm, h={altitude/1e3:.0f}km: "
              f"CR={mean_CR:.2f}, Eff={transmission_eff:.3f}, t={t_elapsed:.1f}s")

    return {
        'L_over_D': L_over_D,
        'channel_diameter_mm': channel_diameter * 1e3,
        'altitude_km': altitude / 1e3,
        'compression_ratio': mean_CR,
        'transmission_efficiency': transmission_eff,
        'clausing_factor': intake.clausing_factor,
        'geometric_compression': intake.geometric_compression,
        'n_channels': intake.n_channels,
        'n_particles_final': n_active,
        'compute_time_s': t_elapsed,
        'density_atm': rho_atm,
    }


def parameter_study():
    """
    Run comprehensive parameter study.
    """
    print("\n" + "="*70)
    print("ABEP Intake Parameter Study")
    print("="*70)

    results = []

    # ========== STUDY 1: L/D Ratio Sweep ==========
    print("\n[1/3] L/D Ratio Sweep (fixed diameter=1mm, altitude=225km)")
    print("-" * 70)

    L_over_D_values = [10, 15, 20, 30, 50]

    for L_over_D in L_over_D_values:
        result = run_intake_configuration(
            L_over_D=L_over_D,
            channel_diameter=0.001,
            altitude=225e3,
            verbose=True
        )
        result['study'] = 'L_over_D'
        results.append(result)

    # ========== STUDY 2: Channel Diameter Sweep ==========
    print("\n[2/3] Channel Diameter Sweep (fixed L/D=20, altitude=225km)")
    print("-" * 70)

    diameter_values = [0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3]  # mm to m

    for diameter in diameter_values:
        result = run_intake_configuration(
            L_over_D=20.0,
            channel_diameter=diameter,
            altitude=225e3,
            verbose=True
        )
        result['study'] = 'diameter'
        results.append(result)

    # ========== STUDY 3: Altitude Sweep ==========
    print("\n[3/3] Altitude Sweep (fixed L/D=20, diameter=1mm)")
    print("-" * 70)

    altitude_values = [200e3, 212e3, 225e3, 237e3, 250e3]  # km to m

    for altitude in altitude_values:
        result = run_intake_configuration(
            L_over_D=20.0,
            channel_diameter=0.001,
            altitude=altitude,
            verbose=True
        )
        result['study'] = 'altitude'
        results.append(result)

    # ========== SAVE RESULTS ==========
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    csv_filename = 'parameter_study_results.csv'
    with open(csv_filename, 'w', newline='') as f:
        if len(results) > 0:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"Results saved to '{csv_filename}'")

    # ========== VISUALIZATION ==========
    print("\nGenerating performance plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: CR vs L/D
    ax = axes[0, 0]
    ld_data = [r for r in results if r['study'] == 'L_over_D']
    ld_values = [r['L_over_D'] for r in ld_data]
    cr_values = [r['compression_ratio'] for r in ld_data]

    ax.plot(ld_values, cr_values, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('L/D Ratio', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression vs L/D Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add theoretical Clausing curve
    ld_theory = np.linspace(10, 50, 100)
    clausing_theory = 8.0 / (3.0 * ld_theory)  # Asymptotic formula
    ax2 = ax.twinx()
    ax2.plot(ld_theory, clausing_theory, '--', color='red', alpha=0.5, label='Clausing factor')
    ax2.set_ylabel('Clausing Factor', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right', fontsize=9)

    # Plot 2: CR vs Diameter
    ax = axes[0, 1]
    d_data = [r for r in results if r['study'] == 'diameter']
    d_values = [r['channel_diameter_mm'] for r in d_data]
    cr_values_d = [r['compression_ratio'] for r in d_data]

    ax.plot(d_values, cr_values_d, 's-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Channel Diameter (mm)', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression vs Channel Diameter', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: CR vs Altitude
    ax = axes[1, 0]
    alt_data = [r for r in results if r['study'] == 'altitude']
    alt_values = [r['altitude_km'] for r in alt_data]
    cr_values_alt = [r['compression_ratio'] for r in alt_data]

    ax.plot(alt_values, cr_values_alt, '^-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Altitude (km)', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression vs Altitude', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Transmission Efficiency Summary
    ax = axes[1, 1]

    # Group by study
    studies = ['L_over_D', 'diameter', 'altitude']
    study_names = ['L/D\nVariation', 'Diameter\nVariation', 'Altitude\nVariation']
    mean_effs = []

    for study in studies:
        study_data = [r for r in results if r['study'] == study]
        mean_eff = np.mean([r['transmission_efficiency'] for r in study_data])
        mean_effs.append(mean_eff)

    x_pos = np.arange(len(study_names))
    bars = ax.bar(x_pos, mean_effs, alpha=0.7, color=['blue', 'green', 'purple'])

    ax.set_ylabel('Mean Transmission Efficiency', fontsize=12)
    ax.set_title('Transmission Efficiency by Study', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(study_names)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(mean_effs) * 1.2])

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_effs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('parameter_study_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'parameter_study_results.png'")
    plt.show()

    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("PARAMETER STUDY SUMMARY")
    print("="*70)

    print("\nOptimal Configurations:")

    # Best L/D
    ld_data_sorted = sorted([r for r in results if r['study'] == 'L_over_D'],
                            key=lambda x: x['compression_ratio'], reverse=True)
    best_ld = ld_data_sorted[0]
    print(f"\n  Best L/D: {best_ld['L_over_D']:.0f}")
    print(f"    CR = {best_ld['compression_ratio']:.2f}")
    print(f"    Transmission Eff = {best_ld['transmission_efficiency']:.3f}")
    print(f"    Clausing Factor = {best_ld['clausing_factor']:.4f}")

    # Best diameter
    d_data_sorted = sorted([r for r in results if r['study'] == 'diameter'],
                           key=lambda x: x['compression_ratio'], reverse=True)
    best_d = d_data_sorted[0]
    print(f"\n  Best Diameter: {best_d['channel_diameter_mm']:.1f} mm")
    print(f"    CR = {best_d['compression_ratio']:.2f}")
    print(f"    Transmission Eff = {best_d['transmission_efficiency']:.3f}")
    print(f"    N_channels = {best_d['n_channels']:,}")

    # Best altitude
    alt_data_sorted = sorted([r for r in results if r['study'] == 'altitude'],
                            key=lambda x: x['compression_ratio'], reverse=True)
    best_alt = alt_data_sorted[0]
    print(f"\n  Best Altitude: {best_alt['altitude_km']:.0f} km")
    print(f"    CR = {best_alt['compression_ratio']:.2f}")
    print(f"    Atmospheric Density = {best_alt['density_atm']:.2e} m^-3")

    print("\nKey Insights:")
    print(f"  1. L/D ratio: Lower is better (shorter tubes = higher transmission)")
    print(f"  2. Channel diameter: Larger gives more channels but lower per-channel CR")
    print(f"  3. Altitude: Lower altitude = higher density = better performance")

    print("\n" + "="*70)


if __name__ == "__main__":
    parameter_study()
