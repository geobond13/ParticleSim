"""
Timestep Study for PIC Numerical Heating Investigation

Goal: Identify stable timestep for PIC simulation by testing multiple dt values
      and tracking electron temperature evolution.

Problem: Week 11 revealed T_e jumps from 6.67 eV to 894 eV in one 50ps timestep.
         This prevents sheath BC validation.

Approach:
    - Test dt = [50ps, 20ps, 10ps, 5ps, 2ps, 1ps]
    - Disable RF, MCC, SEE (isolate PIC E-field acceleration)
    - Use sheath BC to prevent particle loss
    - Track T_e over 100 timesteps
    - Find maximum stable dt (T_e growth <10%)

Expected Result: Identify stable dt (likely 5-10 ps) for ABEP chamber

Author: AeriSat Systems
Date: October 31, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from intakesim.particles import ParticleArrayNumba
from intakesim.pic.mesh import Mesh1DPIC
from intakesim.pic.mover import push_pic_particles_1d
from intakesim.constants import e, m_e

# Suppress numpy warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PIC Timestep Stability Study")
print("=" * 70)
print()
print("Investigating numerical heating in PIC simulation.")
print("Testing timesteps: [50ps, 20ps, 10ps, 5ps, 2ps, 1ps]")
print()

# ==================== SETUP ====================

# Chamber geometry (simplified)
L_chamber = 0.06  # 6 cm
n_cells = 120  # dx = 0.5 mm
mesh = Mesh1DPIC(0.0, L_chamber, n_cells)

# Seed electrons (same as ABEP chamber)
n_seed = 500
weight_seed = 1e12
E_seed_eV = 10.0  # Initial energy

print("Setup:")
print(f"  Domain: {L_chamber*100:.1f} cm, {n_cells} cells")
print(f"  Seed: {n_seed} electrons at {E_seed_eV} eV")
print(f"  BC: sheath (energy-dependent)")
print()

# Timesteps to test (seconds)
timesteps = {
    '50ps': 50e-12,
    '20ps': 20e-12,
    '10ps': 10e-12,
    '5ps': 5e-12,
    '2ps': 2e-12,
    '1ps': 1e-12,
}

n_steps = 100  # Number of timesteps to evolve

# Storage for results
results = {}

# ==================== RUN STUDIES ====================

for label, dt in timesteps.items():
    print(f"Testing dt = {label}...")

    # Create fresh particle array
    max_particles = 20000
    particles = ParticleArrayNumba(max_particles=max_particles)

    # Initialize seed electrons (isotropic distribution)
    np.random.seed(42)  # Same initial condition for all tests
    x_seed = np.random.uniform(0, L_chamber, (n_seed, 3))
    x_seed[:, 1:] = 0

    # Maxwell-Boltzmann at E_seed_eV
    # For thermal plasma: KE = (3/2)kT, so T_eV = (2/3)*E_eV
    T_seed_eV = (2.0/3.0) * E_seed_eV
    v_thermal = np.sqrt(e * T_seed_eV / m_e)
    v_seed = np.random.randn(n_seed, 3) * v_thermal

    particles.add_particles(x_seed, v_seed, "e", weight=weight_seed)

    # Storage for this dt
    T_e_history = []
    time_history = []

    # Time loop
    for step in range(n_steps + 1):
        time = step * dt

        # Get T_e from diagnostics
        if step % 10 == 0:  # Sample every 10 steps
            # Calculate T_e
            from intakesim.pic.mover import calculate_electron_temperature_eV

            n_active = np.sum(particles.active)

            # DEBUG: Check initial state
            if step == 0 and label == '50ps':
                print(f"  DEBUG - Step 0:")
                print(f"    n_active = {n_active}")
                print(f"    n_particles = {particles.n_particles}")
                # Check first 3 electron velocities
                for i in range(min(3, particles.n_particles)):
                    if particles.active[i]:
                        v_mag = np.sqrt(np.sum(particles.v[i, :]**2))
                        E_eV = 0.5 * m_e * v_mag**2 / e
                        print(f"    Particle {i}: species_id={particles.species_id[i]}, v_mag = {v_mag:.2e} m/s, E = {E_eV:.2f} eV")

                # Manually calculate T_e to debug
                v_sq_sum = 0.0
                n_e = 0
                for i in range(particles.n_particles):
                    if particles.active[i] and particles.species_id[i] == 0:
                        v_sq = particles.v[i, 0]**2 + particles.v[i, 1]**2 + particles.v[i, 2]**2
                        v_sq_sum += v_sq
                        n_e += 1
                if n_e > 0:
                    v_sq_mean = v_sq_sum / n_e
                    T_e_manual = (m_e / (3.0 * e)) * v_sq_mean
                    print(f"    Manual T_e calc: n_e={n_e}, <v²>={v_sq_mean:.2e}, T_e={T_e_manual:.2f} eV")

            if n_active > 0:
                T_e_eV = calculate_electron_temperature_eV(
                    particles.v,
                    particles.active,
                    particles.species_id,
                    particles.n_particles
                )
            else:
                T_e_eV = 0.0

            T_e_history.append(T_e_eV)
            time_history.append(time * 1e9)  # Convert to nanoseconds

        # PIC push (no RF, no MCC, no SEE)
        if step < n_steps:
            push_diagnostics = push_pic_particles_1d(
                particles, mesh, dt,
                boundary_condition="sheath",
                phi_left=0.0,
                phi_right=0.0
            )

    # Store results
    results[label] = {
        'dt': dt,
        'time_ns': np.array(time_history),
        'T_e_eV': np.array(T_e_history),
        'final_T_e': T_e_history[-1] if len(T_e_history) > 0 else 0.0,
        'T_e_growth': (T_e_history[-1] / T_e_history[0] - 1.0) * 100 if len(T_e_history) > 1 else 0.0,
    }

    print(f"  Initial T_e: {T_e_history[0]:.2f} eV")
    print(f"  Final T_e: {T_e_history[-1]:.2f} eV")
    print(f"  Growth: {results[label]['T_e_growth']:.1f}%")
    print()

# ==================== ANALYSIS ====================

print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print(f"{'Timestep':<10} {'Initial T_e':<15} {'Final T_e':<15} {'Growth':<15} {'Status':<10}")
print("-" * 70)

stable_threshold = 10.0  # <10% growth considered stable

for label in ['50ps', '20ps', '10ps', '5ps', '2ps', '1ps']:
    res = results[label]
    initial_T_e = res['T_e_eV'][0]
    final_T_e = res['final_T_e']
    growth = res['T_e_growth']

    if growth < stable_threshold:
        status = "[OK] STABLE"
    else:
        status = "[X] UNSTABLE"

    print(f"{label:<10} {initial_T_e:>12.2f} eV  {final_T_e:>12.2f} eV  {growth:>12.1f}%  {status:<10}")

print()

# Find maximum stable timestep
stable_dts = []
for label in ['50ps', '20ps', '10ps', '5ps', '2ps', '1ps']:
    if results[label]['T_e_growth'] < stable_threshold:
        stable_dts.append((label, results[label]['dt']))

if len(stable_dts) > 0:
    # Find largest stable dt
    stable_dts.sort(key=lambda x: x[1], reverse=True)
    best_dt_label, best_dt_value = stable_dts[0]

    print(f"RECOMMENDATION: Use dt = {best_dt_label} ({best_dt_value*1e12:.1f} ps)")
    print(f"  T_e growth: {results[best_dt_label]['T_e_growth']:.1f}% (below {stable_threshold}% threshold)")
    print()
else:
    print("WARNING: No stable timestep found in tested range!")
    print("  Recommend testing smaller dt or adding E-field limiter.")
    print()

# ==================== VISUALIZATION ====================

print("Creating visualization...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: T_e vs time for all dt
for label in ['50ps', '20ps', '10ps', '5ps', '2ps', '1ps']:
    res = results[label]

    if res['T_e_growth'] < stable_threshold:
        linestyle = '-'
        linewidth = 2
        alpha = 1.0
    else:
        linestyle = '--'
        linewidth = 1.5
        alpha = 0.6

    ax1.plot(res['time_ns'], res['T_e_eV'],
             label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha)

ax1.set_xlabel('Time [ns]', fontsize=12)
ax1.set_ylabel('Electron Temperature [eV]', fontsize=12)
ax1.set_title('PIC Timestep Study: Temperature Evolution', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(title='Timestep', fontsize=10)
ax1.axhline(E_seed_eV * (2.0/3.0), color='black', linestyle=':',
            label=f'Initial T_e = {E_seed_eV * (2.0/3.0):.1f} eV', linewidth=1.5)
ax1.text(0.02, 0.98, 'Solid lines: Stable (growth <10%)\nDashed lines: Unstable',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Final T_e vs timestep
dt_values_ps = [50, 20, 10, 5, 2, 1]
final_T_e_values = [results[f'{dt}ps']['final_T_e'] for dt in dt_values_ps]

colors = ['red' if results[f'{dt}ps']['T_e_growth'] >= stable_threshold else 'green'
          for dt in dt_values_ps]

ax2.bar(range(len(dt_values_ps)), final_T_e_values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(dt_values_ps)))
ax2.set_xticklabels([f'{dt}ps' for dt in dt_values_ps])
ax2.set_xlabel('Timestep', fontsize=12)
ax2.set_ylabel('Final T_e [eV]', fontsize=12)
ax2.set_title('Final Temperature vs Timestep', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(E_seed_eV * (2.0/3.0) * 1.1, color='orange', linestyle='--',
            label=f'10% above initial', linewidth=2)
ax2.legend(fontsize=10)

# Add stability annotation
for i, dt in enumerate(dt_values_ps):
    label = f'{dt}ps'
    growth = results[label]['T_e_growth']
    if growth < stable_threshold:
        ax2.text(i, final_T_e_values[i] * 1.5, 'OK',
                ha='center', va='bottom', fontsize=12, color='green', fontweight='bold')
    else:
        ax2.text(i, final_T_e_values[i] * 1.5, 'X',
                ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('timestep_study.png', dpi=300, bbox_inches='tight')
print("Saved: timestep_study.png")
print()

print("=" * 70)
print("Timestep study complete!")
print("=" * 70)
