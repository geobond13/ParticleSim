"""
Grid Resolution Diagnostic Test

Goal: Verify that numerical heating is caused by dx >> lambda_D (under-resolved grid)
      by comparing coarse vs fine grid behavior.

Root Cause Hypothesis:
    - Current grid: dx = 0.5 mm, lambda_D = 0.051 mm → dx/lambda_D = 9.78
    - PIC stability requires: dx ≤ 0.5 × lambda_D
    - Under-resolved grid causes finite grid instability (aliasing of plasma oscillations)

Test:
    - Coarse grid: 120 cells (dx = 500 μm) - CURRENT
    - Fine grid: 2400 cells (dx = 25 μm) - PROPER RESOLUTION (dx ~ 0.5×λ_D)

Expected Result:
    - Coarse grid: Numerical heating (temperature grows)
    - Fine grid: Stable (temperature constant or oscillates with small amplitude)

If hypothesis is correct: Fine grid will be stable, confirming grid resolution is the issue.

Author: AeriSat Systems
Date: October 31, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from intakesim.particles import ParticleArrayNumba
from intakesim.pic.mesh import Mesh1DPIC
from intakesim.pic.mover import push_pic_particles_1d, calculate_electron_temperature_eV
from intakesim.constants import e, m_e, eps0

# Suppress numpy warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Grid Resolution Diagnostic Test")
print("=" * 70)
print()
print("Testing hypothesis: Numerical heating caused by dx >> lambda_D")
print()

# ==================== SETUP ====================

# Chamber geometry
L_chamber = 0.06  # 6 cm

# Target plasma parameters
n_e_target = 1.65e17  # m^-3
T_e_target = 7.8  # eV

# Calculate Debye length
lambda_D = np.sqrt(eps0 * T_e_target * e / (n_e_target * e**2))

print("Target Plasma Parameters:")
print(f"  n_e = {n_e_target:.2e} m^-3")
print(f"  T_e = {T_e_target:.1f} eV")
print(f"  lambda_D = {lambda_D*1e6:.2f} um")
print()

# Test configurations
configs = {
    'Coarse (CURRENT)': {
        'n_cells': 120,
        'description': 'Under-resolved (dx >> lambda_D)'
    },
    'Fine (PROPER)': {
        'n_cells': 2400,
        'description': 'Properly resolved (dx ~ 0.5*lambda_D)'
    }
}

# Simulation parameters
n_seed = 500
weight_seed = 1e12
T_seed_eV = (2.0/3.0) * 10.0  # Initial temperature
dt = 5e-12  # 5 ps timestep (passed plasma frequency test)
n_steps = 100

results = {}

# ==================== RUN TESTS ====================

for config_name, config in configs.items():
    n_cells = config['n_cells']
    dx = L_chamber / n_cells

    print(f"\nTesting: {config_name}")
    print(f"  {config['description']}")
    print(f"  n_cells = {n_cells}")
    print(f"  dx = {dx*1e6:.2f} um")
    print(f"  dx / lambda_D = {dx/lambda_D:.2f}")

    # Create mesh
    mesh = Mesh1DPIC(0.0, L_chamber, n_cells)

    # Create particle array
    max_particles = 20000
    particles = ParticleArrayNumba(max_particles=max_particles)

    # Initialize seed electrons (Maxwell-Boltzmann)
    np.random.seed(42)  # Same initial condition for both tests
    x_seed = np.random.uniform(0, L_chamber, (n_seed, 3))
    x_seed[:, 1:] = 0

    v_thermal = np.sqrt(e * T_seed_eV / m_e)
    v_seed = np.random.randn(n_seed, 3) * v_thermal

    particles.add_particles(x_seed, v_seed, "e", weight=weight_seed)

    # Storage
    T_e_history = []
    time_history = []
    E_max_history = []

    # Time loop
    for step in range(n_steps + 1):
        time = step * dt

        # Diagnostics every 10 steps
        if step % 10 == 0:
            n_active = np.sum(particles.active)

            if n_active > 0:
                T_e = calculate_electron_temperature_eV(
                    particles.v,
                    particles.active,
                    particles.species_id,
                    particles.n_particles,
                    electron_id=8
                )
            else:
                T_e = 0.0

            T_e_history.append(T_e)
            time_history.append(time * 1e9)  # ns

            # Track max E-field
            E_max_history.append(np.max(np.abs(mesh.E)))

            # Progress indicator for fine grid (takes longer)
            if n_cells > 1000 and step % 20 == 0:
                print(f"    Step {step}/{n_steps}: T_e = {T_e:.2f} eV, E_max = {np.max(np.abs(mesh.E)):.2e} V/m")

        # PIC push (no RF, no MCC, no SEE)
        if step < n_steps:
            push_diagnostics = push_pic_particles_1d(
                particles, mesh, dt,
                boundary_condition="sheath",
                phi_left=0.0,
                phi_right=0.0
            )

    # Store results
    results[config_name] = {
        'n_cells': n_cells,
        'dx': dx,
        'dx_over_lambda_D': dx / lambda_D,
        'time_ns': np.array(time_history),
        'T_e_eV': np.array(T_e_history),
        'E_max': np.array(E_max_history),
        'initial_T_e': T_e_history[0],
        'final_T_e': T_e_history[-1],
        'T_e_growth': (T_e_history[-1] / T_e_history[0] - 1.0) * 100,
    }

    print(f"  Initial T_e: {T_e_history[0]:.2f} eV")
    print(f"  Final T_e: {T_e_history[-1]:.2f} eV")
    print(f"  Growth: {results[config_name]['T_e_growth']:.1f}%")


# ==================== ANALYSIS ====================

print()
print("=" * 70)
print("DIAGNOSTIC RESULTS")
print("=" * 70)
print()

print(f"{'Configuration':<20} {'dx/lambda_D':<12} {'Initial T_e':<12} {'Final T_e':<12} {'Growth':<12} {'Status':<15}")
print("-" * 90)

stability_threshold = 20.0  # <20% growth considered acceptable

for config_name in ['Coarse (CURRENT)', 'Fine (PROPER)']:
    res = results[config_name]

    if res['T_e_growth'] < stability_threshold:
        status = "[OK] STABLE"
    else:
        status = "[X] UNSTABLE"

    print(f"{config_name:<20} {res['dx_over_lambda_D']:>8.2f}  {res['initial_T_e']:>10.2f} eV  {res['final_T_e']:>10.2f} eV  {res['T_e_growth']:>10.1f}%  {status:<15}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()

coarse_growth = results['Coarse (CURRENT)']['T_e_growth']
fine_growth = results['Fine (PROPER)']['T_e_growth']

if coarse_growth > stability_threshold and fine_growth < stability_threshold:
    print("[HYPOTHESIS CONFIRMED]")
    print()
    print("Numerical heating IS caused by under-resolved grid (dx >> lambda_D).")
    print()
    print("Evidence:")
    print(f"  - Coarse grid (dx/λ_D = {results['Coarse (CURRENT)']['dx_over_lambda_D']:.2f}): Growth = {coarse_growth:.1f}% [UNSTABLE]")
    print(f"  - Fine grid (dx/λ_D = {results['Fine (PROPER)']['dx_over_lambda_D']:.2f}): Growth = {fine_growth:.1f}% [STABLE]")
    print()
    print("ROOT CAUSE CONFIRMED: Grid resolution is the problem, not timestep or physics.")
    print()
    print("SOLUTION: Use fine grid (2400 cells) OR implement implicit solver.")
elif coarse_growth > stability_threshold and fine_growth > stability_threshold:
    print("[HYPOTHESIS PARTIALLY REJECTED]")
    print()
    print("Both grids are unstable! Grid resolution is not the only problem.")
    print()
    print("Evidence:")
    print(f"  - Coarse grid: Growth = {coarse_growth:.1f}%")
    print(f"  - Fine grid: Growth = {fine_growth:.1f}%")
    print()
    print("There may be a deeper issue with the PIC algorithm or physics model.")
    print("Consider: Implicit solver, energy-conserving scheme, or 0D model.")
elif coarse_growth < stability_threshold and fine_growth < stability_threshold:
    print("[UNEXPECTED: Both grids are stable]")
    print()
    print("Neither grid shows significant heating. Week 12 timestep study may have")
    print("had different initial conditions or setup. Investigate discrepancy.")
else:
    print("[UNEXPECTED: Fine grid is unstable but coarse grid is stable]")
    print()
    print("This contradicts the hypothesis. Investigate further.")

print()

# ==================== VISUALIZATION ====================

print("Creating visualization...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Temperature evolution
for config_name in ['Coarse (CURRENT)', 'Fine (PROPER)']:
    res = results[config_name]

    if res['T_e_growth'] < stability_threshold:
        linestyle = '-'
        linewidth = 2.5
        alpha = 1.0
    else:
        linestyle = '--'
        linewidth = 2.0
        alpha = 0.7

    label = f"{config_name} (dx/lambda_D={res['dx_over_lambda_D']:.1f})"
    ax1.plot(res['time_ns'], res['T_e_eV'], linestyle=linestyle,
             linewidth=linewidth, alpha=alpha, label=label)

ax1.axhline(T_seed_eV, color='black', linestyle=':', linewidth=1.5, label='Initial T_e')
ax1.set_xlabel('Time [ns]', fontsize=12)
ax1.set_ylabel('Electron Temperature [eV]', fontsize=12)
ax1.set_title('Grid Resolution Test: Temperature Evolution', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='best')
ax1.text(0.02, 0.98, 'Hypothesis: Fine grid (dx~λ_D) should be stable',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Max E-field evolution
for config_name in ['Coarse (CURRENT)', 'Fine (PROPER)']:
    res = results[config_name]
    ax2.plot(res['time_ns'], res['E_max'], linewidth=2, label=f"{config_name}")

ax2.set_xlabel('Time [ns]', fontsize=12)
ax2.set_ylabel('Max E-field [V/m]', fontsize=12)
ax2.set_title('Max Electric Field vs Time', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('grid_resolution_test.png', dpi=300, bbox_inches='tight')
print("Saved: grid_resolution_test.png")
print()

print("=" * 70)
print("Grid resolution diagnostic complete!")
print("=" * 70)
