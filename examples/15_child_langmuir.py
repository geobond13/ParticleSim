"""
Child-Langmuir Benchmark for Sheath Boundary Condition Validation

Goal: Validate energy-dependent sheath BC against analytical Child-Langmuir law
      in simple parallel-plate geometry.

Physics:
    Child-Langmuir law governs space-charge-limited current in vacuum diode:

    j = (4*ε₀/9) * sqrt(2*e/m_e) * V^(3/2) / d²

    where:
        j = current density [A/m²]
        V = anode-cathode potential [V]
        d = gap distance [m]

    For sheath at plasma wall:
        V_sheath ≈ 4.5 * k*T_e / e  (Bohm criterion)
        Sheath thickness ≈ 5 * λ_D

Expected Result:
    - Measured sheath thickness within 20% of 5*λ_D
    - Current density follows j ∝ V^(3/2)
    - Energy-dependent reflection reproduces Bohm criterion

Author: AeriSat Systems
Date: October 31, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from intakesim.particles import ParticleArrayNumba
from intakesim.pic.mesh import Mesh1DPIC
from intakesim.pic.mover import push_pic_particles_1d, calculate_electron_temperature_eV
from intakesim.constants import e, m_e, eps0, kB

# Suppress numpy warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Child-Langmuir Sheath Benchmark")
print("=" * 70)
print()
print("Validating energy-dependent sheath BC against analytical solution.")
print()

# ==================== SETUP ====================

# Simple parallel-plate geometry
d_gap = 0.01  # 1 cm gap
n_cells = 100  # 0.1 mm resolution
mesh = Mesh1DPIC(0.0, d_gap, n_cells)

# Plasma parameters (similar to ABEP chamber)
T_e_eV = 7.0  # Target electron temperature
n_plasma = 1e17  # Plasma density [m^-3]

# Derived parameters
lambda_D = np.sqrt(eps0 * T_e_eV * e / (n_plasma * e**2))  # Debye length
V_sheath_analytical = 4.5 * T_e_eV  # Bohm criterion [V]
sheath_thickness_analytical = 5.0 * lambda_D  # Expected sheath thickness [m]

print("Analytical Predictions:")
print(f"  T_e = {T_e_eV:.1f} eV")
print(f"  n_e = {n_plasma:.2e} m^-3")
print(f"  lambda_D = {lambda_D*1e3:.3f} mm")
print(f"  V_sheath = {V_sheath_analytical:.2f} V (Bohm criterion)")
print(f"  Sheath thickness = {sheath_thickness_analytical*1e3:.3f} mm (5*lambda_D)")
print()

# Child-Langmuir current density
def child_langmuir_current(V_applied, gap):
    """
    Child-Langmuir law for space-charge-limited current.

    j = (4*ε₀/9) * sqrt(2*e/m_e) * V^(3/2) / d²
    """
    coeff = (4.0 * eps0 / 9.0) * np.sqrt(2.0 * e / m_e)
    # Handle both scalars and arrays
    V_use = np.maximum(V_applied, 0.0)  # Clip to non-negative
    return coeff * V_use**(3.0/2.0) / gap**2


# ==================== SIMULATION ====================

# Test multiple applied voltages
V_applied_list = [5.0, 10.0, 20.0, 30.0, 40.0]  # Volts
results = {}

for V_applied in V_applied_list:
    print(f"Testing V_applied = {V_applied:.1f} V...")

    # Create particle array
    max_particles = 50000
    particles = ParticleArrayNumba(max_particles=max_particles)

    # Initialize seed electrons (isotropic at T_e)
    n_seed = 1000
    weight_seed = n_plasma * (d_gap * 0.001**2) / n_seed  # Volume ~ d_gap × 1mm²

    np.random.seed(42)
    x_seed = np.random.uniform(0.002, 0.008, (n_seed, 3))  # Central region (2-8 mm)
    x_seed[:, 1:] = 0

    # Maxwell-Boltzmann velocity distribution at T_e
    # For 3D MB: each component has variance sigma² = kT/m = e*T_eV/m_e
    v_thermal = np.sqrt(e * T_e_eV / m_e)
    v_seed = np.random.randn(n_seed, 3) * v_thermal

    particles.add_particles(x_seed, v_seed, "e", weight=weight_seed)

    # Storage for diagnostics
    time_history = []
    current_history = []
    n_active_history = []
    T_e_history = []
    n_absorbed_cumulative = 0

    # Time evolution
    dt = 1e-12  # 1 ps timestep (small for stability in simple test)
    n_steps = 1000
    area = 0.001**2  # 1 mm² cross-section

    for step in range(n_steps):
        time = step * dt

        # PIC push with sheath BC and applied voltage
        push_diagnostics = push_pic_particles_1d(
            particles, mesh, dt,
            boundary_condition="sheath",
            phi_left=0.0,
            phi_right=V_applied  # Applied voltage at right wall
        )

        # Track absorbed particles
        n_absorbed_this_step = push_diagnostics.get('n_absorbed', 0)
        n_absorbed_cumulative += n_absorbed_this_step

        # Diagnostics every 50 steps
        if step % 50 == 0:
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

            time_history.append(time * 1e9)  # ns
            n_active_history.append(n_active)
            T_e_history.append(T_e)

            # Calculate current density from absorbed particles
            # j = (charge × weight × n_absorbed) / (area × time)
            # Note: n_absorbed_cumulative accounts for all absorbed since last diagnostic
            if step > 0:
                dt_diagnostic = 50 * dt  # Time between diagnostics
                j_avg = (e * weight_seed * n_absorbed_cumulative) / (area * dt_diagnostic)
                current_history.append(j_avg)
                n_absorbed_cumulative = 0  # Reset counter
            else:
                current_history.append(0.0)

    # Calculate average steady-state current (last 50% of simulation)
    steady_start = len(current_history) // 2
    j_measured = np.mean(current_history[steady_start:]) if len(current_history) > 0 else 0.0

    # Child-Langmuir prediction
    j_analytical = child_langmuir_current(V_applied, d_gap)

    # Store results
    results[V_applied] = {
        'time_ns': np.array(time_history),
        'n_active': np.array(n_active_history),
        'T_e_eV': np.array(T_e_history),
        'current_A_m2': np.array(current_history),
        'j_measured': j_measured,
        'j_analytical': j_analytical,
        'ratio': j_measured / j_analytical if j_analytical > 0 else 0.0,
    }

    # Additional diagnostics
    final_n_active = n_active_history[-1] if len(n_active_history) > 0 else 0
    final_T_e = T_e_history[-1] if len(T_e_history) > 0 else 0

    print(f"  j_measured = {j_measured:.3e} A/m²")
    print(f"  j_analytical = {j_analytical:.3e} A/m² (Child-Langmuir)")
    print(f"  Ratio = {results[V_applied]['ratio']:.2f}")
    print(f"  Final: n_active = {final_n_active}, T_e = {final_T_e:.2f} eV")
    print(f"  V_sheath (from T_e) = {4.5 * final_T_e:.2f} V")
    print()


# ==================== ANALYSIS ====================

print("=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)
print()
print(f"{'V_applied [V]':<15} {'j_measured':<18} {'j_CL':<18} {'Ratio':<10} {'Status':<10}")
print("-" * 70)

for V in V_applied_list:
    res = results[V]
    j_meas = res['j_measured']
    j_anal = res['j_analytical']
    ratio = res['ratio']

    # Within 30% considered good agreement for this benchmark
    if 0.7 <= ratio <= 1.3:
        status = "[OK]"
    else:
        status = "[X]"

    print(f"{V:<15.1f} {j_meas:<18.3e} {j_anal:<18.3e} {ratio:<10.2f} {status:<10}")

print()

# Check if j ∝ V^(3/2) power law holds
V_array = np.array(V_applied_list)
j_measured_array = np.array([results[V]['j_measured'] for V in V_applied_list])

# Fit power law: j = A * V^n
# log(j) = log(A) + n*log(V)
valid_points = j_measured_array > 0
if np.sum(valid_points) >= 3:
    log_V = np.log(V_array[valid_points])
    log_j = np.log(j_measured_array[valid_points])
    coeffs = np.polyfit(log_V, log_j, 1)
    n_fitted = coeffs[0]

    print(f"Power Law Fit:")
    print(f"  j proportional to V^{n_fitted:.2f}")
    print(f"  Expected: j proportional to V^1.5 (Child-Langmuir)")

    if 1.3 <= n_fitted <= 1.7:
        print(f"  [OK] Power law within 15% of theoretical 3/2")
    else:
        print(f"  [X] Power law deviates from Child-Langmuir prediction")
    print()
else:
    print("WARNING: Insufficient data points for power law fit")
    print()


# ==================== SHEATH THICKNESS ANALYSIS ====================

# Use V_applied = 30V case for spatial analysis
V_test = 30.0
print("Sheath Thickness Analysis (V = 30 V):")
print(f"  Analytical prediction: {sheath_thickness_analytical*1e3:.3f} mm (5*lambda_D)")

# For actual sheath thickness measurement, we'd need to look at density profile
# This requires field solver which we're not using here
# For now, report what we can measure

print(f"  Note: Detailed spatial analysis requires field solver")
print(f"        Current test validates energy-dependent BC only")
print()


# ==================== VISUALIZATION ====================

print("Creating visualization...")

fig = plt.figure(figsize=(14, 10))

# Plot 1: Current density vs voltage
ax1 = plt.subplot(2, 2, 1)
V_plot = np.linspace(min(V_applied_list), max(V_applied_list), 100)
j_CL_plot = child_langmuir_current(V_plot, d_gap)

ax1.plot(V_plot, j_CL_plot, 'k-', linewidth=2, label='Child-Langmuir (analytical)')
ax1.plot(V_array[valid_points], j_measured_array[valid_points], 'ro',
         markersize=10, label='PIC simulation')
ax1.set_xlabel('Applied Voltage [V]', fontsize=12)
ax1.set_ylabel('Current Density [A/m²]', fontsize=12)
ax1.set_title('Current Density vs Voltage', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Power law check (log-log)
ax2 = plt.subplot(2, 2, 2)
if np.sum(valid_points) >= 3:
    ax2.loglog(V_array[valid_points], j_measured_array[valid_points], 'ro',
               markersize=10, label='Measured')

    # Fitted line
    V_fit = np.linspace(min(V_array), max(V_array), 100)
    j_fit = np.exp(coeffs[1]) * V_fit**coeffs[0]
    ax2.loglog(V_fit, j_fit, 'r--', linewidth=2,
               label=f'Fit: j ∝ V^{n_fitted:.2f}')

    # Theoretical line
    ax2.loglog(V_plot, j_CL_plot, 'k-', linewidth=2,
               label='Theory: j ∝ V^1.5')

    ax2.set_xlabel('Voltage [V]', fontsize=12)
    ax2.set_ylabel('Current Density [A/m²]', fontsize=12)
    ax2.set_title('Power Law Validation (log-log)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)

# Plot 3: Time evolution of particle count (V=30V)
ax3 = plt.subplot(2, 2, 3)
for V in [10.0, 20.0, 30.0, 40.0]:
    res = results[V]
    ax3.plot(res['time_ns'], res['n_active'], label=f'{V:.0f} V', linewidth=2)

ax3.set_xlabel('Time [ns]', fontsize=12)
ax3.set_ylabel('Active Particles', fontsize=12)
ax3.set_title('Particle Count Evolution', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(title='V_applied', fontsize=10)

# Plot 4: Temperature evolution (V=30V)
ax4 = plt.subplot(2, 2, 4)
for V in [10.0, 20.0, 30.0, 40.0]:
    res = results[V]
    ax4.plot(res['time_ns'], res['T_e_eV'], label=f'{V:.0f} V', linewidth=2)

ax4.axhline(T_e_eV, color='black', linestyle='--', linewidth=2, label='Target T_e')
ax4.set_xlabel('Time [ns]', fontsize=12)
ax4.set_ylabel('Electron Temperature [eV]', fontsize=12)
ax4.set_title('Temperature Evolution', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(title='V_applied', fontsize=10)

plt.tight_layout()
plt.savefig('child_langmuir_benchmark.png', dpi=300, bbox_inches='tight')
print("Saved: child_langmuir_benchmark.png")
print()

print("=" * 70)
print("Child-Langmuir benchmark complete!")
print("=" * 70)
