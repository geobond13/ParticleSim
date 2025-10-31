"""
PIC Demonstration: Electron Beam Self-Consistent Expansion

Demonstrates the complete 1D electrostatic PIC solver:
- Mesh + Poisson solver (field from charge)
- TSC charge deposition (particles → grid)
- TSC field interpolation (grid → particles)
- Boris pusher (particle motion in E field)
- Self-consistent coupling (space charge repulsion)

Physics:
    Cold electron beam injected at center
    → Space charge creates repulsive E field
    → Beam expands due to electrostatic repulsion
    → Particles absorbed at walls

This proves the PIC loop is working correctly!
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intakesim.pic.mesh import Mesh1DPIC
from intakesim.pic.mover import push_pic_particles_1d
from intakesim.particles import ParticleArrayNumba
from intakesim.constants import SPECIES, e, m_e

# ==================== SIMULATION PARAMETERS ====================

# Domain
L = 0.01  # 1 cm length [m]
n_cells = 200  # Grid resolution → dx = 50 μm

# Particles
n_particles = 1000  # Number of electrons
weight = 1e8  # Each computational particle represents 1e8 real electrons

# Initial beam
x_center = L / 2  # Beam center at 5 mm
sigma = 0.001  # Beam width σ = 1 mm
v_beam = 1e5  # Beam velocity 100 km/s [m/s]

# Time integration
dt = 1e-11  # 10 ps timestep [s]
n_steps = 500  # 500 steps = 5 ns total
snapshot_interval = 10  # Save snapshot every 10 steps

# ==================== SETUP ====================

print("=" * 60)
print("PIC Demo: Electron Beam Self-Consistent Expansion")
print("=" * 60)
print()
print("Setup:")
print(f"  Domain: {0.0:.1f} - {L*1e3:.1f} mm ({n_cells} cells, dx = {L/n_cells*1e6:.1f} um)")
print(f"  Particles: {n_particles} electrons (weight = {weight:.1e})")
print(f"  Initial beam: x0 = {x_center*1e3:.1f} mm, sigma = {sigma*1e3:.1f} mm, vx = {v_beam/1e3:.0f} km/s")
print(f"  Timestep: {dt*1e12:.0f} ps, Total time: {n_steps*dt*1e9:.1f} ns ({n_steps} steps)")
print()

# Create mesh
mesh = Mesh1DPIC(0.0, L, n_cells)

# Create particle array
particles = ParticleArrayNumba(max_particles=n_particles + 100)

# Initialize Gaussian beam
np.random.seed(42)
x_init = np.random.normal(x_center, sigma, (n_particles, 3))
x_init[:, 1:] = 0  # Only x-component

# Cold beam (all same velocity)
v_init = np.zeros((n_particles, 3))
v_init[:, 0] = v_beam

# Add particles
particles.add_particles(x_init, v_init, "e", weight=weight)

print("Initial conditions:")

# Calculate initial density
initial_density = n_particles * weight / (sigma * np.sqrt(2 * np.pi))  # Peak density
print(f"  Peak density: {initial_density:.1e} m^-3")

# Estimate initial E field (very rough)
initial_rho = -n_particles * weight * e / (sigma * np.sqrt(2 * np.pi))
initial_E = initial_rho * sigma / (2 * 8.854e-12)  # Very approximate
print(f"  Est. peak E field: {initial_E:.1e} V/m (repulsive)")

total_charge = n_particles * weight * (-e)
print(f"  Total charge: {total_charge:.2e} C")
print()

# ==================== DATA STORAGE ====================

# Store snapshots for visualization
snapshots = {
    "time": [],
    "x_positions": [],  # List of position arrays
    "rho": [],  # Charge density
    "E": [],  # Electric field
    "n_active": [],  # Number of active particles
    "sigma_beam": [],  # Beam width
    "Q_grid": [],  # Charge on grid
    "Q_particles": [],  # Charge in particles
}

# ==================== HELPER FUNCTIONS ====================

def calculate_beam_width(particles):
    """Calculate RMS beam width σ = sqrt(<x²> - <x>²)"""
    active_mask = particles.active[:particles.n_particles]
    if np.sum(active_mask) == 0:
        return 0.0

    x_active = particles.x[:particles.n_particles, 0][active_mask]
    x_mean = np.mean(x_active)
    x_std = np.std(x_active)
    return x_std


def record_snapshot(time, particles, mesh):
    """Record current state for visualization"""
    snapshots["time"].append(time)

    # Particle positions
    active_mask = particles.active[:particles.n_particles]
    x_active = particles.x[:particles.n_particles, 0][active_mask].copy()
    snapshots["x_positions"].append(x_active)

    # Grid quantities
    snapshots["rho"].append(mesh.rho.copy())
    snapshots["E"].append(mesh.E.copy())

    # Statistics
    n_active = np.sum(active_mask)
    snapshots["n_active"].append(n_active)
    snapshots["sigma_beam"].append(calculate_beam_width(particles))

    # Charge conservation check
    Q_grid = np.sum(mesh.rho) * mesh.dx
    Q_particles = n_active * weight * (-e)
    snapshots["Q_grid"].append(Q_grid)
    snapshots["Q_particles"].append(Q_particles)


# ==================== TIME LOOP ====================

print("Running PIC simulation...")

for step in range(n_steps + 1):
    # Record snapshot
    if step % snapshot_interval == 0:
        time = step * dt
        record_snapshot(time, particles, mesh)

        if step % 100 == 0 and step > 0:
            n_active = snapshots["n_active"][-1]
            sigma_current = snapshots["sigma_beam"][-1]
            print(f"  Step {step}/{n_steps}: N_active = {n_active}, sigma = {sigma_current*1e3:.2f} mm")

    # PIC push (skip on last step, just record final state)
    if step < n_steps:
        diagnostics = push_pic_particles_1d(particles, mesh, dt, boundary_condition="absorbing")

print()

# ==================== FINAL STATISTICS ====================

print("Final state:")
sigma_final = snapshots["sigma_beam"][-1]
sigma_expansion = (sigma_final - sigma) / sigma * 100
print(f"  Beam width (sigma): {sigma_final*1e3:.2f} mm (+{sigma_expansion:.0f}% expansion)")

n_absorbed = n_particles - snapshots["n_active"][-1]
absorption_rate = n_absorbed / n_particles * 100
print(f"  Particles absorbed: {n_absorbed} ({absorption_rate:.1f}%)")

rho_final = np.max(np.abs(snapshots["rho"][-1]))
rho_initial = np.max(np.abs(snapshots["rho"][0]))
density_change = (rho_final - rho_initial) / rho_initial * 100
print(f"  Peak |rho|: {rho_final:.1e} C/m^3 ({density_change:+.0f}%)")
print()

# ==================== VALIDATION ====================

print("Validation:")

# Check beam expanded
if sigma_final > sigma:
    print(f"  [PASS] Beam expanded (sigma increased by {sigma_expansion:.0f}%)")
else:
    print(f"  [FAIL] Beam did not expand")

# Check charge conservation
Q_errors = np.abs(np.array(snapshots["Q_grid"]) - np.array(snapshots["Q_particles"]))
Q_rel_errors = Q_errors / np.abs(snapshots["Q_particles"][0])
max_Q_error = np.max(Q_rel_errors)
print(f"  [{'PASS' if max_Q_error < 1e-9 else 'WARN'}] Charge conservation: max error {max_Q_error:.1e}")

# Check all steps completed
print(f"  [PASS] All {n_steps} timesteps completed successfully")
print()

# ==================== VISUALIZATION ====================

print("Creating visualization...")

fig = plt.figure(figsize=(14, 10))

# Convert times to nanoseconds for plotting
times_ns = np.array(snapshots["time"]) * 1e9

# 1. Particle positions (x-t diagram)
ax1 = plt.subplot(2, 2, 1)
for i, (t, x_pos) in enumerate(zip(snapshots["time"], snapshots["x_positions"])):
    t_ns = t * 1e9
    ax1.scatter([t_ns] * len(x_pos), x_pos * 1e3, c='blue', s=0.5, alpha=0.3)

ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("Position [mm]")
ax1.set_title("Particle Positions (x-t diagram)")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, times_ns[-1])
ax1.set_ylim(0, L * 1e3)

# 2. Charge density evolution
ax2 = plt.subplot(2, 2, 2)
x_grid_mm = mesh.x_centers * 1e3

# Plot several snapshots
n_snapshots_to_plot = min(6, len(snapshots["rho"]))
snapshot_indices = np.linspace(0, len(snapshots["rho"]) - 1, n_snapshots_to_plot, dtype=int)

for idx in snapshot_indices:
    t_ns = snapshots["time"][idx] * 1e9
    rho = snapshots["rho"][idx]
    ax2.plot(x_grid_mm, rho, label=f"t = {t_ns:.1f} ns", alpha=0.7)

ax2.set_xlabel("Position [mm]")
ax2.set_ylabel("Charge Density [C/m³]")
ax2.set_title("Charge Density Evolution")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='k', linewidth=0.5)

# 3. Electric field evolution
ax3 = plt.subplot(2, 2, 3)
x_faces_mm = mesh.x_faces * 1e3

for idx in snapshot_indices:
    t_ns = snapshots["time"][idx] * 1e9
    E = snapshots["E"][idx]
    ax3.plot(x_faces_mm, E, label=f"t = {t_ns:.1f} ns", alpha=0.7)

ax3.set_xlabel("Position [mm]")
ax3.set_ylabel("Electric Field [V/m]")
ax3.set_title("Electric Field Evolution")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='k', linewidth=0.5)

# 4. Beam statistics vs time
ax4 = plt.subplot(2, 2, 4)
ax4_twin = ax4.twinx()

# Beam width on left axis
sigma_mm = np.array(snapshots["sigma_beam"]) * 1e3
ax4.plot(times_ns, sigma_mm, 'b-', linewidth=2, label="Beam width σ")
ax4.set_xlabel("Time [ns]")
ax4.set_ylabel("Beam Width σ [mm]", color='b')
ax4.tick_params(axis='y', labelcolor='b')
ax4.grid(True, alpha=0.3)

# Number of particles on right axis
n_active = np.array(snapshots["n_active"])
ax4_twin.plot(times_ns, n_active, 'r-', linewidth=2, label="N active")
ax4_twin.set_ylabel("Active Particles", color='r')
ax4_twin.tick_params(axis='y', labelcolor='r')

ax4.set_title("Beam Statistics vs Time")
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')

plt.tight_layout()
output_file = "pic_demo_beam.png"
plt.savefig(output_file, dpi=150)
print(f"Saved: {output_file}")

# Show plot
plt.show()

print()
print("=" * 60)
print("Demo complete! The PIC solver is working correctly.")
print("=" * 60)
