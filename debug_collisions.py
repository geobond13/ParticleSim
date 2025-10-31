"""Debug script for collision algorithm."""
import numpy as np
from intakesim.constants import SPECIES, kB
from intakesim.mesh import Mesh1D, index_particles_to_cells
from intakesim.particles import sample_maxwellian_velocity
from intakesim.dsmc.collisions import compute_majorant_frequency

# Simple test case
n_particles = 100
length = 1.0
n_cells = 10

# Create mesh
mesh = Mesh1D(length=length, n_cells=n_cells)

print(f"Mesh properties:")
print(f"  n_cells: {mesh.n_cells}")
print(f"  dx: {mesh.dx} m")
print(f"  cross_section: {mesh.cross_section} m^2")
print(f"  cell_volumes: {mesh.cell_volumes}")

# Create particles
x = np.random.rand(n_particles, 3) * length
x[:, 1:] = 0.0

v = sample_maxwellian_velocity(300.0, SPECIES['N2'].mass, n_particles)

species_id = np.zeros(n_particles, dtype=np.int32)
active = np.ones(n_particles, dtype=np.bool_)

# Index to cells
cell_particles, cell_counts = index_particles_to_cells(
    x[:, 0], active, mesh.n_cells, mesh.dx, max_per_cell=1000
)

print(f"\nParticle distribution:")
print(f"  Total particles: {n_particles}")
for i in range(n_cells):
    print(f"  Cell {i}: {cell_counts[i]} particles")

# Compute majorant frequency
mass_array = np.array([SPECIES['N2'].mass], dtype=np.float64)
d_ref_array = np.array([SPECIES['N2'].diameter], dtype=np.float64)
omega_array = np.array([SPECIES['N2'].omega], dtype=np.float64)

nu_majorant = compute_majorant_frequency(
    n_cells, mesh.cell_volumes, cell_counts, cell_particles, 1000,
    x, v, species_id, active, d_ref_array, omega_array
)

print(f"\nMajorant frequencies [Hz]:")
for i in range(n_cells):
    print(f"  Cell {i}: {nu_majorant[i]:.3e} Hz")

# Expected collisions in one timestep
dt = 1e-6  # 1 microsecond
print(f"\nExpected collisions (dt = {dt} s):")
for i in range(n_cells):
    N_expected = nu_majorant[i] * dt
    print(f"  Cell {i}: {N_expected:.6f}")

total_expected = np.sum(nu_majorant * dt)
print(f"\nTotal expected collisions per timestep: {total_expected:.6f}")
print(f"Collision rate per particle per second: {np.sum(nu_majorant) / n_particles:.3e} Hz")

# Compare to theoretical collision frequency
v_mean = np.sqrt(8 * kB * 300 / (np.pi * SPECIES['N2'].mass))
print(f"\nTheoretical values:")
print(f"  Mean thermal velocity: {v_mean:.1f} m/s")

# Number density in a typical cell
n_typical_cell = np.mean(cell_counts[cell_counts > 0])
V_cell = mesh.cell_volumes[0]
n_density = n_typical_cell / V_cell
print(f"  Typical cell number density: {n_density:.3e} particles/m^3")

# Theoretical collision frequency
sigma_T = np.pi * SPECIES['N2'].diameter**2
nu_theoretical = n_density * sigma_T * np.sqrt(2) * v_mean
print(f"  Theoretical collision frequency: {nu_theoretical:.3e} Hz")

print(f"\nExpected collisions in 500 timesteps: {500 * total_expected:.1f}")
