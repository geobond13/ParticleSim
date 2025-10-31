"""
Diagnostic script to investigate O2 particle disappearance in Parodi validation.

Run shorter simulation with detailed species tracking.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from validation.validation_framework import ValidationCase, ValidationMetric
from intakesim.particles import ParticleArrayNumba, sample_maxwellian_velocity
from intakesim.mesh import Mesh1D
from intakesim.dsmc.mover import push_particles_ballistic
from intakesim.dsmc.surfaces import cll_reflect_particle, attempt_catalytic_recombination
from intakesim.dsmc.collisions import perform_collisions_1d
from intakesim.geometry.intake import HoneycombIntake, sample_freestream_velocity, apply_attitude_jitter
from intakesim.constants import SPECIES, SPECIES_ID
from intakesim.diagnostics import compute_compression_ratio
import time
import math

print("="*70)
print("O2 PARTICLE TRACKING DIAGNOSTIC")
print("="*70)

# Configuration matching Parodi
altitude_km = 200
rho_atm = 4.2e17  # m^-3
T_atm = 900.0  # K
v_orbital = 7780.0  # m/s
composition = {'O': 0.83, 'N2': 0.14, 'O2': 0.02}

# Geometry
inlet_area = 0.01  # m^2
outlet_area = 0.001  # m^2
channel_diameter = 0.001  # m
L_over_D = 20.0
channel_length = L_over_D * channel_diameter

# Simulation parameters (SHORT for debugging)
n_steps = 500  # Just 500 steps for quick diagnosis
n_particles_per_step = 100  # Use 100 to match Day 3 parameters
dt = 1e-6
max_particles = 50000

# Initialize
buffer_inlet = 0.01
buffer_outlet = 0.01
domain_length = buffer_inlet + channel_length + buffer_outlet
n_cells = 20

mesh = Mesh1D(length=domain_length, n_cells=n_cells, cross_section=inlet_area)
particles = ParticleArrayNumba(max_particles=max_particles)

# CLL parameters
alpha_n = 1.0
alpha_t = 0.9
T_wall = 300.0
v_wall = np.zeros(3, dtype=np.float64)

# Particle injection rates
total_flux = rho_atm * v_orbital * inlet_area
total_sim_flux = n_particles_per_step / dt
particle_weight = total_flux / total_sim_flux

# Species-specific injection
n_inject_O = int(n_particles_per_step * composition['O'])
n_inject_N2 = int(n_particles_per_step * composition['N2'])
n_inject_O2 = int(n_particles_per_step * composition['O2'])

print(f"\nParticles per step:")
print(f"  O:  {n_inject_O}")
print(f"  N2: {n_inject_N2}")
print(f"  O2: {n_inject_O2}")
print(f"\nExpected O2 particles after {n_steps} steps: {n_inject_O2 * n_steps}")

# Get species masses
mass_O = SPECIES['O'].mass
mass_N2 = SPECIES['N2'].mass
mass_O2 = SPECIES['O2'].mass

# VHS collision parameters
n_species_total = len(SPECIES_ID)
mass_array = np.zeros(n_species_total, dtype=np.float64)
d_ref_array = np.zeros(n_species_total, dtype=np.float64)
omega_array = np.zeros(n_species_total, dtype=np.float64)

for sp_name, sp_id in SPECIES_ID.items():
    mass_array[sp_id] = SPECIES[sp_name].mass
    d_ref_array[sp_id] = SPECIES[sp_name].diameter
    omega_array[sp_id] = SPECIES[sp_name].omega

# Cell indexing
max_particles_per_cell = 2000
cell_particles = np.zeros((n_cells, max_particles_per_cell), dtype=np.int32)
cell_counts = np.zeros(n_cells, dtype=np.int32)
cell_edges = mesh.cell_edges
cell_volumes = mesh.cell_volumes

# Catalytic recombination parameters
gamma_0 = 0.02
E_a_over_k = 2000.0
gamma_recomb = gamma_0 * math.exp(-E_a_over_k / T_wall)

print(f"\nCatalytic recombination gamma: {gamma_recomb:.6e}")

# Diagnostics
z_inlet = buffer_inlet
z_outlet = buffer_inlet + channel_length

# TRACKING COUNTERS
o2_injected = 0
o2_recombined = 0
o2_deleted = 0
o2_species_changes = []  # Track when particles change species

print(f"\n{'='*70}")
print("Running simulation with O2 tracking...")
print(f"{'='*70}")

t_start = time.time()

for step in range(n_steps):
    if step % 100 == 0:
        # Count O2 particles currently in system
        o2_count = 0
        for i in range(particles.n_particles):
            if particles.active[i] and particles.species_id[i] == SPECIES_ID['O2']:
                o2_count += 1

        print(f"Step {step}/{n_steps}: O2 particles in system = {o2_count}")

    # Inject particles
    if particles.n_particles + n_particles_per_step < max_particles:
        # Inject O
        if n_inject_O > 0:
            v_inject_O = sample_freestream_velocity(v_orbital, T_atm, mass_O, n_inject_O)
            v_inject_O = apply_attitude_jitter(v_inject_O, 7.0)

            x_inject_O = np.zeros((n_inject_O, 3), dtype=np.float64)
            x_inject_O[:, 0] = buffer_inlet
            x_inject_O[:, 1] = (np.random.rand(n_inject_O) - 0.5) * np.sqrt(inlet_area)
            x_inject_O[:, 2] = (np.random.rand(n_inject_O) - 0.5) * np.sqrt(inlet_area)

            particles.add_particles(x_inject_O, v_inject_O, species='O', weight=particle_weight)

        # Inject N2
        if n_inject_N2 > 0:
            v_inject_N2 = sample_freestream_velocity(v_orbital, T_atm, mass_N2, n_inject_N2)
            v_inject_N2 = apply_attitude_jitter(v_inject_N2, 7.0)

            x_inject_N2 = np.zeros((n_inject_N2, 3), dtype=np.float64)
            x_inject_N2[:, 0] = buffer_inlet
            x_inject_N2[:, 1] = (np.random.rand(n_inject_N2) - 0.5) * np.sqrt(inlet_area)
            x_inject_N2[:, 2] = (np.random.rand(n_inject_N2) - 0.5) * np.sqrt(inlet_area)

            particles.add_particles(x_inject_N2, v_inject_N2, species='N2', weight=particle_weight)

        # Inject O2
        if n_inject_O2 > 0:
            v_inject_O2 = sample_freestream_velocity(v_orbital, T_atm, mass_O2, n_inject_O2)
            v_inject_O2 = apply_attitude_jitter(v_inject_O2, 7.0)

            x_inject_O2 = np.zeros((n_inject_O2, 3), dtype=np.float64)
            x_inject_O2[:, 0] = buffer_inlet
            x_inject_O2[:, 1] = (np.random.rand(n_inject_O2) - 0.5) * np.sqrt(inlet_area)
            x_inject_O2[:, 2] = (np.random.rand(n_inject_O2) - 0.5) * np.sqrt(inlet_area)

            particles.add_particles(x_inject_O2, v_inject_O2, species='O2', weight=particle_weight)
            o2_injected += n_inject_O2

    # Ballistic motion
    push_particles_ballistic(
        particles.x, particles.v, particles.active, dt, particles.n_particles
    )

    # VHS collisions
    cell_counts[:] = 0
    for i in range(particles.n_particles):
        if not particles.active[i]:
            continue

        z = particles.x[i, 0]
        if z < 0 or z >= domain_length:
            continue

        cell_idx = int((z / domain_length) * n_cells)
        if cell_idx < 0:
            cell_idx = 0
        if cell_idx >= n_cells:
            cell_idx = n_cells - 1

        if cell_counts[cell_idx] < max_particles_per_cell:
            cell_particles[cell_idx, cell_counts[cell_idx]] = i
            cell_counts[cell_idx] += 1

    n_collisions = perform_collisions_1d(
        particles.x, particles.v, particles.species_id, particles.active,
        particles.weight, particles.n_particles,
        cell_edges, cell_volumes,
        cell_particles, cell_counts, max_particles_per_cell,
        mass_array, d_ref_array, omega_array,
        dt
    )

    # Boundary conditions (outflow)
    deleted_this_step = 0
    for i in range(particles.n_particles):
        if not particles.active[i]:
            continue
        z = particles.x[i, 0]
        if z < 0 or z > domain_length:
            # Track O2 deletions
            if particles.species_id[i] == SPECIES_ID['O2']:
                o2_deleted += 1
                deleted_this_step += 1
            particles.active[i] = False

    # Wall collisions with catalytic recombination
    r_inlet = np.sqrt(inlet_area / np.pi)
    r_outlet = np.sqrt(outlet_area / np.pi)

    for i in range(particles.n_particles):
        if not particles.active[i]:
            continue

        z = particles.x[i, 0]
        if z_inlet <= z <= z_outlet:
            z_rel = (z - z_inlet) / (z_outlet - z_inlet)
            r_local = r_inlet * (1 - z_rel) + r_outlet * z_rel

            r_perp = np.sqrt(particles.x[i, 1]**2 + particles.x[i, 2]**2)

            if r_perp > r_local:
                species_idx = particles.species_id[i]
                if species_idx == 0:  # O
                    mass = mass_O
                elif species_idx == 1:  # N2
                    mass = mass_N2
                else:  # O2
                    mass = mass_O2

                # Attempt catalytic recombination for O
                if species_idx == 0:
                    recombined, v_product, new_species_id = attempt_catalytic_recombination(
                        particles.v[i], mass_O, T_wall, species_idx, gamma_recomb
                    )

                    if recombined:
                        particles.v[i] = v_product
                        particles.species_id[i] = new_species_id
                        o2_recombined += 1
                        o2_species_changes.append((step, i, 0, new_species_id))

                        if r_perp > 0:
                            scale = r_local * 0.95 / r_perp
                            particles.x[i, 1] *= scale
                            particles.x[i, 2] *= scale
                        continue

                # Normal CLL reflection
                v_reflected = cll_reflect_particle(
                    particles.v[i], v_wall, mass, T_wall, alpha_n, alpha_t
                )
                particles.v[i] = v_reflected

                if r_perp > 0:
                    scale = r_local * 0.95 / r_perp
                    particles.x[i, 1] *= scale
                    particles.x[i, 2] *= scale

t_elapsed = time.time() - t_start

# Final count
o2_final = 0
o2_at_inlet = 0
o2_at_outlet = 0
for i in range(particles.n_particles):
    if particles.active[i] and particles.species_id[i] == SPECIES_ID['O2']:
        o2_final += 1
        z = particles.x[i, 0]
        if z_inlet <= z < z_inlet + 0.005:
            o2_at_inlet += 1
        if z_outlet - 0.005 <= z < z_outlet:
            o2_at_outlet += 1

print(f"\n{'='*70}")
print("DIAGNOSTIC RESULTS")
print(f"{'='*70}")
print(f"Simulation time: {t_elapsed:.1f} s")
print(f"\nO2 Particle Budget:")
print(f"  Injected:     {o2_injected}")
print(f"  Recombined:   {o2_recombined}")
print(f"  Deleted:      {o2_deleted}")
print(f"  Final in system: {o2_final}")
print(f"\nO2 at measurement regions:")
print(f"  At inlet (z={z_inlet:.3f} to {z_inlet+0.005:.3f}): {o2_at_inlet}")
print(f"  At outlet (z={z_outlet-0.005:.3f} to {z_outlet:.3f}): {o2_at_outlet}")
print(f"\nExpected steady-state O2: ~{int(n_inject_O2 * 500 * 0.5)} (50% transmission)")
print(f"Actual O2 in system: {o2_final}")
print(f"\nBalance check: {o2_injected} injected + {o2_recombined} recombined - {o2_deleted} deleted = {o2_injected + o2_recombined - o2_deleted}")
print(f"Should equal final: {o2_final}")
print(f"Discrepancy: {(o2_injected + o2_recombined - o2_deleted) - o2_final}")

if len(o2_species_changes) > 0:
    print(f"\nSpecies changes (O -> O2 via recombination): {len(o2_species_changes)} events")
    print("First few events:")
    for step, idx, old_sp, new_sp in o2_species_changes[:5]:
        print(f"  Step {step}: particle {idx} changed from species {old_sp} to {new_sp}")
