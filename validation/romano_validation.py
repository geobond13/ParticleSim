"""
Romano et al. (2021) Diffuse Intake Benchmark

"Intake Design for an Atmospheric Breathing Electric Propulsion System"
https://doi.org/10.2514/6.2021-3381

Validation target: Diffuse intake compression efficiency
- h = 150 km: eta_c = 0.458 (acceptable: 0.3-0.6)
- Altitude sweep: 150-250 km

Focus: Diffuse surfaces only (sigma_n = 1.0, sigma_t = 0.9)
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
from intakesim.dsmc.surfaces import cll_reflect_particle
from intakesim.dsmc.collisions import perform_collisions_1d  # VHS collision integration
from intakesim.geometry.intake import HoneycombIntake, sample_freestream_velocity, apply_attitude_jitter, sample_channel_positions
from intakesim.constants import SPECIES, SPECIES_ID
from intakesim.diagnostics import compute_compression_ratio
import time


class RomanoDiffuseValidation(ValidationCase):
    """
    Validate IntakeSIM compression efficiency against Romano et al. (2021) diffuse intake.

    Compression efficiency: eta_c = (n_outlet / n_inlet) / (A_inlet / A_outlet)
                                  = CR_measured / CR_geometric

    Reference: eta_c = 0.458 at 150 km with diffuse surfaces
    """

    def __init__(self, altitude_km: float = 150.0):
        super().__init__(
            name=f"Romano Diffuse Intake (h={altitude_km:.0f} km)",
            description="Validate compression efficiency against Romano et al. (2021) diffuse benchmark"
        )

        # Atmospheric data for different altitudes (from NRLMSISE-00 model)
        atm_data = {
            150: {'rho': 2.1e18, 'T': 600, 'composition': {'N2': 0.78, 'O2': 0.21, 'O': 0.01}},
            175: {'rho': 3.8e17, 'T': 700, 'composition': {'N2': 0.60, 'O2': 0.15, 'O': 0.25}},
            200: {'rho': 4.2e17, 'T': 900, 'composition': {'N2': 0.14, 'O2': 0.02, 'O': 0.83}},
            225: {'rho': 1.8e17, 'T': 1000, 'composition': {'N2': 0.05, 'O2': 0.005, 'O': 0.945}},
            250: {'rho': 7.5e16, 'T': 1100, 'composition': {'N2': 0.02, 'O2': 0.001, 'O': 0.979}},
        }

        # Get atmospheric properties
        atm = atm_data.get(altitude_km)
        if atm is None:
            raise ValueError(f"Altitude {altitude_km} km not in database. Use: {list(atm_data.keys())}")

        # Configuration matching Romano paper
        self.config = {
            'altitude_km': altitude_km,
            'rho_atm': atm['rho'],  # m^-3
            'T_atm': atm['T'],  # K
            'v_orbital': 7780.0,  # m/s
            'composition': atm['composition'],
            # Diffuse surface (CLL)
            'sigma_n': 1.0,  # Fully diffuse normal
            'sigma_t': 0.9,  # Nearly diffuse tangential
            'T_wall': 300.0,  # K
            # Geometry (similar to Romano)
            'inlet_area': 0.01,  # m^2
            'outlet_area': 0.001,  # m^2
            'channel_diameter': 0.001,  # m (1 mm)
            'L_over_D': 20.0,
        }

        # Reference values from Romano Table 8
        self.reference_data = {
            'eta_c_diffuse': {
                150: 0.458,
                175: 0.412,
                200: 0.381,
                225: 0.358,
                250: 0.342,
            }.get(altitude_km, 0.458),  # Default to 150 km
            'eta_c_min': 0.30,  # Acceptable range
            'eta_c_max': 0.60,
        }

    def load_reference_data(self):
        """Load Romano et al. reference data."""
        return self.reference_data

    def run_simulation(self, n_steps: int = 2000, n_particles_per_step: int = 50, verbose: bool = True):
        """
        Run IntakeSIM with Romano diffuse configuration.

        Args:
            n_steps: Number of simulation steps
            n_particles_per_step: Particles injected per step
            verbose: Print progress

        Returns:
            Dictionary with simulation results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running Romano et al. (2021) Diffuse Intake Validation")
            print(f"{'='*70}\n")
            print(f"Configuration:")
            print(f"  Altitude: {self.config['altitude_km']:.0f} km")
            print(f"  Density: {self.config['rho_atm']:.2e} m^-3")
            print(f"  Temperature: {self.config['T_atm']:.0f} K")
            print(f"  Orbital velocity: {self.config['v_orbital']:.0f} m/s")
            print(f"  Surface: Diffuse (sigma_n={self.config['sigma_n']}, sigma_t={self.config['sigma_t']})")
            print(f"\n  Timesteps: {n_steps}")
            print(f"  Particles/step: {n_particles_per_step}")

        # Extract configuration
        rho_atm = self.config['rho_atm']
        T_atm = self.config['T_atm']
        v_orbital = self.config['v_orbital']
        composition = self.config['composition']

        # Intake geometry
        inlet_area = self.config['inlet_area']
        outlet_area = self.config['outlet_area']
        channel_diameter = self.config['channel_diameter']
        channel_length = self.config['L_over_D'] * channel_diameter

        # Phase II: Enable multi-channel honeycomb geometry for realistic wall collisions
        intake = HoneycombIntake(inlet_area, outlet_area, channel_length, channel_diameter,
                                 use_multichannel=True)

        if verbose:
            print(f"\n  Intake geometry:")
            print(f"    Inlet area: {inlet_area} m^2")
            print(f"    Outlet area: {outlet_area} m^2")
            print(f"    Channel length: {channel_length*1e3:.1f} mm")
            print(f"    Channel diameter: {channel_diameter*1e3:.1f} mm")
            print(f"    L/D ratio: {self.config['L_over_D']:.1f}")
            print(f"    Geometric compression: {intake.geometric_compression:.1f}")
            print(f"    Clausing factor: {intake.clausing_factor:.4f}")
            print(f"    Expected eta_c (diffuse): {self.reference_data['eta_c_diffuse']:.3f}")

        # Simulation domain
        buffer_inlet = 0.01
        buffer_outlet = 0.01
        domain_length = buffer_inlet + channel_length + buffer_outlet

        n_cells = 20
        dt = 1e-6  # 1 microsecond
        max_particles = 50000

        # Initialize
        mesh = Mesh1D(length=domain_length, n_cells=n_cells, cross_section=inlet_area)
        particles = ParticleArrayNumba(max_particles=max_particles)

        # CLL parameters
        alpha_n = self.config['sigma_n']
        alpha_t = self.config['sigma_t']
        T_wall = self.config['T_wall']
        v_wall = np.zeros(3, dtype=np.float64)

        # Particle injection rates for each species
        total_flux = rho_atm * v_orbital * inlet_area  # particles/s
        total_sim_flux = n_particles_per_step / dt
        particle_weight = total_flux / total_sim_flux

        # Species-specific injection
        species_list = list(composition.keys())
        n_inject = {sp: int(n_particles_per_step * composition[sp]) for sp in species_list}

        # Get species masses
        species_mass = {sp: SPECIES[sp].mass for sp in species_list}

        # VHS collision parameters (arrays indexed by species_id)
        # Create arrays for all species in SPECIES_ID order
        n_species_total = len(SPECIES_ID)
        mass_array = np.zeros(n_species_total, dtype=np.float64)
        d_ref_array = np.zeros(n_species_total, dtype=np.float64)
        omega_array = np.zeros(n_species_total, dtype=np.float64)

        for sp_name, sp_id in SPECIES_ID.items():
            mass_array[sp_id] = SPECIES[sp_name].mass
            d_ref_array[sp_id] = SPECIES[sp_name].diameter
            omega_array[sp_id] = SPECIES[sp_name].omega

        # Cell indexing for collisions
        max_particles_per_cell = 2000
        cell_particles = np.zeros((n_cells, max_particles_per_cell), dtype=np.int32)
        cell_counts = np.zeros(n_cells, dtype=np.int32)
        cell_edges = mesh.cell_edges
        cell_volumes = mesh.cell_volumes

        # Diagnostics
        z_inlet = buffer_inlet
        z_outlet = buffer_inlet + channel_length

        # Storage for CR measurements (steady-state only)
        CR_list = []

        # Storage for collision diagnostics
        collision_counts = []

        # INVESTIGATION PHASE 1: Particle loss diagnostics
        particles_lost_to_channels = 0  # Cumulative count of particles lost due to channel_id < 0
        particles_lost_to_domain = 0     # Cumulative count of particles lost due to domain boundaries

        # WEEK 4 FIX: Channel transition diagnostics
        particles_transitioned = 0       # Count of particles recovered via nearest-channel transition

        # Main loop
        t_start = time.time()

        if verbose:
            print(f"\n{'='*70}")
            print("Running simulation...")
            print(f"{'='*70}")

        for step in range(n_steps):
            # Progress reporting
            if verbose and step % 400 == 0:
                # Count particle distribution
                n_upstream = 0
                n_in_channel = 0
                n_downstream = 0
                for i in range(particles.n_particles):
                    if particles.active[i]:
                        z = particles.x[i, 0]
                        if z < z_inlet:
                            n_upstream += 1
                        elif z <= z_outlet:
                            n_in_channel += 1
                        else:
                            n_downstream += 1

                print(f"  Step {step}/{n_steps} ({step/n_steps*100:.0f}%)")
                print(f"    Particles: total={particles.n_particles}, " +
                      f"upstream={n_upstream}, in_channel={n_in_channel}, downstream={n_downstream}")
                print(f"    Lost to channels: {particles_lost_to_channels}, " +
                      f"Lost to domain: {particles_lost_to_domain}")
                print(f"    Channel transitions (recovered): {particles_transitioned}")

            # Inject particles for each species
            if particles.n_particles + n_particles_per_step < max_particles:
                for species_name in species_list:
                    n_sp = n_inject[species_name]
                    if n_sp > 0:
                        mass_sp = species_mass[species_name]

                        # Sample freestream velocity
                        v_inject = sample_freestream_velocity(v_orbital, T_atm, mass_sp, n_sp)
                        v_inject = apply_attitude_jitter(v_inject, 7.0)

                        # Week 3: Channel-only injection (eliminates ~9% waste from bounding box sampling)
                        x_inject = np.zeros((n_sp, 3), dtype=np.float64)
                        x_inject[:, 0] = buffer_inlet
                        y_z = sample_channel_positions(n_sp, intake.channel_centers, intake.channel_radius)
                        x_inject[:, 1] = y_z[:, 0]  # y-positions from channel sampling
                        x_inject[:, 2] = y_z[:, 1]  # z-positions from channel sampling

                        particles.add_particles(x_inject, v_inject, species=species_name, weight=particle_weight)

            # Ballistic motion
            push_particles_ballistic(
                particles.x, particles.v, particles.active, dt, particles.n_particles
            )

            # VHS collisions (Day 1 integration)
            # Index particles into cells
            cell_counts[:] = 0
            for i in range(particles.n_particles):
                if not particles.active[i]:
                    continue

                z = particles.x[i, 0]
                if z < 0 or z >= domain_length:
                    continue

                # Find cell index
                cell_idx = int((z / domain_length) * n_cells)
                if cell_idx < 0:
                    cell_idx = 0
                if cell_idx >= n_cells:
                    cell_idx = n_cells - 1

                # Add to cell if not full
                if cell_counts[cell_idx] < max_particles_per_cell:
                    cell_particles[cell_idx, cell_counts[cell_idx]] = i
                    cell_counts[cell_idx] += 1

            # Perform collisions
            n_collisions = perform_collisions_1d(
                particles.x, particles.v, particles.species_id, particles.active,
                particles.weight, particles.n_particles,
                cell_edges, cell_volumes,
                cell_particles, cell_counts, max_particles_per_cell,
                mass_array, d_ref_array, omega_array,
                dt
            )

            if step % 400 == 0 and verbose:
                collision_counts.append(n_collisions)

            # Boundary conditions (outflow)
            for i in range(particles.n_particles):
                if not particles.active[i]:
                    continue
                z = particles.x[i, 0]
                if z < 0 or z > domain_length:
                    particles.active[i] = False

            # Wall collisions (CLL reflection) - MULTI-CHANNEL HONEYCOMB GEOMETRY (Phase II)
            # Now uses per-channel cylindrical walls with generalized CLL reflection
            # Each channel has its own radial wall normal for physically correct reflections
            for i in range(particles.n_particles):
                if not particles.active[i]:
                    continue

                z = particles.x[i, 0]
                if z_inlet <= z <= z_outlet:
                    # Determine which channel particle is in
                    channel_id = intake.get_channel_id(particles.x[i, 1], particles.x[i, 2])

                    if channel_id < 0:
                        # WEEK 4 FIX: Particle in inter-channel gap
                        # Instead of instant death, find nearest channel and push particle into it
                        nearest_id = intake.get_nearest_channel_id(particles.x[i, 1], particles.x[i, 2])

                        if nearest_id >= 0:
                            # Recover particle by pushing into nearest channel center
                            cy, cz = intake.channel_centers[nearest_id]
                            particles.x[i, 1] = cy
                            particles.x[i, 2] = cz

                            # Treat as wall collision with honeycomb structure
                            wall_normal = intake.get_wall_normal(particles.x[i], nearest_id)
                            species_idx = particles.species_id[i]
                            mass = list(species_mass.values())[species_idx]

                            from intakesim.dsmc.surfaces import cll_reflect_particle_general
                            v_reflected = cll_reflect_particle_general(
                                particles.v[i], v_wall, mass, T_wall, alpha_n, alpha_t, wall_normal
                            )
                            particles.v[i] = v_reflected

                            # Track successful transition
                            particles_transitioned += 1

                            # Update channel_id for subsequent wall collision check
                            channel_id = nearest_id
                        else:
                            # Truly outside intake structure (rare) - deactivate
                            particles.active[i] = False
                            continue

                    # Get radial distance from this channel's centerline
                    r_perp = intake.get_radial_distance(particles.x[i], channel_id)

                    # Check if particle hit channel wall
                    if r_perp > intake.channel_radius:
                        # Get wall normal for this channel (radially outward)
                        wall_normal = intake.get_wall_normal(particles.x[i], channel_id)

                        # Get particle mass
                        species_idx = particles.species_id[i]
                        mass = list(species_mass.values())[species_idx]

                        # Generalized CLL reflection with arbitrary wall normal
                        from intakesim.dsmc.surfaces import cll_reflect_particle_general
                        v_reflected = cll_reflect_particle_general(
                            particles.v[i], v_wall, mass, T_wall, alpha_n, alpha_t, wall_normal
                        )
                        particles.v[i] = v_reflected

                        # Push back inside channel (radial direction)
                        cy, cz = intake.channel_centers[channel_id]
                        dy = particles.x[i, 1] - cy
                        dz = particles.x[i, 2] - cz
                        if r_perp > 0:
                            scale = (intake.channel_radius * 0.95) / r_perp
                            particles.x[i, 1] = cy + dy * scale
                            particles.x[i, 2] = cz + dz * scale

            # INVESTIGATION PHASE 1: Count particles lost in this timestep
            step_lost_channels = 0
            step_lost_domain = 0
            for i in range(particles.n_particles):
                if not particles.active[i]:
                    z = particles.x[i, 0]
                    # Check if lost inside intake region (channel geometry)
                    if z_inlet <= z <= z_outlet:
                        step_lost_channels += 1
                    else:
                        step_lost_domain += 1

            particles_lost_to_channels += step_lost_channels
            particles_lost_to_domain += step_lost_domain

            # Measure compression ratio (steady-state only, every 50 steps)
            # NOTE: eta_c uses DENSITY ratio only, not mass flux
            # Velocity change is inherent in continuity equation (A1*v1 = A2*v2)
            if step > n_steps // 2 and step % 50 == 0:
                # Total particle count (all species)
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

                # Volume normalization
                inlet_volume = 0.005 * inlet_area
                outlet_volume = 0.005 * outlet_area

                n_density_inlet = n_inlet / inlet_volume if inlet_volume > 0 else 0.0
                n_density_outlet = n_outlet / outlet_volume if outlet_volume > 0 else 0.0

                # Density compression ratio (NOT including velocity - that's from continuity)
                CR = n_density_outlet / n_density_inlet if n_density_inlet > 0 else 0.0

                if CR > 0:
                    CR_list.append(CR)

                    # INVESTIGATION PHASE 1: Log CR measurements (every 400 steps)
                    if verbose and step % 400 == 0:
                        print(f"    CR measurement at step {step}:")
                        print(f"      n_density_inlet = {n_density_inlet:.2e} m^-3")
                        print(f"      n_density_outlet = {n_density_outlet:.2e} m^-3")
                        print(f"      CR = {CR:.3f}, CR_mean so far = {np.mean(CR_list):.3f}")

        t_elapsed = time.time() - t_start

        # Compute results
        CR_mean = np.mean(CR_list) if len(CR_list) > 0 else 0.0
        CR_std = np.std(CR_list) if len(CR_list) > 0 else 0.0

        # Steady-state validation (check last 20% of measurements)
        # BUG FIX (Dec 13, 2025): Added convergence validation
        steady_state_reached = True
        CV = 0.0
        if len(CR_list) >= 10:
            recent_CR = CR_list[-max(1, len(CR_list)//5):]  # Last 20%
            CV = np.std(recent_CR) / np.mean(recent_CR) if np.mean(recent_CR) > 0 else 0.0
            if CV > 0.05:
                steady_state_reached = False

        # Compression efficiency
        CR_geometric = intake.geometric_compression
        eta_c = CR_mean / CR_geometric if CR_geometric > 0 else 0.0

        # Collision statistics
        mean_collisions_per_step = np.mean(collision_counts) if len(collision_counts) > 0 else 0.0
        collision_rate_per_particle = mean_collisions_per_step / particles.n_particles if particles.n_particles > 0 else 0.0

        if verbose:
            print(f"\n{'='*70}")
            print("Simulation complete!")
            print(f"{'='*70}")
            print(f"  Total time: {t_elapsed:.1f} s")
            print(f"  Final particle count: {particles.n_particles}")
            print(f"  Steady-state: {'REACHED' if steady_state_reached else 'NOT REACHED'} (CV={CV:.1%})")
            print(f"  VHS collisions: {mean_collisions_per_step:.0f} per step ({collision_rate_per_particle:.4f} per particle)")
            print(f"\n  INVESTIGATION PHASE 1: Particle Loss Summary:")
            print(f"    Lost to channel geometry (channel_id < 0): {particles_lost_to_channels}")
            print(f"    Lost to domain boundaries: {particles_lost_to_domain}")
            print(f"    Total lost: {particles_lost_to_channels + particles_lost_to_domain}")
            print(f"\n  WEEK 4 FIX: Channel Transition Summary:")
            print(f"    Particles recovered (transitioned to nearest channel): {particles_transitioned}")
            recovery_rate = (particles_transitioned / (particles_transitioned + particles_lost_to_channels) * 100
                           if (particles_transitioned + particles_lost_to_channels) > 0 else 0.0)
            print(f"    Recovery rate: {recovery_rate:.1f}% of inter-channel particles")
            print(f"\n  Results:")
            print(f"    CR (measured) = {CR_mean:.2f} +/- {CR_std:.2f}")
            print(f"    CR (geometric) = {CR_geometric:.2f}")
            print(f"    Compression efficiency eta_c = {eta_c:.3f}")
            print(f"    Reference (Romano): eta_c = {self.reference_data['eta_c_diffuse']:.3f}")
            if not steady_state_reached:
                print(f"\n  WARNING: Steady-state not reached! Consider increasing n_steps.")

        self.simulation_data = {
            'CR_measured': CR_mean,
            'CR_std': CR_std,
            'CR_geometric': CR_geometric,
            'eta_c': eta_c,
            'compute_time_s': t_elapsed,
            'n_particles_final': particles.n_particles,
        }

        return self.simulation_data

    def compare_results(self):
        """Compare simulation to Romano reference."""
        # Create validation metric
        self.metrics = [
            ValidationMetric(
                name='Compression Efficiency (eta_c)',
                reference_value=self.reference_data['eta_c_diffuse'],
                simulated_value=self.simulation_data['eta_c'],
                tolerance_percent=30.0,  # ±30% acceptable
                units=''
            ),
        ]

        return super().compare_results()


if __name__ == "__main__":
    # Run Romano validation at 150 km
    # Day 3: Increased simulation length for better statistics
    validation = RomanoDiffuseValidation(altitude_km=150)
    validation.load_reference_data()
    validation.run_simulation(n_steps=5000, n_particles_per_step=100, verbose=True)
    validation.compare_results()
    validation.print_summary()
    validation.save_results_csv('romano_validation_150km_results.csv')
