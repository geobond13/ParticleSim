"""
Tests for DSMC Collision Module (VHS Model)

Validates:
- VHS cross-section calculation
- Momentum conservation in collisions
- Energy conservation in collisions
- Thermal equilibration to Maxwell-Boltzmann distribution
- Binary collision algorithm correctness

Week 2 Deliverable - IntakeSIM
"""

import pytest
import numpy as np
from intakesim.dsmc.collisions import (
    vhs_collision_cross_section,
    compute_collision_frequency,
    compute_post_collision_velocity,
    attempt_collision,
    perform_collisions_1d,
)
from intakesim.constants import SPECIES, kB
from intakesim.mesh import Mesh1D, index_particles_to_cells
from intakesim.particles import sample_maxwellian_velocity


class TestVHSCrossSection:
    """Test Variable Hard Sphere cross-section model."""

    def test_vhs_cross_section_decreases_with_velocity(self):
        """VHS cross-section should decrease with increasing relative velocity for ω > 0.5."""
        T_ref = 273.0  # K
        d_ref = 4.17e-10  # N2 diameter [m]
        omega = 0.74  # N2 VHS exponent

        v_low = 100.0  # m/s
        v_high = 1000.0  # m/s

        sigma_low = vhs_collision_cross_section(v_low, T_ref, d_ref, omega)
        sigma_high = vhs_collision_cross_section(v_high, T_ref, d_ref, omega)

        # For ω > 0.5, cross-section decreases with velocity
        assert sigma_low > sigma_high, "VHS cross-section should decrease with velocity"

    def test_vhs_reduces_to_hard_sphere(self):
        """For ω = 0.5, VHS should reduce to constant hard-sphere cross-section."""
        T_ref = 273.0
        d_ref = 4.17e-10
        omega = 0.5  # Hard sphere

        v1 = 100.0
        v2 = 1000.0

        sigma1 = vhs_collision_cross_section(v1, T_ref, d_ref, omega)
        sigma2 = vhs_collision_cross_section(v2, T_ref, d_ref, omega)

        # For ω = 0.5, exponent is 2ω - 1 = 0, so σ = π*d^2 (constant)
        # Allow small numerical error
        assert abs(sigma1 - sigma2) / sigma1 < 0.1, "Hard sphere should have constant cross-section"

        # Check against analytical value
        sigma_expected = np.pi * d_ref**2
        assert abs(sigma1 - sigma_expected) / sigma_expected < 0.1

    def test_vhs_cross_section_positive(self):
        """VHS cross-section should always be positive."""
        T_ref = 273.0
        d_ref = 4.17e-10
        omega = 0.74

        for v_rel in [10.0, 100.0, 1000.0, 5000.0]:
            sigma = vhs_collision_cross_section(v_rel, T_ref, d_ref, omega)
            assert sigma > 0, f"Cross-section must be positive (v_rel={v_rel})"

    def test_collision_frequency_scales_with_density(self):
        """Collision frequency should scale linearly with density."""
        d_ref = 4.17e-10
        omega = 0.74
        v_mean = 400.0  # m/s
        T = 300.0  # K

        n1 = 1e20  # m^-3
        n2 = 2e20

        nu1 = compute_collision_frequency(n1, d_ref, omega, v_mean, T)
        nu2 = compute_collision_frequency(n2, d_ref, omega, v_mean, T)

        # Frequency should double with density
        assert abs(nu2 / nu1 - 2.0) < 0.01, "Collision frequency should scale linearly with density"


class TestPostCollisionVelocity:
    """Test post-collision velocity calculation."""

    def test_momentum_conservation(self):
        """Post-collision velocities should conserve momentum."""
        m1 = SPECIES['N2'].mass
        m2 = SPECIES['O'].mass

        # Random initial velocities
        v1 = np.array([500.0, 100.0, -200.0], dtype=np.float64)
        v2 = np.array([-300.0, 400.0, 150.0], dtype=np.float64)

        # Compute post-collision velocities
        v1_post, v2_post = compute_post_collision_velocity(v1, v2, m1, m2, 4e-10, 0.74)

        # Check momentum conservation
        p_initial = m1 * v1 + m2 * v2
        p_final = m1 * v1_post + m2 * v2_post

        np.testing.assert_allclose(p_final, p_initial, rtol=1e-10,
                                    err_msg="Momentum not conserved in collision")

    def test_energy_conservation(self):
        """Post-collision velocities should conserve kinetic energy."""
        m1 = SPECIES['N2'].mass
        m2 = SPECIES['O'].mass

        v1 = np.array([500.0, 100.0, -200.0], dtype=np.float64)
        v2 = np.array([-300.0, 400.0, 150.0], dtype=np.float64)

        v1_post, v2_post = compute_post_collision_velocity(v1, v2, m1, m2, 4e-10, 0.74)

        # Check energy conservation
        KE_initial = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)
        KE_final = 0.5 * m1 * np.dot(v1_post, v1_post) + 0.5 * m2 * np.dot(v2_post, v2_post)

        assert abs(KE_final - KE_initial) / KE_initial < 1e-10, "Kinetic energy not conserved"

    def test_scattering_is_isotropic(self):
        """Over many collisions, scattering should be isotropic (uniform on sphere)."""
        m1 = SPECIES['N2'].mass
        m2 = SPECIES['N2'].mass

        v1 = np.array([500.0, 0.0, 0.0], dtype=np.float64)
        v2 = np.array([-500.0, 0.0, 0.0], dtype=np.float64)

        # Perform many collisions and collect scattering angles
        n_samples = 1000
        cos_theta_samples = []

        for _ in range(n_samples):
            v1_post, v2_post = compute_post_collision_velocity(v1, v2, m1, m2, 4e-10, 0.74)

            # Compute scattering angle in COM frame
            v_rel_initial = v1 - v2
            v_rel_final = v1_post - v2_post

            # Cosine of scattering angle
            cos_theta = np.dot(v_rel_initial, v_rel_final) / (
                np.linalg.norm(v_rel_initial) * np.linalg.norm(v_rel_final)
            )
            cos_theta_samples.append(cos_theta)

        # For isotropic scattering, cos(θ) should be uniformly distributed in [-1, 1]
        # Mean of cos(θ) should be ~0
        mean_cos_theta = np.mean(cos_theta_samples)
        assert abs(mean_cos_theta) < 0.1, "Scattering should be isotropic (mean cos(θ) ≈ 0)"

        # Standard deviation of cos(θ) for uniform distribution on [-1,1] is 1/sqrt(3) ≈ 0.577
        std_cos_theta = np.std(cos_theta_samples)
        assert abs(std_cos_theta - 1.0 / np.sqrt(3)) < 0.1, "Scattering angle distribution incorrect"


class TestCollisionConservation:
    """Test that collision algorithm conserves total momentum and energy."""

    def test_single_collision_conserves_momentum_and_energy(self):
        """A single collision should conserve momentum and energy."""
        n_particles = 100
        n_species = 1  # Single species (N2)

        # Create particle arrays
        v = np.random.randn(n_particles, 3).astype(np.float64) * 400.0  # ~400 m/s
        species_id = np.zeros(n_particles, dtype=np.int32)  # All N2
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)

        # Species properties
        mass_array = np.array([SPECIES['N2'].mass], dtype=np.float64)
        d_ref_array = np.array([SPECIES['N2'].diameter], dtype=np.float64)
        omega_array = np.array([SPECIES['N2'].omega], dtype=np.float64)

        # Compute initial momentum and energy
        p_initial = mass_array[0] * np.sum(v, axis=0)
        KE_initial = 0.5 * mass_array[0] * np.sum(v**2)

        # Perform a single collision manually
        idx1 = 0
        idx2 = 1
        v1_post, v2_post = compute_post_collision_velocity(
            v[idx1], v[idx2], mass_array[0], mass_array[0],
            d_ref_array[0], omega_array[0]
        )
        v[idx1] = v1_post
        v[idx2] = v2_post

        # Compute final momentum and energy
        p_final = mass_array[0] * np.sum(v, axis=0)
        KE_final = 0.5 * mass_array[0] * np.sum(v**2)

        # Check conservation
        np.testing.assert_allclose(p_final, p_initial, rtol=1e-10,
                                    err_msg="Total momentum not conserved")
        assert abs(KE_final - KE_initial) / KE_initial < 1e-10, "Total kinetic energy not conserved"


class TestThermalEquilibration:
    """Test that collisions drive system to thermal equilibrium."""

    def test_thermal_equilibration_single_species(self):
        """
        Two thermal distributions at different temperatures should equilibrate.

        This is a critical validation test for the collision module.
        """
        n_particles = 5000
        n_steps = 500
        dt = 1e-6  # 1 microsecond

        # Create mesh (1D, 1 meter domain with realistic cross-section)
        length = 1.0
        cross_section = 0.01  # 0.01 m^2 = 10 cm × 10 cm cross-section
        mesh = Mesh1D(length=length, n_cells=10, cross_section=cross_section)

        # Create particles: half hot, half cold
        n_hot = n_particles // 2
        n_cold = n_particles - n_hot

        T_hot = 600.0  # K
        T_cold = 200.0  # K

        # Sample velocities
        v_hot = sample_maxwellian_velocity(T_hot, SPECIES['N2'].mass, n_hot)
        v_cold = sample_maxwellian_velocity(T_cold, SPECIES['N2'].mass, n_cold)

        v = np.vstack([v_hot, v_cold])

        # Random positions
        x = np.random.rand(n_particles, 3) * length
        x[:, 1:] = 0.0  # Keep in 1D

        # All N2, all active
        species_id = np.zeros(n_particles, dtype=np.int32)
        active = np.ones(n_particles, dtype=np.bool_)

        # Set particle weights to represent realistic number density
        # Target: n = 1e20 m^-3 (typical for VLEO at ~200 km)
        # Volume = length × cross_section = 1.0 × 0.01 = 0.01 m^3
        # Real molecules = n × V = 1e20 × 0.01 = 1e18
        # Weight = real_molecules / n_particles = 1e18 / 5000 = 2e14
        target_density = 1e20  # m^-3
        volume_total = length * cross_section
        real_molecules_total = target_density * volume_total
        particle_weight = real_molecules_total / n_particles

        weight = np.full(n_particles, particle_weight, dtype=np.float64)

        print(f"\n  Particle weight: {particle_weight:.3e} molecules/particle")
        print(f"  Effective number density: {target_density:.3e} m^-3")

        # Species arrays
        mass_array = np.array([SPECIES['N2'].mass], dtype=np.float64)
        d_ref_array = np.array([SPECIES['N2'].diameter], dtype=np.float64)
        omega_array = np.array([SPECIES['N2'].omega], dtype=np.float64)

        # Initial temperatures
        T_initial_hot = np.mean(v_hot**2) * SPECIES['N2'].mass / (3 * kB)
        T_initial_cold = np.mean(v_cold**2) * SPECIES['N2'].mass / (3 * kB)
        T_initial_avg = np.mean(v**2) * SPECIES['N2'].mass / (3 * kB)

        print(f"\nInitial: T_hot={T_initial_hot:.1f} K, T_cold={T_initial_cold:.1f} K, "
              f"T_avg={T_initial_avg:.1f} K")

        # Time integration with collisions
        for step in range(n_steps):
            # Index particles to cells
            cell_particles, cell_counts = index_particles_to_cells(
                x[:, 0], active, mesh.n_cells, mesh.dx, max_per_cell=1000
            )

            # Perform collisions
            n_collisions = perform_collisions_1d(
                x, v, species_id, active, weight, n_particles,
                mesh.cell_edges, mesh.cell_volumes,
                cell_particles, cell_counts, 1000,
                mass_array, d_ref_array, omega_array, dt
            )

            if step % 100 == 0:
                # Compute current temperatures
                T_hot_current = np.mean(v[:n_hot]**2) * SPECIES['N2'].mass / (3 * kB)
                T_cold_current = np.mean(v[n_hot:]**2) * SPECIES['N2'].mass / (3 * kB)
                T_avg = np.mean(v**2) * SPECIES['N2'].mass / (3 * kB)
                print(f"Step {step}: T_hot={T_hot_current:.1f} K, T_cold={T_cold_current:.1f} K, "
                      f"T_avg={T_avg:.1f} K, collisions={n_collisions}")

        # Final temperatures
        T_final_hot = np.mean(v[:n_hot]**2) * SPECIES['N2'].mass / (3 * kB)
        T_final_cold = np.mean(v[n_hot:]**2) * SPECIES['N2'].mass / (3 * kB)
        T_final_avg = np.mean(v**2) * SPECIES['N2'].mass / (3 * kB)

        print(f"\nFinal: T_hot={T_final_hot:.1f} K, T_cold={T_final_cold:.1f} K, "
              f"T_avg={T_final_avg:.1f} K")
        print(f"Expected (from energy conservation): {T_initial_avg:.1f} K")

        # Check that temperatures have equilibrated
        assert abs(T_final_hot - T_final_cold) < 50.0, \
            "Hot and cold populations should equilibrate within 50 K"

        # Check that average temperature is conserved (energy conservation)
        assert abs(T_final_avg - T_initial_avg) / T_initial_avg < 0.05, \
            "Average temperature should be conserved (energy conservation)"

    @pytest.mark.slow
    def test_maxwell_boltzmann_distribution(self):
        """
        After many collisions, velocity distribution should be Maxwell-Boltzmann.

        This test validates that the collision algorithm produces the correct
        equilibrium distribution.
        """
        n_particles = 10000
        n_steps = 1000
        dt = 1e-6

        length = 1.0
        mesh = Mesh1D(length=length, n_cells=20)

        # Start with non-Maxwellian distribution (e.g., all particles moving right)
        v = np.zeros((n_particles, 3), dtype=np.float64)
        v[:, 0] = 500.0  # All moving at 500 m/s in x-direction

        x = np.random.rand(n_particles, 3) * length
        x[:, 1:] = 0.0

        species_id = np.zeros(n_particles, dtype=np.int32)
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)

        mass_array = np.array([SPECIES['N2'].mass], dtype=np.float64)
        d_ref_array = np.array([SPECIES['N2'].diameter], dtype=np.float64)
        omega_array = np.array([SPECIES['N2'].omega], dtype=np.float64)

        # Time integration
        for step in range(n_steps):
            cell_particles, cell_counts = index_particles_to_cells(
                x[:, 0], active, mesh.n_cells, mesh.dx, max_per_cell=1000
            )

            perform_collisions_1d(
                x, v, species_id, active, weight, n_particles,
                mesh.cell_edges, mesh.cell_volumes,
                cell_particles, cell_counts, 1000,
                mass_array, d_ref_array, omega_array, dt
            )

        # Compute temperature from final velocity distribution
        T_final = np.mean(v**2) * SPECIES['N2'].mass / (3 * kB)

        # Expected temperature from initial kinetic energy
        KE_initial = 0.5 * SPECIES['N2'].mass * n_particles * 500.0**2
        T_expected = KE_initial / (n_particles * 1.5 * kB)

        print(f"\nFinal temperature: {T_final:.1f} K (expected: {T_expected:.1f} K)")

        # Check temperature matches expected
        assert abs(T_final - T_expected) / T_expected < 0.1

        # Check that velocity distribution is isotropic (all components have same variance)
        var_x = np.var(v[:, 0])
        var_y = np.var(v[:, 1])
        var_z = np.var(v[:, 2])

        print(f"Velocity variance: x={var_x:.1e}, y={var_y:.1e}, z={var_z:.1e}")

        # Isotropic: variance should be equal in all directions
        assert abs(var_x - var_y) / var_x < 0.15, "Velocity distribution should be isotropic"
        assert abs(var_x - var_z) / var_x < 0.15, "Velocity distribution should be isotropic"

        # Check that mean velocity is near zero (COM frame)
        v_mean = np.mean(v, axis=0)
        v_thermal = np.sqrt(kB * T_final / SPECIES['N2'].mass)

        print(f"Mean velocity: {v_mean} (should be ~0)")
        print(f"Thermal velocity: {v_thermal:.1f} m/s")

        # Mean velocity should be much less than thermal velocity
        assert np.linalg.norm(v_mean) < 0.1 * v_thermal, "Mean velocity should be near zero"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
