"""
Unit tests for DSMC ballistic motion and boundary conditions
"""

import pytest
import numpy as np
from intakesim.dsmc.mover import (
    push_particles_ballistic,
    apply_periodic_bc,
    apply_outflow_bc,
    apply_reflecting_bc,
    compute_number_density_1d,
    compute_mean_velocity_1d,
)


class TestBallisticMotion:
    """Test ballistic particle motion."""

    def test_ballistic_trajectory_straight_line(self):
        """Particles move in straight lines with constant velocity."""
        # Setup
        n_particles = 100
        dt = 1e-6  # 1 microsecond
        n_steps = 10000  # 10 ms total

        x = np.zeros((n_particles, 3), dtype=np.float64)
        v = np.ones((n_particles, 3), dtype=np.float64) * 100  # 100 m/s
        active = np.ones(n_particles, dtype=np.bool_)

        # Initial position
        x_initial = x.copy()

        # Integrate
        for _ in range(n_steps):
            push_particles_ballistic(x, v, active, dt, n_particles)

        # Expected displacement
        expected_displacement = 100 * dt * n_steps  # v * t = 1.0 meter

        # Check all particles moved same distance
        displacement = np.linalg.norm(x - x_initial, axis=1)
        np.testing.assert_array_almost_equal(
            displacement,
            np.full(n_particles, expected_displacement),
            decimal=6
        )

    def test_ballistic_inactive_particles_dont_move(self):
        """Inactive particles should not move."""
        n_particles = 10
        dt = 1e-3

        x = np.zeros((n_particles, 3), dtype=np.float64)
        v = np.ones((n_particles, 3), dtype=np.float64) * 100
        active = np.zeros(n_particles, dtype=np.bool_)
        active[0] = True  # Only first particle active

        x_initial = x.copy()

        # Push
        push_particles_ballistic(x, v, active, dt, n_particles)

        # Only first particle should have moved
        assert not np.allclose(x[0], x_initial[0])
        np.testing.assert_array_equal(x[1:], x_initial[1:])

    def test_ballistic_momentum_conserved(self):
        """Momentum is conserved during ballistic motion."""
        n_particles = 1000
        dt = 1e-6
        n_steps = 100

        # Random initial conditions
        x = np.random.rand(n_particles, 3)
        v = np.random.randn(n_particles, 3) * 100
        active = np.ones(n_particles, dtype=np.bool_)

        # Initial total momentum
        p_initial = np.sum(v, axis=0)

        # Integrate
        for _ in range(n_steps):
            push_particles_ballistic(x, v, active, dt, n_particles)

        # Final momentum (velocity unchanged in ballistic motion)
        p_final = np.sum(v, axis=0)

        np.testing.assert_array_almost_equal(p_initial, p_final, decimal=10)


class TestPeriodicBoundaries:
    """Test periodic boundary conditions."""

    def test_periodic_wrapping_right(self):
        """Particles exiting right boundary wrap to left."""
        n_particles = 3
        length = 1.0

        x = np.array([[1.1, 0, 0], [0.5, 0, 0], [0.9, 0, 0]], dtype=np.float64)
        active = np.ones(n_particles, dtype=np.bool_)

        apply_periodic_bc(x, active, length, n_particles)

        # First particle should wrap
        assert abs(x[0, 0] - 0.1) < 1e-10
        assert abs(x[1, 0] - 0.5) < 1e-10  # Unchanged
        assert abs(x[2, 0] - 0.9) < 1e-10  # Unchanged

    def test_periodic_wrapping_left(self):
        """Particles exiting left boundary wrap to right."""
        n_particles = 2
        length = 1.0

        x = np.array([[-0.1, 0, 0], [0.5, 0, 0]], dtype=np.float64)
        active = np.ones(n_particles, dtype=np.bool_)

        apply_periodic_bc(x, active, length, n_particles)

        assert abs(x[0, 0] - 0.9) < 1e-10
        assert abs(x[1, 0] - 0.5) < 1e-10

    def test_periodic_multiple_wraps(self):
        """Test wrapping with displacement > domain length."""
        n_particles = 1
        length = 1.0

        x = np.array([[2.3, 0, 0]], dtype=np.float64)
        active = np.ones(n_particles, dtype=np.bool_)

        # Apply multiple times to handle large displacement
        while x[0, 0] >= length:
            apply_periodic_bc(x, active, length, n_particles)

        assert 0 <= x[0, 0] < length


class TestOutflowBoundaries:
    """Test outflow boundary conditions."""

    def test_outflow_removes_particles(self):
        """Particles leaving domain are deactivated."""
        n_particles = 4
        length = 1.0

        x = np.array([[1.1, 0, 0], [-0.1, 0, 0], [0.5, 0, 0], [0.99, 0, 0]], dtype=np.float64)
        active = np.ones(n_particles, dtype=np.bool_)

        n_removed = apply_outflow_bc(x, active, length, n_particles)

        assert n_removed == 2
        assert active[0] == False  # Past right boundary
        assert active[1] == False  # Past left boundary
        assert active[2] == True   # Inside domain
        assert active[3] == True   # Inside domain

    def test_outflow_all_particles_remain(self):
        """Particles inside domain remain active."""
        n_particles = 10
        length = 1.0

        x = np.random.rand(n_particles, 3) * length  # All inside [0, length)
        active = np.ones(n_particles, dtype=np.bool_)

        n_removed = apply_outflow_bc(x, active, length, n_particles)

        assert n_removed == 0
        assert np.all(active)


class TestReflectingBoundaries:
    """Test specular reflecting boundary conditions."""

    def test_reflecting_right_boundary(self):
        """Particle reflects from right boundary."""
        n_particles = 1
        length = 1.0

        x = np.array([[1.1, 0, 0]], dtype=np.float64)
        v = np.array([[100, 0, 0]], dtype=np.float64)
        active = np.ones(n_particles, dtype=np.bool_)

        apply_reflecting_bc(x, v, active, length, n_particles)

        # Position mirrored: 2*length - 1.1 = 0.9
        assert abs(x[0, 0] - 0.9) < 1e-10

        # Velocity reversed
        assert abs(v[0, 0] - (-100)) < 1e-10

    def test_reflecting_left_boundary(self):
        """Particle reflects from left boundary."""
        n_particles = 1
        length = 1.0

        x = np.array([[-0.1, 0, 0]], dtype=np.float64)
        v = np.array([[-100, 0, 0]], dtype=np.float64)
        active = np.ones(n_particles, dtype=np.bool_)

        apply_reflecting_bc(x, v, active, length, n_particles)

        # Position mirrored: -(-0.1) = 0.1
        assert abs(x[0, 0] - 0.1) < 1e-10

        # Velocity reversed
        assert abs(v[0, 0] - 100) < 1e-10

    def test_reflecting_energy_conserved(self):
        """Kinetic energy conserved during reflection."""
        n_particles = 1
        length = 1.0

        x = np.array([[1.1, 0.2, 0.3]], dtype=np.float64)
        v = np.array([[100, 50, -30]], dtype=np.float64)
        active = np.ones(n_particles, dtype=np.bool_)

        # Initial kinetic energy
        KE_initial = 0.5 * np.sum(v**2)

        apply_reflecting_bc(x, v, active, length, n_particles)

        # Final kinetic energy
        KE_final = 0.5 * np.sum(v**2)

        assert abs(KE_final - KE_initial) < 1e-10


class TestDiagnostics:
    """Test diagnostic functions."""

    def test_number_density_uniform(self):
        """Uniform particle distribution gives uniform density."""
        n_particles = 10000
        length = 1.0
        n_cells = 10

        # Uniform distribution
        x = np.random.rand(n_particles, 3) * length
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)

        cell_edges = np.linspace(0, length, n_cells + 1)

        density = compute_number_density_1d(x, active, weight, cell_edges, n_particles)

        # Expected density: n_particles / length
        expected_density = n_particles / length

        # Each cell should have approximately the same density
        # Allow 10% variation due to random sampling
        np.testing.assert_allclose(density, expected_density, rtol=0.1)

    def test_mean_velocity_stationary(self):
        """Stationary gas has zero mean velocity."""
        n_particles = 1000
        length = 1.0
        n_cells = 5

        x = np.random.rand(n_particles, 3) * length

        # Zero net velocity (thermal fluctuations cancel)
        v = np.random.randn(n_particles, 3) * 100

        # Make exactly zero mean
        v -= np.mean(v, axis=0)

        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)
        cell_edges = np.linspace(0, length, n_cells + 1)

        v_mean = compute_mean_velocity_1d(x, v, active, weight, cell_edges, n_particles)

        # Mean velocity should be close to zero in all cells
        np.testing.assert_allclose(v_mean, 0, atol=10)  # Within 10 m/s for random fluctuations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
