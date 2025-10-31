"""
Tests for PIC particle mover (Boris push, TSC weighting, boundary conditions)
"""

import pytest
import numpy as np
from intakesim.particles import ParticleArrayNumba
from intakesim.pic.mesh import Mesh1DPIC
from intakesim.pic.mover import (
    apply_reflecting_bc_1d,
    tsc_weight_1d,
    get_tsc_stencil_1d,
)


class TestReflectingBoundaryCondition:
    """Test suite for reflecting boundary conditions"""

    def test_left_wall_reflection(self):
        """Test that particles reflect off left wall with reversed velocity"""
        # Setup
        n_particles = 10
        x = np.zeros((n_particles, 3))
        v = np.zeros((n_particles, 3))
        active = np.ones(n_particles, dtype=bool)

        # Particle beyond left wall
        x[0, 0] = -0.001  # 1 mm beyond x_min = 0
        v[0, 0] = -1000.0  # Moving left

        x_min, x_max = 0.0, 0.01

        # Apply reflection
        n_reflected = apply_reflecting_bc_1d(x, v, active, x_min, x_max, n_particles)

        # Verify reflection
        assert n_reflected == 1, "Should reflect 1 particle"
        assert x[0, 0] == 0.001, f"Position should be mirrored: {x[0, 0]}"
        assert v[0, 0] == 1000.0, f"Velocity should reverse: {v[0, 0]}"

    def test_right_wall_reflection(self):
        """Test that particles reflect off right wall with reversed velocity"""
        # Setup
        n_particles = 10
        x = np.zeros((n_particles, 3))
        v = np.zeros((n_particles, 3))
        active = np.ones(n_particles, dtype=bool)

        # Particle beyond right wall
        x[0, 0] = 0.012  # 2 mm beyond x_max = 0.01
        v[0, 0] = 1000.0  # Moving right

        x_min, x_max = 0.0, 0.01

        # Apply reflection
        n_reflected = apply_reflecting_bc_1d(x, v, active, x_min, x_max, n_particles)

        # Verify reflection
        assert n_reflected == 1, "Should reflect 1 particle"
        assert x[0, 0] == 0.008, f"Position should be mirrored: {x[0, 0]}"
        assert v[0, 0] == -1000.0, f"Velocity should reverse: {v[0, 0]}"

    def test_energy_conservation(self):
        """Test that reflection conserves kinetic energy"""
        # Setup
        n_particles = 100
        x = np.random.uniform(-0.002, 0.012, (n_particles, 3))
        x[:, 1:] = 0  # Only x-component
        v = np.random.uniform(-5000, 5000, (n_particles, 3))
        v[:, 1:] = 0  # Only x-component
        active = np.ones(n_particles, dtype=bool)

        # Calculate initial energy
        E_initial = 0.5 * np.sum(v[active] ** 2)

        x_min, x_max = 0.0, 0.01

        # Apply reflection
        apply_reflecting_bc_1d(x, v, active, x_min, x_max, n_particles)

        # Calculate final energy
        E_final = 0.5 * np.sum(v[active] ** 2)

        # Verify energy conservation
        relative_error = abs(E_final - E_initial) / E_initial
        assert (
            relative_error < 1e-12
        ), f"Energy should be conserved, error = {relative_error}"

    def test_no_reflection_inside_domain(self):
        """Test that particles inside domain are not reflected"""
        # Setup
        n_particles = 50
        x = np.random.uniform(0.001, 0.009, (n_particles, 3))
        x[:, 1:] = 0
        v = np.random.uniform(-1000, 1000, (n_particles, 3))
        v[:, 1:] = 0
        active = np.ones(n_particles, dtype=bool)

        # Store initial state
        x_initial = x.copy()
        v_initial = v.copy()

        x_min, x_max = 0.0, 0.01

        # Apply reflection
        n_reflected = apply_reflecting_bc_1d(x, v, active, x_min, x_max, n_particles)

        # Verify no reflection
        assert n_reflected == 0, "Should not reflect particles inside domain"
        np.testing.assert_array_equal(x, x_initial, err_msg="Positions should not change")
        np.testing.assert_array_equal(v, v_initial, err_msg="Velocities should not change")

    def test_inactive_particles_ignored(self):
        """Test that inactive particles are not affected"""
        # Setup
        n_particles = 10
        x = np.zeros((n_particles, 3))
        v = np.zeros((n_particles, 3))
        active = np.zeros(n_particles, dtype=bool)

        # Inactive particle beyond wall
        x[0, 0] = -0.001
        v[0, 0] = -1000.0
        active[0] = False

        x_min, x_max = 0.0, 0.01

        # Apply reflection
        n_reflected = apply_reflecting_bc_1d(x, v, active, x_min, x_max, n_particles)

        # Verify no reflection
        assert n_reflected == 0, "Should not reflect inactive particles"
        assert x[0, 0] == -0.001, "Inactive particle position should not change"
        assert v[0, 0] == -1000.0, "Inactive particle velocity should not change"

    def test_multiple_reflections(self):
        """Test that multiple particles can reflect in one call"""
        # Setup
        n_particles = 5
        x = np.array([[-0.001], [0.011], [0.005], [-0.002], [0.012]])
        x = np.hstack([x, np.zeros((n_particles, 2))])
        v = np.array([[-1000.0], [1000.0], [500.0], [-2000.0], [2000.0]])
        v = np.hstack([v, np.zeros((n_particles, 2))])
        active = np.ones(n_particles, dtype=bool)

        x_min, x_max = 0.0, 0.01

        # Apply reflection
        n_reflected = apply_reflecting_bc_1d(x, v, active, x_min, x_max, n_particles)

        # Verify 4 reflections (indices 0, 1, 3, 4)
        assert n_reflected == 4, f"Should reflect 4 particles, got {n_reflected}"
        np.testing.assert_almost_equal(x[0, 0], 0.001, decimal=10, err_msg="Left wall reflection position")
        np.testing.assert_almost_equal(v[0, 0], 1000.0, decimal=10, err_msg="Left wall reflection velocity")
        np.testing.assert_almost_equal(x[1, 0], 0.009, decimal=10, err_msg="Right wall reflection position")
        np.testing.assert_almost_equal(v[1, 0], -1000.0, decimal=10, err_msg="Right wall reflection velocity")
        np.testing.assert_almost_equal(x[2, 0], 0.005, decimal=10, err_msg="Interior particle unchanged")
        np.testing.assert_almost_equal(v[2, 0], 500.0, decimal=10, err_msg="Interior velocity unchanged")


class TestTSCWeighting:
    """Test suite for Triangular-Shaped Cloud (TSC) weighting"""

    def test_tsc_weight_central_cell(self):
        """Test TSC weight at cell center"""
        dx = 0.001
        x_cell = 0.005
        x_particle = 0.005  # Exactly at center

        weight = tsc_weight_1d(x_particle, x_cell, dx)

        # At center, distance = 0, weight = 0.75 - 0 = 0.75
        assert abs(weight - 0.75) < 1e-12, f"Central weight should be 0.75, got {weight}"

    def test_tsc_weight_falls_off(self):
        """Test TSC weight decreases with distance"""
        dx = 0.001
        x_cell = 0.005

        # Test at different distances
        w_center = tsc_weight_1d(0.005, x_cell, dx)  # distance = 0
        w_near = tsc_weight_1d(0.0055, x_cell, dx)  # distance = 0.5 dx
        w_far = tsc_weight_1d(0.006, x_cell, dx)  # distance = 1.0 dx

        assert w_center > w_near > w_far, "Weight should decrease with distance"

    def test_tsc_weight_beyond_1_5dx(self):
        """Test TSC weight is zero beyond 1.5 * dx"""
        dx = 0.001
        x_cell = 0.005
        x_particle = 0.0065  # 1.5 dx away

        weight = tsc_weight_1d(x_particle, x_cell, dx)

        assert weight < 1e-12, f"Weight should be zero beyond 1.5*dx, got {weight}"

    def test_tsc_stencil_sum_to_one(self):
        """Test that TSC weights sum to 1.0 for any particle position"""
        dx = 0.001
        x_grid = np.arange(0.0005, 0.0105, dx)  # Cell centers
        n_cells = len(x_grid)

        # Test 100 random positions
        np.random.seed(42)
        for _ in range(100):
            x_particle = np.random.uniform(x_grid[0], x_grid[-1])
            indices, weights = get_tsc_stencil_1d(x_particle, x_grid, dx, n_cells)

            weight_sum = np.sum(weights)
            assert (
                abs(weight_sum - 1.0) < 1e-12
            ), f"Weights should sum to 1.0, got {weight_sum}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
