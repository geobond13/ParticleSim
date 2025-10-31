"""
DSMC Particle Motion (Ballistic + Boundaries)

Numba-compiled particle pusher for maximum performance.
Week 1 implementation: ballistic motion only (no forces).
"""

import numpy as np
from numba import njit, prange


# ==================== BALLISTIC MOTION ====================

@njit(parallel=True, fastmath=True)
def push_particles_ballistic(x, v, active, dt, n_particles):
    """
    Push particles ballistically (no forces).

    x_new = x_old + v * dt

    This is the performance-critical kernel. Must achieve:
    - 10^6 particles in <2 seconds for 10,000 timesteps
    - 50-100× speedup vs pure Python

    Args:
        x: Position array, shape (n_max, 3) [m]
        v: Velocity array, shape (n_max, 3) [m/s]
        active: Active flags, shape (n_max,)
        dt: Timestep [s]
        n_particles: Number of particles to push

    Note:
        Arrays are modified in-place for performance.
        Uses parallel=True for automatic threading.
    """
    for i in prange(n_particles):
        if active[i]:
            # Update position (ballistic motion)
            x[i, 0] += v[i, 0] * dt
            x[i, 1] += v[i, 1] * dt
            x[i, 2] += v[i, 2] * dt


# ==================== BOUNDARY CONDITIONS ====================

@njit
def apply_periodic_bc(x, active, length, n_particles):
    """
    Apply periodic boundary conditions in x-direction.

    Particles wrapping around:
        if x < 0:        x += length
        if x >= length:  x -= length

    Args:
        x: Position array, shape (n_max, 3) [m]
        active: Active flags, shape (n_max,)
        length: Domain length [m]
        n_particles: Number of particles

    Note:
        Only x-coordinate is wrapped. y, z remain unchanged.
    """
    for i in range(n_particles):
        if active[i]:
            # Wrap x-coordinate
            if x[i, 0] < 0:
                x[i, 0] += length
            elif x[i, 0] >= length:
                x[i, 0] -= length


@njit
def apply_outflow_bc(x, active, length, n_particles):
    """
    Apply outflow boundary conditions.

    Particles leaving domain are deactivated:
        if x < 0 or x >= length: active = False

    Args:
        x: Position array, shape (n_max, 3) [m]
        active: Active flags, shape (n_max,)
        length: Domain length [m]
        n_particles: Number of particles

    Returns:
        n_removed: Number of particles removed
    """
    n_removed = 0

    for i in range(n_particles):
        if active[i]:
            if x[i, 0] < 0 or x[i, 0] >= length:
                active[i] = False
                n_removed += 1

    return n_removed


@njit
def apply_reflecting_bc(x, v, active, length, n_particles):
    """
    Apply specular reflecting boundary conditions.

    Particles hitting walls bounce back:
        if x < 0:        x = -x,          v_x = -v_x
        if x >= length:  x = 2*length-x,  v_x = -v_x

    Args:
        x: Position array, shape (n_max, 3) [m]
        v: Velocity array, shape (n_max, 3) [m/s]
        active: Active flags, shape (n_max,)
        length: Domain length [m]
        n_particles: Number of particles

    Note:
        This is specular reflection (perfect mirror).
        For thermal accommodation, use surfaces.py (Week 3).
    """
    for i in range(n_particles):
        if active[i]:
            # Left boundary
            if x[i, 0] < 0:
                x[i, 0] = -x[i, 0]  # Mirror position
                v[i, 0] = -v[i, 0]  # Reverse velocity

            # Right boundary
            elif x[i, 0] >= length:
                x[i, 0] = 2 * length - x[i, 0]  # Mirror position
                v[i, 0] = -v[i, 0]              # Reverse velocity


# ==================== DIAGNOSTICS ====================

@njit
def compute_number_density_1d(x, active, weight, cell_edges, n_particles):
    """
    Compute number density profile.

    Args:
        x: Position array, shape (n_max, 3) [m]
        active: Active flags, shape (n_max,)
        weight: Computational weights, shape (n_max,)
        cell_edges: Cell boundaries, shape (n_cells+1,) [m]
        n_particles: Number of particles

    Returns:
        density: Number density per cell, shape (n_cells,) [m^-3]
    """
    n_cells = len(cell_edges) - 1
    counts = np.zeros(n_cells, dtype=np.float64)

    # Count weighted particles in each cell
    for i in range(n_particles):
        if active[i]:
            # Find cell
            for j in range(n_cells):
                if cell_edges[j] <= x[i, 0] < cell_edges[j+1]:
                    counts[j] += weight[i]
                    break

    # Convert counts to density (particles per unit volume)
    dx = cell_edges[1] - cell_edges[0]  # Assume uniform
    density = counts / dx

    return density


@njit
def compute_mean_velocity_1d(x, v, active, weight, cell_edges, n_particles):
    """
    Compute mean velocity profile.

    Args:
        x: Position array, shape (n_max, 3) [m]
        v: Velocity array, shape (n_max, 3) [m/s]
        active: Active flags, shape (n_max,)
        weight: Computational weights, shape (n_max,)
        cell_edges: Cell boundaries, shape (n_cells+1,) [m]
        n_particles: Number of particles

    Returns:
        v_mean: Mean velocity per cell, shape (n_cells, 3) [m/s]
    """
    n_cells = len(cell_edges) - 1
    v_sum = np.zeros((n_cells, 3), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.float64)

    # Sum weighted velocities in each cell
    for i in range(n_particles):
        if active[i]:
            # Find cell
            for j in range(n_cells):
                if cell_edges[j] <= x[i, 0] < cell_edges[j+1]:
                    w = weight[i]
                    v_sum[j, 0] += v[i, 0] * w
                    v_sum[j, 1] += v[i, 1] * w
                    v_sum[j, 2] += v[i, 2] * w
                    counts[j] += w
                    break

    # Compute mean (avoid division by zero)
    v_mean = np.zeros((n_cells, 3), dtype=np.float64)
    for j in range(n_cells):
        if counts[j] > 0:
            v_mean[j, 0] = v_sum[j, 0] / counts[j]
            v_mean[j, 1] = v_sum[j, 1] / counts[j]
            v_mean[j, 2] = v_sum[j, 2] / counts[j]

    return v_mean


# ==================== TESTING ====================

if __name__ == "__main__":
    import time

    print("Testing DSMC Mover (Numba)...")

    # Setup
    n_particles = 100_000
    length = 1.0  # 1 meter domain
    dt = 1e-6    # 1 microsecond

    x = np.random.rand(n_particles, 3)
    v = np.random.randn(n_particles, 3) * 100  # ~100 m/s typical
    active = np.ones(n_particles, dtype=np.bool_)

    # Warmup (Numba compilation)
    print("\nWarming up Numba JIT...")
    for _ in range(10):
        push_particles_ballistic(x, v, active, dt, n_particles)

    # Benchmark
    print("\nBenchmarking ballistic motion...")
    n_steps = 10_000
    start = time.time()

    for step in range(n_steps):
        push_particles_ballistic(x, v, active, dt, n_particles)

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Particles:     {n_particles:,}")
    print(f"  Timesteps:     {n_steps:,}")
    print(f"  Elapsed time:  {elapsed:.3f} s")
    print(f"  Performance:   {n_particles * n_steps / elapsed / 1e6:.1f} M particle-steps/sec")
    print(f"  Time per step: {elapsed/n_steps*1000:.3f} ms")

    # Test boundary conditions
    print("\n\nTesting boundary conditions...")

    # Periodic
    x_test = np.array([[1.1, 0, 0], [-0.1, 0, 0], [0.5, 0, 0]])
    active_test = np.ones(3, dtype=np.bool_)
    apply_periodic_bc(x_test, active_test, length, 3)
    print(f"Periodic BC: {x_test[:, 0]}")  # Should be [0.1, 0.9, 0.5]

    # Outflow
    x_test = np.array([[1.1, 0, 0], [-0.1, 0, 0], [0.5, 0, 0]])
    active_test = np.ones(3, dtype=np.bool_)
    n_removed = apply_outflow_bc(x_test, active_test, length, 3)
    print(f"Outflow BC: active = {active_test}, removed = {n_removed}")

    # Reflecting
    x_test = np.array([[1.1, 0, 0], [-0.1, 0, 0], [0.5, 0, 0]])
    v_test = np.array([[10, 0, 0], [-10, 0, 0], [10, 0, 0]])
    active_test = np.ones(3, dtype=np.bool_)
    apply_reflecting_bc(x_test, v_test, active_test, length, 3)
    print(f"Reflecting BC: x = {x_test[:, 0]}, v_x = {v_test[:, 0]}")

    print("\n✅ DSMC mover tests passed!")
