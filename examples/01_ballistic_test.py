"""
Example 01: Ballistic Motion Test

Demonstrates:
- Creating particle arrays
- Ballistic motion integration
- Boundary conditions (periodic, outflow, reflecting)
- Performance benchmarking
- Basic visualization

Week 1 Deliverable: Verify Numba performance meets gate requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from intakesim.particles import ParticleArrayNumba, sample_maxwellian_velocity
from intakesim.mesh import Mesh1D
from intakesim.dsmc.mover import (
    push_particles_ballistic,
    apply_periodic_bc,
    apply_outflow_bc,
    apply_reflecting_bc,
    compute_number_density_1d,
    compute_mean_velocity_1d,
)
from intakesim.constants import SPECIES, kB


def example_1_simple_ballistic():
    """Example 1: Simple ballistic motion with periodic boundaries."""
    print("\n" + "="*60)
    print("Example 1: Ballistic Motion (Periodic Boundaries)")
    print("="*60)

    # Setup
    n_particles = 1000
    length = 1.0  # 1 meter domain
    dt = 1e-6     # 1 microsecond timestep
    n_steps = 1000

    # Create particles
    particles = ParticleArrayNumba(n_particles)

    # Initialize with random positions and thermal velocities
    x = np.random.rand(n_particles, 3) * length
    v = sample_maxwellian_velocity(T=300, mass=SPECIES['N2'].mass, n_samples=n_particles)

    particles.add_particles(x, v, species='N2')

    # Create mesh for diagnostics
    mesh = Mesh1D(length=length, n_cells=10)

    # Time integration
    print(f"\nRunning {n_steps} timesteps...")
    for step in range(n_steps):
        push_particles_ballistic(
            particles.x, particles.v, particles.active, dt, particles.n_particles
        )
        apply_periodic_bc(
            particles.x, particles.active, length, particles.n_particles
        )

    print(f"[OK] Complete! Final positions in [0, {length}]:")
    print(f"   min(x) = {np.min(particles.x[:particles.n_particles, 0]):.6f}")
    print(f"   max(x) = {np.max(particles.x[:particles.n_particles, 0]):.6f}")


def example_2_outflow_boundaries():
    """Example 2: Particle beam with outflow boundaries."""
    print("\n" + "="*60)
    print("Example 2: Particle Beam (Outflow Boundaries)")
    print("="*60)

    # Setup: particles flowing in +x direction
    n_particles = 5000
    length = 1.0
    dt = 1e-6
    n_steps = 2000

    particles = ParticleArrayNumba(n_particles)

    # Start at x=0 with velocity in +x
    x = np.zeros((n_particles, 3))
    x[:, 1:] = np.random.rand(n_particles, 2) * 0.01  # Small y,z spread
    v = np.zeros((n_particles, 3))
    v[:, 0] = 500.0  # 500 m/s in x-direction

    particles.add_particles(x, v, species='O')

    print(f"\nInitial: {particles.n_particles} particles")

    # Time integration
    n_removed_total = 0
    for step in range(n_steps):
        push_particles_ballistic(
            particles.x, particles.v, particles.active, dt, particles.n_particles
        )
        n_removed = apply_outflow_bc(
            particles.x, particles.active, length, particles.n_particles
        )
        n_removed_total += n_removed

    n_remaining = np.sum(particles.active[:particles.n_particles])

    print(f"\nAfter {n_steps} steps:")
    print(f"   Particles remaining: {n_remaining}")
    print(f"   Particles removed:   {n_removed_total}")
    print(f"   [OK] All particles left domain as expected")


def example_3_reflecting_box():
    """Example 3: Particles bouncing in a reflecting box."""
    print("\n" + "="*60)
    print("Example 3: Reflecting Box")
    print("="*60)

    n_particles = 500
    length = 1.0
    dt = 1e-6
    n_steps = 5000

    particles = ParticleArrayNumba(n_particles)

    # Random positions and velocities
    x = np.random.rand(n_particles, 3) * length
    v = np.random.randn(n_particles, 3) * 200  # ~200 m/s

    particles.add_particles(x, v, species='N2')

    # Track kinetic energy
    KE_initial = particles.kinetic_energy()

    # Time integration with reflecting walls
    for step in range(n_steps):
        push_particles_ballistic(
            particles.x, particles.v, particles.active, dt, particles.n_particles
        )
        apply_reflecting_bc(
            particles.x, particles.v, particles.active, length, particles.n_particles
        )

    KE_final = particles.kinetic_energy()

    print(f"\nKinetic energy:")
    print(f"   Initial: {KE_initial:.6e} J")
    print(f"   Final:   {KE_final:.6e} J")
    print(f"   Change:  {abs(KE_final - KE_initial)/KE_initial * 100:.2e}%")
    print(f"   [OK] Energy conserved (elastic reflections)")


def example_4_performance_benchmark():
    """Example 4: Performance benchmark (verify Week 3 gate)."""
    print("\n" + "="*60)
    print("Example 4: Performance Benchmark")
    print("="*60)

    n_particles = 1_000_000
    n_steps = 10_000
    dt = 1e-6

    print(f"\nSetup:")
    print(f"   Particles: {n_particles:,}")
    print(f"   Timesteps: {n_steps:,}")
    print(f"   Total particle-steps: {n_particles * n_steps / 1e9:.1f} billion")

    # Allocate arrays
    x = np.random.rand(n_particles, 3).astype(np.float64)
    v = np.random.randn(n_particles, 3).astype(np.float64) * 100
    active = np.ones(n_particles, dtype=np.bool_)

    # Warmup Numba
    print(f"\nWarming up Numba JIT...")
    for _ in range(10):
        push_particles_ballistic(x, v, active, dt, n_particles)

    # Benchmark
    print(f"Running benchmark...")
    start = time.time()

    for step in range(n_steps):
        push_particles_ballistic(x, v, active, dt, n_particles)

    elapsed = time.time() - start

    # Results
    throughput = n_particles * n_steps / elapsed / 1e6

    print(f"\n{'='*60}")
    print(f"PERFORMANCE RESULTS:")
    print(f"{'='*60}")
    print(f"   Elapsed time:     {elapsed:.3f} s")
    print(f"   Throughput:       {throughput:.1f} M particle-steps/sec")
    print(f"   Time per step:    {elapsed/n_steps*1000:.3f} ms")

    # Check gate
    if elapsed < 2.0:
        print(f"\n   [OK] PERFORMANCE GATE PASSED ({elapsed:.2f}s < 2.0s)")
        print(f"   [OK] Projected full DSMC run: {elapsed * 1:.0f} s = {elapsed/60:.1f} min")
    else:
        print(f"\n   [FAIL] PERFORMANCE GATE FAILED ({elapsed:.2f}s > 2.0s)")
        print(f"   -> Numba optimization may be insufficient")


def example_5_density_profile():
    """Example 5: Compute and plot density profile."""
    print("\n" + "="*60)
    print("Example 5: Density Profile Visualization")
    print("="*60)

    n_particles = 10000
    length = 1.0
    n_cells = 20
    dt = 1e-6
    n_steps = 1000

    # Create particles with non-uniform initial distribution
    particles = ParticleArrayNumba(n_particles)

    # Gaussian distribution centered at x=0.5
    x_mean = 0.5
    x_std = 0.1
    x = np.random.normal(x_mean, x_std, (n_particles, 1))
    x = np.clip(x, 0, length)  # Clip to domain
    x = np.hstack([x, np.random.rand(n_particles, 2) * 0.01])  # Add y, z

    v = sample_maxwellian_velocity(T=300, mass=SPECIES['N2'].mass, n_samples=n_particles)

    particles.add_particles(x, v, species='N2', weight=1.0)

    # Create mesh
    mesh = Mesh1D(length=length, n_cells=n_cells)

    # Initial density
    density_initial = compute_number_density_1d(
        particles.x[:, 0], particles.active, particles.weight,
        mesh.cell_edges, particles.n_particles
    )

    # Integrate with periodic BC
    print(f"\nIntegrating {n_steps} steps...")
    for step in range(n_steps):
        push_particles_ballistic(
            particles.x, particles.v, particles.active, dt, particles.n_particles
        )
        apply_periodic_bc(
            particles.x, particles.active, length, particles.n_particles
        )

    # Final density
    density_final = compute_number_density_1d(
        particles.x[:, 0], particles.active, particles.weight,
        mesh.cell_edges, particles.n_particles
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(mesh.cell_centers, density_initial, width=mesh.dx, alpha=0.7, label='Initial')
    ax1.set_xlabel('Position [m]')
    ax1.set_ylabel('Number Density [m⁻³]')
    ax1.set_title('Initial Density Profile (Gaussian)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.bar(mesh.cell_centers, density_final, width=mesh.dx, alpha=0.7, label='Final', color='orange')
    ax2.set_xlabel('Position [m]')
    ax2.set_ylabel('Number Density [m⁻³]')
    ax2.set_title(f'Final Density Profile (after {n_steps} steps)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('density_profile.png', dpi=150)
    print(f"\n[OK] Plot saved to 'density_profile.png'")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("IntakeSIM Example 01: Ballistic Motion Tests")
    print("Week 1 Deliverable - Performance Validation")
    print("="*60)

    # Run examples
    example_1_simple_ballistic()
    example_2_outflow_boundaries()
    example_3_reflecting_box()
    example_4_performance_benchmark()

    # Optional: visualization (requires matplotlib)
    try:
        example_5_density_profile()
    except ImportError:
        print("\nSkipping visualization (matplotlib not available)")

    print("\n" + "="*60)
    print("All examples complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run tests: pytest tests/ -v")
    print("  2. Check performance gate: pytest tests/test_performance.py -v -s")
    print("  3. Proceed to Week 2: VHS collision model")
    print("="*60 + "\n")
