"""
Performance Gate Tests (Week 3 Checkpoint)

These tests enforce minimum performance requirements.
If they fail, Numba optimization is insufficient.

CRITICAL GATES:
1. Ballistic motion: 10^6 particles, 10k steps in <2 sec
2. Numba speedup: >50× vs pure Python
"""

import pytest
import time
import numpy as np
from intakesim.dsmc.mover import push_particles_ballistic


@pytest.mark.performance
class TestPerformanceGates:
    """Critical performance gates for Week 3 checkpoint."""

    def test_ballistic_motion_performance_gate(self):
        """
        WEEK 3 PERFORMANCE GATE

        Requirement: 10^6 particles, 10,000 timesteps in <2 seconds

        This ensures that a full DSMC run (10 ms simulation time)
        completes in <60 minutes on an 8-core laptop.

        Calculation:
        - Full run: 10 ms / 1 μs = 10,000 timesteps
        - If each timestep takes <2s / 10,000 = 0.2 ms
        - Then full run: 10,000 × 0.2 ms = 2,000 s = 33 min ✓

        GATE FAILURE → Add Cython or consider C++
        """
        n_particles = 1_000_000
        n_steps = 10_000
        dt = 1e-6  # 1 microsecond

        # Setup
        x = np.random.rand(n_particles, 3).astype(np.float64)
        v = np.random.randn(n_particles, 3).astype(np.float64) * 100
        active = np.ones(n_particles, dtype=np.bool_)

        # Warmup Numba compilation
        print("\n  Warming up Numba JIT (first 10 iterations)...")
        for _ in range(10):
            push_particles_ballistic(x, v, active, dt, n_particles)

        # Actual benchmark
        print(f"\n  Benchmarking {n_particles:,} particles × {n_steps:,} steps...")
        start = time.time()

        for step in range(n_steps):
            push_particles_ballistic(x, v, active, dt, n_particles)

        elapsed = time.time() - start

        # Compute metrics
        particle_steps = n_particles * n_steps
        throughput = particle_steps / elapsed / 1e6  # Million particle-steps/sec

        print(f"\n  Results:")
        print(f"    Elapsed time:  {elapsed:.3f} s")
        print(f"    Throughput:    {throughput:.1f} M particle-steps/sec")
        print(f"    Time per step: {elapsed/n_steps*1000:.3f} ms")

        # GATE: Must complete in <2 seconds
        assert elapsed < 2.0, (
            f"PERFORMANCE GATE FAILED: {elapsed:.2f}s > 2.0s\n"
            f"Throughput too low ({throughput:.1f} M particle-steps/sec)\n"
            f"→ Numba optimization insufficient. Consider Cython or C++."
        )

        print(f"\n  ✅ PERFORMANCE GATE PASSED ({elapsed:.2f}s < 2.0s)")

    def test_numba_speedup_gate(self):
        """
        NUMBA SPEEDUP GATE

        Requirement: >50× speedup vs pure Python

        This confirms Numba JIT compilation is working correctly.

        GATE FAILURE → Check:
        1. @njit decorator present
        2. No Python objects in hot loop
        3. Correct array types (float64, bool_)
        4. parallel=True for multi-threading
        """
        n_particles = 10_000
        n_steps = 1_000
        dt = 1e-6

        # Setup
        x_numba = np.random.rand(n_particles, 3).astype(np.float64)
        v_numba = np.random.randn(n_particles, 3).astype(np.float64) * 100
        active = np.ones(n_particles, dtype=np.bool_)

        x_python = x_numba.copy()
        v_python = v_numba.copy()

        # Pure Python version (for comparison)
        def push_particles_python(x, v, dt, n):
            """Pure Python ballistic motion (slow!)."""
            for i in range(n):
                x[i, 0] += v[i, 0] * dt
                x[i, 1] += v[i, 1] * dt
                x[i, 2] += v[i, 2] * dt

        # Warm up Numba
        push_particles_ballistic(x_numba, v_numba, active, dt, n_particles)

        # Benchmark Python
        print("\n  Benchmarking pure Python...")
        start = time.time()
        for _ in range(n_steps):
            push_particles_python(x_python, v_python, dt, n_particles)
        t_python = time.time() - start

        # Benchmark Numba
        print(f"  Benchmarking Numba...")
        start = time.time()
        for _ in range(n_steps):
            push_particles_ballistic(x_numba, v_numba, active, dt, n_particles)
        t_numba = time.time() - start

        # Compute speedup
        speedup = t_python / t_numba

        print(f"\n  Results:")
        print(f"    Python time: {t_python:.3f} s")
        print(f"    Numba time:  {t_numba:.3f} s")
        print(f"    Speedup:     {speedup:.1f}×")

        # GATE: Must achieve >50× speedup
        assert speedup > 50, (
            f"NUMBA SPEEDUP GATE FAILED: {speedup:.1f}× < 50×\n"
            f"JIT compilation not working properly.\n"
            f"→ Check @njit decorator and array types."
        )

        print(f"\n  ✅ NUMBA SPEEDUP GATE PASSED ({speedup:.1f}× > 50×)")

    @pytest.mark.slow
    def test_full_dsmc_run_estimate(self):
        """
        Estimate runtime for full DSMC simulation.

        Full simulation: 10^6 particles, 10 ms (10,000 steps @ 1 μs)
        Target: <60 minutes

        This is a slower test that gives realistic estimate.
        Mark as 'slow' so it can be skipped in quick tests.
        """
        n_particles = 1_000_000
        n_steps_sample = 1_000  # Sample with 1000 steps
        dt = 1e-6

        x = np.random.rand(n_particles, 3).astype(np.float64)
        v = np.random.randn(n_particles, 3).astype(np.float64) * 100
        active = np.ones(n_particles, dtype=np.bool_)

        # Warmup
        for _ in range(10):
            push_particles_ballistic(x, v, active, dt, n_particles)

        # Sample timing
        print(f"\n  Running {n_steps_sample} steps for time estimate...")
        start = time.time()
        for _ in range(n_steps_sample):
            push_particles_ballistic(x, v, active, dt, n_particles)
        elapsed_sample = time.time() - start

        # Extrapolate to full run
        n_steps_full = 10_000
        elapsed_full_estimate = elapsed_sample * (n_steps_full / n_steps_sample)

        print(f"\n  Sample run ({n_steps_sample} steps): {elapsed_sample:.1f} s")
        print(f"  Estimated full run ({n_steps_full} steps): {elapsed_full_estimate/60:.1f} min")

        # Goal: <60 minutes
        assert elapsed_full_estimate < 3600, (
            f"Full DSMC run would take {elapsed_full_estimate/60:.1f} min > 60 min\n"
            f"Performance insufficient for production runs."
        )

        print(f"\n  ✅ Full DSMC estimate OK ({elapsed_full_estimate/60:.1f} min < 60 min)")


@pytest.mark.performance
class TestMemoryEfficiency:
    """Test memory usage and efficiency."""

    def test_soa_layout_size(self):
        """Verify Structure-of-Arrays layout is memory efficient."""
        from intakesim.particles import ParticleArrayNumba

        n_particles = 1_000_000

        particles = ParticleArrayNumba(n_particles)

        # Compute memory usage
        mem_x = particles.x.nbytes
        mem_v = particles.v.nbytes
        mem_weight = particles.weight.nbytes
        mem_species = particles.species_id.nbytes
        mem_active = particles.active.nbytes

        total_mem_mb = (mem_x + mem_v + mem_weight + mem_species + mem_active) / 1e6

        bytes_per_particle = total_mem_mb * 1e6 / n_particles

        print(f"\n  Memory usage for {n_particles:,} particles:")
        print(f"    x:         {mem_x/1e6:.1f} MB")
        print(f"    v:         {mem_v/1e6:.1f} MB")
        print(f"    weight:    {mem_weight/1e6:.1f} MB")
        print(f"    species:   {mem_species/1e6:.1f} MB")
        print(f"    active:    {mem_active/1e6:.1f} MB")
        print(f"    Total:     {total_mem_mb:.1f} MB")
        print(f"    Per particle: {bytes_per_particle:.0f} bytes")

        # Each particle: 3×8 (x) + 3×8 (v) + 8 (weight) + 4 (species) + 1 (active) = 61 bytes
        # With padding: ~64 bytes expected

        assert bytes_per_particle < 100, "Memory usage too high (>100 bytes/particle)"

        print(f"\n  ✅ Memory efficiency OK ({bytes_per_particle:.0f} bytes/particle)")


if __name__ == "__main__":
    # Run performance tests with verbose output
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
