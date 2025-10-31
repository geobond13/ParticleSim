"""Quick performance verification for Week 1 gate"""
import numpy as np
import time
from intakesim.dsmc.mover import push_particles_ballistic

print("Week 1 Performance Gate Test")
print("="*60)

# Smaller test first
n_particles = 100_000
n_steps = 1_000
dt = 1e-6

x = np.random.rand(n_particles, 3).astype(np.float64)
v = np.random.randn(n_particles, 3).astype(np.float64) * 100
active = np.ones(n_particles, dtype=np.bool_)

print(f"\nWarmup (Numba compilation)...")
for _ in range(10):
    push_particles_ballistic(x, v, active, dt, n_particles)

print(f"\nTest 1: {n_particles:,} particles × {n_steps:,} steps")
start = time.time()
for _ in range(n_steps):
    push_particles_ballistic(x, v, active, dt, n_particles)
elapsed = time.time() - start

throughput = n_particles * n_steps / elapsed / 1e6
print(f"  Time: {elapsed:.3f}s")
print(f"  Throughput: {throughput:.1f} M particle-steps/sec")

# Now try the full gate
n_particles = 1_000_000
n_steps = 10_000

x = np.random.rand(n_particles, 3).astype(np.float64)
v = np.random.randn(n_particles, 3).astype(np.float64) * 100
active = np.ones(n_particles, dtype=np.bool_)

# Warmup
for _ in range(5):
    push_particles_ballistic(x, v, active, dt, n_particles)

print(f"\nTest 2 (Performance Gate): {n_particles:,} particles × {n_steps:,} steps")
print("This should take <2 seconds if Numba is working correctly...")
start = time.time()
for _ in range(n_steps):
    push_particles_ballistic(x, v, active, dt, n_particles)
elapsed = time.time() - start

throughput = n_particles * n_steps / elapsed / 1e6
print(f"  Time: {elapsed:.3f}s")
print(f"  Throughput: {throughput:.1f} M particle-steps/sec")

if elapsed < 2.0:
    print(f"\n[PASS] Performance gate PASSED ({elapsed:.2f}s < 2.0s)")
else:
    print(f"\n[FAIL] Performance gate FAILED ({elapsed:.2f}s > 2.0s)")
    print("  -> Check that Numba is installed correctly")
    print("  -> Try: pip install numba --upgrade")
