"""
Quick test to verify temperature calculation
"""
import numpy as np
from intakesim.constants import e, m_e
from intakesim.pic.mover import calculate_electron_temperature_eV

# Create seed electrons with 10 eV energy
n_seed = 500
E_seed_eV = 10.0

# All electrons have same kinetic energy
v_magnitude = np.sqrt(2.0 * E_seed_eV * e / m_e)
print(f"v_magnitude = {v_magnitude:.3e} m/s")

# Isotropic distribution
v = np.zeros((n_seed, 3))
np.random.seed(42)
for i in range(n_seed):
    theta = np.arccos(2.0 * np.random.random() - 1.0)
    phi = 2.0 * np.pi * np.random.random()
    v[i, 0] = v_magnitude * np.sin(theta) * np.cos(phi)
    v[i, 1] = v_magnitude * np.sin(theta) * np.sin(phi)
    v[i, 2] = v_magnitude * np.cos(theta)

# Calculate T_e using new function
active = np.ones(n_seed, dtype=bool)
species_id = np.zeros(n_seed, dtype=np.int32)  # All electrons

T_e_eV = calculate_electron_temperature_eV(v, active, species_id, n_seed)

print(f"\nResults:")
print(f"  Input energy: {E_seed_eV} eV")
print(f"  Calculated T_e: {T_e_eV:.2f} eV")
print(f"  Expected T_e: {(2.0/3.0) * E_seed_eV:.2f} eV")
print(f"  Error: {abs(T_e_eV - (2.0/3.0)*E_seed_eV) / ((2.0/3.0)*E_seed_eV) * 100:.1f}%")

# Manual calculation
v_sq_sum = 0.0
for i in range(n_seed):
    v_sq_sum += v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2
v_sq_mean = v_sq_sum / n_seed
print(f"\n  <v²> = {v_sq_mean:.3e} m²/s²")

T_e_manual = (m_e / (3.0 * e)) * v_sq_mean
print(f"  T_e (manual) = {T_e_manual:.2f} eV")

# Energy calculation
E_mean_eV = 0.5 * m_e * v_sq_mean / e
print(f"  <E> = {E_mean_eV:.2f} eV")
print(f"  (2/3) <E> = {(2.0/3.0) * E_mean_eV:.2f} eV")
