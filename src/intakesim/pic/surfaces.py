"""
Plasma-Surface Interactions for PIC Simulations

Implements:
    - Secondary Electron Emission (SEE) using Vaughan model
    - Ion-induced electron emission
    - Surface boundary conditions
    - Material property database

Secondary Electron Emission (SEE):
    When electrons or ions hit walls, they can release secondary electrons.
    This affects:
        - Electron temperature (lowers T_e by 1-2 eV)
        - Sheath potential
        - Discharge sustainability

Vaughan Formula:
    delta(E) = delta_max * (E/E_max)^n * exp(n*(1-E/E_max))

    where:
        delta_max: Maximum yield
        E_max: Energy at maximum yield [eV]
        n: Shape parameter (~0.62 for most materials)

References:
    Vaughan (1989), IEEE Trans. Electron Devices
    Phelps & Petrovic (1999), Plasma Sources Sci. Technol.

Author: AeriSat Systems
Date: 2025
"""

import numpy as np
import numba
import sys
import os

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from intakesim.constants import e, m_e, ID_TO_SPECIES
else:
    from ..constants import e, m_e

# ==================== MATERIAL PROPERTIES ====================

# Secondary Electron Emission (SEE) parameters
MATERIAL_PROPERTIES = {
    'molybdenum': {
        'delta_max': 1.25,  # Maximum SEE yield
        'E_max': 350.0,  # Energy at max yield [eV]
        'n': 0.62,  # Shape parameter
        'E_secondary_mean': 3.0,  # Mean secondary electron energy [eV]
        'work_function': 4.6,  # Work function [eV]
    },
    'stainless_steel': {
        'delta_max': 1.1,
        'E_max': 300.0,
        'n': 0.62,
        'E_secondary_mean': 3.5,
        'work_function': 4.5,
    },
    'aluminum': {
        'delta_max': 0.97,
        'E_max': 300.0,
        'n': 0.62,
        'E_secondary_mean': 3.0,
        'work_function': 4.3,
    },
    'ceramic': {
        'delta_max': 2.5,  # Insulators have higher SEE
        'E_max': 300.0,
        'n': 0.62,
        'E_secondary_mean': 4.0,
        'work_function': 5.0,
    },
    'copper': {
        'delta_max': 1.3,
        'E_max': 600.0,
        'n': 0.62,
        'E_secondary_mean': 3.5,
        'work_function': 4.7,
    },
}

# Ion-induced emission yields (typical values for 100 eV ions)
ION_EMISSION_YIELDS = {
    'molybdenum': 0.08,
    'stainless_steel': 0.12,
    'aluminum': 0.10,
    'ceramic': 0.15,
    'copper': 0.09,
}

# ==================== VAUGHAN SEE MODEL ====================

@numba.njit
def vaughan_see_yield(E_impact_eV, delta_max, E_max, n):
    """
    Calculate secondary electron emission yield using Vaughan formula.

    delta(E) = delta_max * (E/E_max)^n * exp(n*(1-E/E_max))

    Args:
        E_impact_eV: Impact energy [eV]
        delta_max: Maximum yield (material parameter)
        E_max: Energy at maximum yield [eV]
        n: Shape parameter (typically ~0.62)

    Returns:
        delta: SEE yield (number of secondary electrons per incident electron)

    Reference:
        Vaughan (1989), "A new formula for secondary emission yield"
    """
    if E_impact_eV <= 0.0:
        return 0.0

    # Vaughan formula
    E_ratio = E_impact_eV / E_max

    # Prevent overflow for very high energies
    if E_ratio > 10.0:
        # For E >> E_max, yield decreases as ~1/E
        return delta_max * np.exp(n) * (E_max / E_impact_eV)

    delta = delta_max * (E_ratio ** n) * np.exp(n * (1.0 - E_ratio))

    return delta


@numba.njit
def sample_secondary_electron_energy(E_mean_eV):
    """
    Sample secondary electron energy from distribution.

    Uses Maxwell-Boltzmann-like distribution:
        f(E) ~ E * exp(-E / E_0)

    where E_0 = E_mean / 2 (so that mean of distribution = E_mean)

    Most secondaries are low energy (2-5 eV).

    Args:
        E_mean_eV: Mean secondary electron energy [eV]

    Returns:
        E_secondary_eV: Sampled energy [eV]
    """
    # For f(E) ~ E * exp(-E/E_0), the mean is 2*E_0
    # So to get desired mean, use E_0 = E_mean / 2
    E_0 = E_mean_eV / 2.0

    r1 = np.random.random()
    r2 = np.random.random()

    # Two-exponential sampling for E*exp(-E/E_0) distribution
    E_secondary_eV = -E_0 * np.log(r1 * r2)

    # Clamp to reasonable range
    if E_secondary_eV < 0.5:
        E_secondary_eV = 0.5
    elif E_secondary_eV > 50.0:
        E_secondary_eV = 50.0

    return E_secondary_eV


@numba.njit
def sample_secondary_electron_direction(normal_x):
    """
    Sample secondary electron emission direction.

    Cosine distribution: f(theta) ~ cos(theta)
    where theta is angle from surface normal.

    Args:
        normal_x: Surface normal direction (+1 for right wall, -1 for left wall)

    Returns:
        vx, vy, vz: Unit direction vector for secondary electron
    """
    # Sample polar angle with cos(theta) distribution
    # This gives preferential emission normal to surface
    r1 = np.random.random()
    cos_theta = np.sqrt(r1)  # For cos distribution
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    # Azimuthal angle (uniform)
    phi = 2.0 * np.pi * np.random.random()

    # Convert to Cartesian
    # x is normal direction, y and z are tangential
    vx = normal_x * cos_theta  # Into plasma (away from wall)
    vy = sin_theta * np.cos(phi)
    vz = sin_theta * np.sin(phi)

    return vx, vy, vz


@numba.njit
def ion_induced_emission_yield(E_impact_eV, gamma_100eV, E_ref=100.0):
    """
    Ion-induced secondary electron emission yield.

    Scales approximately as: gamma(E) ~ gamma_ref * sqrt(E / E_ref)

    Args:
        E_impact_eV: Ion impact energy [eV]
        gamma_100eV: Yield at 100 eV reference energy
        E_ref: Reference energy [eV] (default: 100)

    Returns:
        gamma: Ion-induced emission yield
    """
    if E_impact_eV <= 0.0:
        return 0.0

    # Square-root energy scaling (approximate)
    gamma = gamma_100eV * np.sqrt(E_impact_eV / E_ref)

    # Clamp to reasonable range
    if gamma > 1.0:
        gamma = 1.0

    return gamma


# ==================== SURFACE BOUNDARY CONDITIONS ====================

def apply_see_boundary_conditions(
    particles,
    mesh,
    material='molybdenum',
    max_new_particles=1000
):
    """
    Apply Secondary Electron Emission at walls.

    When particles hit walls:
        1. Calculate SEE yield based on impact energy
        2. Remove incident particle
        3. Create secondary electrons based on yield

    Args:
        particles: ParticleArrayNumba instance
        mesh: Mesh1DPIC instance
        material: Material name (default: 'molybdenum')
        max_new_particles: Maximum secondaries to create

    Returns:
        diagnostics: dict with SEE statistics
            {
                'n_electrons_absorbed': int,
                'n_ions_absorbed': int,
                'n_secondaries_created': int,
                'mean_see_yield': float,
            }
    """
    # Import here to avoid circular dependency
    try:
        from ..constants import ID_TO_SPECIES
    except ImportError:
        from intakesim.constants import ID_TO_SPECIES

    # Material properties
    if material not in MATERIAL_PROPERTIES:
        raise ValueError(f"Unknown material: {material}")

    mat_props = MATERIAL_PROPERTIES[material]
    delta_max = mat_props['delta_max']
    E_max = mat_props['E_max']
    n_vaughan = mat_props['n']
    E_secondary_mean = mat_props['E_secondary_mean']

    ion_gamma = ION_EMISSION_YIELDS.get(material, 0.1)

    # Statistics
    n_electrons_absorbed = 0
    n_ions_absorbed = 0
    n_secondaries_created = 0
    total_see_yield = 0.0
    n_see_events = 0

    # Arrays for new secondary electrons
    new_particles_x = np.zeros((max_new_particles, 3), dtype=np.float64)
    new_particles_v = np.zeros((max_new_particles, 3), dtype=np.float64)
    new_particles_weight = np.zeros(max_new_particles, dtype=np.float64)
    n_new = 0

    # Domain boundaries
    x_min = mesh.x_min
    x_max = mesh.x_max

    # Process each particle
    for i in range(particles.n_particles):
        if not particles.active[i]:
            continue

        x_pos = particles.x[i, 0]

        # Check if particle is outside domain
        hit_wall = False
        normal_x = 0.0

        if x_pos < x_min:
            hit_wall = True
            normal_x = +1.0  # Left wall, normal points right (into domain)
        elif x_pos > x_max:
            hit_wall = True
            normal_x = -1.0  # Right wall, normal points left (into domain)

        if not hit_wall:
            continue

        # Particle hit wall - process SEE
        species_name = ID_TO_SPECIES[particles.species_id[i]]

        # Calculate impact energy
        v = particles.v[i, :]
        v_mag_sq = np.sum(v * v)

        if species_name == 'e':
            # Electron impact
            E_impact_eV = 0.5 * m_e * v_mag_sq / e

            # Calculate SEE yield
            delta = vaughan_see_yield(E_impact_eV, delta_max, E_max, n_vaughan)

            total_see_yield += delta
            n_see_events += 1

            # Determine number of secondaries to create
            # Stochastic: delta is mean, sample from Poisson-like
            n_secondaries = int(delta)  # Integer part
            if np.random.random() < (delta - n_secondaries):
                n_secondaries += 1  # Fractional part stochastically

            # Create secondary electrons
            for _ in range(n_secondaries):
                if n_new >= max_new_particles:
                    break

                # Position: at wall
                new_particles_x[n_new, 0] = x_min if normal_x > 0 else x_max
                new_particles_x[n_new, 1] = 0.0
                new_particles_x[n_new, 2] = 0.0

                # Energy
                E_sec_eV = sample_secondary_electron_energy(E_secondary_mean)
                E_sec_J = E_sec_eV * e
                v_mag = np.sqrt(2.0 * E_sec_J / m_e)

                # Direction (cosine distribution)
                dir_x, dir_y, dir_z = sample_secondary_electron_direction(normal_x)

                new_particles_v[n_new, 0] = v_mag * dir_x
                new_particles_v[n_new, 1] = v_mag * dir_y
                new_particles_v[n_new, 2] = v_mag * dir_z

                new_particles_weight[n_new] = particles.weight[i]

                n_new += 1
                n_secondaries_created += 1

            n_electrons_absorbed += 1

        elif '+' in species_name:
            # Ion impact
            # Get ion mass (approximate for now)
            if 'N2' in species_name:
                m_ion = 28 * 1.661e-27  # kg
            elif 'O2' in species_name:
                m_ion = 32 * 1.661e-27
            elif 'O' in species_name:
                m_ion = 16 * 1.661e-27
            else:
                m_ion = 28 * 1.661e-27  # Default to N2

            E_impact_eV = 0.5 * m_ion * v_mag_sq / e

            # Ion-induced emission
            gamma = ion_induced_emission_yield(E_impact_eV, ion_gamma)

            # Create secondaries from ion impact
            n_secondaries = int(gamma)
            if np.random.random() < (gamma - n_secondaries):
                n_secondaries += 1

            for _ in range(n_secondaries):
                if n_new >= max_new_particles:
                    break

                # Position: at wall
                new_particles_x[n_new, 0] = x_min if normal_x > 0 else x_max
                new_particles_x[n_new, 1] = 0.0
                new_particles_x[n_new, 2] = 0.0

                # Low energy secondaries from ions
                E_sec_eV = sample_secondary_electron_energy(2.0)  # Lower energy
                E_sec_J = E_sec_eV * e
                v_mag = np.sqrt(2.0 * E_sec_J / m_e)

                # Direction
                dir_x, dir_y, dir_z = sample_secondary_electron_direction(normal_x)

                new_particles_v[n_new, 0] = v_mag * dir_x
                new_particles_v[n_new, 1] = v_mag * dir_y
                new_particles_v[n_new, 2] = v_mag * dir_z

                new_particles_weight[n_new] = particles.weight[i]

                n_new += 1
                n_secondaries_created += 1

            n_ions_absorbed += 1

        # Remove incident particle
        particles.active[i] = False

    # Add secondary electrons to particle array
    for idx in range(n_new):
        if particles.n_particles >= particles.max_particles:
            break

        x_new = new_particles_x[idx:idx+1, :]
        v_new = new_particles_v[idx:idx+1, :]
        particles.add_particles(x_new, v_new, "e", weight=new_particles_weight[idx])

    # Compute mean SEE yield
    mean_see_yield = total_see_yield / n_see_events if n_see_events > 0 else 0.0

    return {
        'n_electrons_absorbed': n_electrons_absorbed,
        'n_ions_absorbed': n_ions_absorbed,
        'n_secondaries_created': n_secondaries_created,
        'mean_see_yield': mean_see_yield,
    }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Surface Physics Module - Self Test")
    print("=" * 60)

    # Test 1: Vaughan SEE yield
    print("\nTest 1: Vaughan SEE yield for molybdenum...")

    mat = MATERIAL_PROPERTIES['molybdenum']
    delta_max = mat['delta_max']
    E_max = mat['E_max']
    n = mat['n']

    # Test at various energies
    energies = [10, 50, 100, 200, 350, 500, 1000]

    print(f"  Material: molybdenum")
    print(f"  delta_max = {delta_max}, E_max = {E_max} eV, n = {n}")
    print()
    print(f"  {'E [eV]':<10} {'delta':>8}")
    print(f"  {'-'*20}")

    for E in energies:
        delta = vaughan_see_yield(E, delta_max, E_max, n)
        print(f"  {E:<10} {delta:>8.3f}")

    # Check that maximum occurs near E_max
    delta_at_max = vaughan_see_yield(E_max, delta_max, E_max, n)
    print(f"\n  At E_max = {E_max} eV: delta = {delta_at_max:.3f}")
    print(f"  Expected: delta ~= delta_max = {delta_max}")

    error = abs(delta_at_max - delta_max) / delta_max
    assert error < 0.05, f"Peak not at E_max! Error = {error:.3f}"
    print(f"  [PASS] Peak yield within 5% of delta_max")

    # Test 2: Secondary electron energy distribution
    print("\nTest 2: Secondary electron energy distribution...")

    E_mean = 3.0  # eV
    n_samples = 10000

    energies_sampled = []
    for _ in range(n_samples):
        E_sec = sample_secondary_electron_energy(E_mean)
        energies_sampled.append(E_sec)

    mean_sampled = np.mean(energies_sampled)
    std_sampled = np.std(energies_sampled)

    print(f"  E_mean (input): {E_mean} eV")
    print(f"  <E> (sampled): {mean_sampled:.2f} eV")
    print(f"  std(E): {std_sampled:.2f} eV")

    # Mean should be close to input (within 10%)
    error_mean = abs(mean_sampled - E_mean) / E_mean
    assert error_mean < 0.1, f"Mean energy error {error_mean:.3f} > 10%"
    print(f"  [PASS] Mean energy within 10% ({error_mean*100:.1f}%)")

    # Test 3: Secondary electron emission direction
    print("\nTest 3: Secondary electron emission direction...")

    normal_x = +1.0  # Left wall (emit to right)
    n_samples = 10000

    vx_samples = []
    for _ in range(n_samples):
        vx, vy, vz = sample_secondary_electron_direction(normal_x)

        # Check normalized
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        assert abs(v_mag - 1.0) < 1e-6, "Direction not normalized!"

        # Check vx > 0 (emitted into domain)
        assert vx > 0, "Secondary emitted into wall!"

        vx_samples.append(vx)

    # For cosine distribution: <cos(theta)> = 2/3
    mean_vx = np.mean(vx_samples)
    expected_vx = 2.0 / 3.0

    print(f"  Normal direction: +x (right)")
    print(f"  <vx> (sampled): {mean_vx:.3f}")
    print(f"  Expected (cos dist): {expected_vx:.3f}")

    error_vx = abs(mean_vx - expected_vx) / expected_vx
    assert error_vx < 0.05, f"Direction distribution error {error_vx:.3f} > 5%"
    print(f"  [PASS] Cosine distribution validated ({error_vx*100:.1f}% error)")

    # Test 4: Ion-induced emission
    print("\nTest 4: Ion-induced emission...")

    gamma_100eV = 0.08  # Molybdenum
    energies_ion = [10, 50, 100, 200, 500]

    print(f"  gamma(100 eV) = {gamma_100eV}")
    print()
    print(f"  {'E_ion [eV]':<12} {'gamma':>8}")
    print(f"  {'-'*22}")

    for E in energies_ion:
        gamma = ion_induced_emission_yield(E, gamma_100eV)
        print(f"  {E:<12} {gamma:>8.4f}")

    # Check scaling: gamma(400 eV) should be ~2 * gamma(100 eV)
    gamma_400 = ion_induced_emission_yield(400.0, gamma_100eV)
    expected_400 = gamma_100eV * np.sqrt(400.0 / 100.0)  # = 2 * gamma_100eV

    error_ion = abs(gamma_400 - expected_400) / expected_400
    assert error_ion < 0.01, f"Ion emission scaling error {error_ion:.3f}"
    print(f"\n  [PASS] Square-root energy scaling validated")

    print("\n" + "=" * 60)
    print("Surface physics module validated!")
    print("=" * 60)
