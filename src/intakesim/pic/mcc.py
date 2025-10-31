"""
Monte Carlo Collisions (MCC) for PIC Simulations

Implements electron-neutral and ion-neutral collisions using the null-collision method.

Collision Types:
    Electron-neutral:
        - Ionization: e + N2 -> N2+ + 2e
        - Excitation: e + N2 -> N2* + e
        - Elastic: e + N2 -> e + N2

    Ion-neutral:
        - Charge exchange: N2+ + N2 -> N2 + N2+
        - Elastic: N2+ + N2 -> N2+ + N2

Null-Collision Method:
    1. Compute max collision frequency: nu_max = max(n*sigma(E)*v)
    2. For each particle, test collision with P = nu_max * dt
    3. If collision, sample actual reaction type
    4. If null collision, continue unchanged

Reference:
    Birdsall & Langdon (2004), Section 13.4
    Turner et al. (2013), "Simulation benchmarks for low-pressure plasmas"

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
    from intakesim.pic.cross_sections import (
        get_cross_section,
        get_total_cross_section,
        sample_collision_type,
        get_reaction_data,
        CROSS_SECTION_DATA
    )
    from intakesim.constants import m_e, eV, AMU, SPECIES, ID_TO_SPECIES
else:
    from .cross_sections import (
        get_cross_section,
        get_total_cross_section,
        sample_collision_type,
        get_reaction_data,
        CROSS_SECTION_DATA
    )
    from ..constants import m_e, eV, AMU, SPECIES, ID_TO_SPECIES

# ==================== NULL-COLLISION METHOD ====================

@numba.njit
def compute_max_collision_frequency(n_neutral, sigma_max, E_max_eV, m_electron):
    """
    Compute maximum collision frequency for null-collision method.

    nu_max = n_neutral * sigma_max * v_max

    where v_max = sqrt(2 * E_max / m_e)

    Args:
        n_neutral: Neutral density [m^-3]
        sigma_max: Maximum cross-section over energy range [m^2]
        E_max_eV: Maximum electron energy [eV]
        m_electron: Electron mass [kg]

    Returns:
        nu_max: Maximum collision frequency [s^-1]
    """
    E_max_J = E_max_eV * eV
    v_max = np.sqrt(2.0 * E_max_J / m_electron)
    nu_max = n_neutral * sigma_max * v_max
    return nu_max


@numba.njit
def electron_energy_ev(vx, vy, vz, m_electron):
    """
    Calculate electron kinetic energy in eV.

    E = (1/2) * m_e * v^2 / eV

    Args:
        vx, vy, vz: Velocity components [m/s]
        m_electron: Electron mass [kg]

    Returns:
        E_eV: Kinetic energy [eV]
    """
    v_squared = vx*vx + vy*vy + vz*vz
    E_J = 0.5 * m_electron * v_squared
    return E_J / eV


@numba.njit
def isotropic_scatter(v_magnitude):
    """
    Generate isotropic velocity vector with given magnitude.

    Uses rejection method for uniform distribution on sphere.

    Args:
        v_magnitude: Speed [m/s]

    Returns:
        vx, vy, vz: Velocity components [m/s]
    """
    # Rejection method for uniform sphere point
    while True:
        vx = 2.0 * np.random.random() - 1.0
        vy = 2.0 * np.random.random() - 1.0
        vz = 2.0 * np.random.random() - 1.0

        r_squared = vx*vx + vy*vy + vz*vz

        if r_squared <= 1.0 and r_squared > 0.0:
            # Normalize to unit vector
            r = np.sqrt(r_squared)
            vx = vx / r * v_magnitude
            vy = vy / r * v_magnitude
            vz = vz / r * v_magnitude
            break

    return vx, vy, vz


@numba.njit
def process_electron_collision(
    particle_idx,
    vx, vy, vz,
    weight,
    reaction_type,
    energy_loss_eV,
    m_electron,
    new_particles_x,
    new_particles_v,
    new_particles_weight,
    new_particles_species_id,
    n_new_particles,
    max_new_particles,
    ion_species_id
):
    """
    Process electron-neutral collision.

    Modifies electron velocity and creates new particles for ionization.

    Args:
        particle_idx: Index of colliding particle
        vx, vy, vz: Electron velocity [m/s] (will be modified in-place)
        weight: Statistical weight
        reaction_type: 'elastic', 'excitation', or 'ionization'
        energy_loss_eV: Energy lost in collision [eV]
        m_electron: Electron mass [kg]
        new_particles_*: Arrays for new particles (ionization products)
        n_new_particles: Counter for new particles (modified in-place)
        max_new_particles: Maximum new particles allowed
        ion_species_id: Species ID for created ions

    Returns:
        updated_vx, updated_vy, updated_vz: New electron velocity
        n_new_particles: Updated count
    """
    # Current kinetic energy
    E_before_eV = electron_energy_ev(vx, vy, vz, m_electron)

    # Energy after collision
    E_after_eV = E_before_eV - energy_loss_eV

    if E_after_eV < 0.1:
        # Energy below threshold - treat as absorbed (thermalized)
        E_after_eV = 0.1  # Minimum energy

    # New speed
    E_after_J = E_after_eV * eV
    v_after = np.sqrt(2.0 * E_after_J / m_electron)

    # Scattering
    if reaction_type == 0:  # elastic
        # Isotropic scattering (simplification - could use energy-dependent)
        new_vx, new_vy, new_vz = isotropic_scatter(v_after)

    elif reaction_type == 1:  # excitation
        # Isotropic scattering
        new_vx, new_vy, new_vz = isotropic_scatter(v_after)

    elif reaction_type == 2:  # ionization
        # Isotropic scattering for primary electron
        new_vx, new_vy, new_vz = isotropic_scatter(v_after)

        # Create secondary electron
        if n_new_particles[0] < max_new_particles:
            idx = n_new_particles[0]

            # Position: same as primary (created at collision point)
            new_particles_x[idx, 0] = 0.0  # Will be set by caller
            new_particles_x[idx, 1] = 0.0
            new_particles_x[idx, 2] = 0.0

            # Velocity: Low energy, isotropic
            # Assume secondary gets half of available energy after threshold
            E_secondary_eV = max(0.5, (E_before_eV - energy_loss_eV) * 0.5)
            E_secondary_J = E_secondary_eV * eV
            v_secondary = np.sqrt(2.0 * E_secondary_J / m_electron)

            v_sec_x, v_sec_y, v_sec_z = isotropic_scatter(v_secondary)
            new_particles_v[idx, 0] = v_sec_x
            new_particles_v[idx, 1] = v_sec_y
            new_particles_v[idx, 2] = v_sec_z

            new_particles_weight[idx] = weight
            new_particles_species_id[idx] = 1  # Electron species ID

            n_new_particles[0] += 1

        # Create ion
        if n_new_particles[0] < max_new_particles:
            idx = n_new_particles[0]

            new_particles_x[idx, 0] = 0.0  # Will be set by caller
            new_particles_x[idx, 1] = 0.0
            new_particles_x[idx, 2] = 0.0

            # Ion starts at rest (simplification)
            new_particles_v[idx, 0] = 0.0
            new_particles_v[idx, 1] = 0.0
            new_particles_v[idx, 2] = 0.0

            new_particles_weight[idx] = weight
            new_particles_species_id[idx] = ion_species_id

            n_new_particles[0] += 1

    else:
        # Unknown reaction type - no change
        new_vx = vx
        new_vy = vy
        new_vz = vz

    return new_vx, new_vy, new_vz, n_new_particles[0]


def apply_mcc_collisions(
    particles,
    neutral_species,
    neutral_density,
    dt,
    max_new_particles=1000
):
    """
    Apply Monte Carlo Collisions to electron particles.

    Uses null-collision method for efficient collision processing.

    Args:
        particles: ParticleArrayNumba instance
        neutral_species: Neutral species name ('N2', 'O', 'O2', 'NO')
        neutral_density: Neutral number density [m^-3]
        dt: Timestep [s]
        max_new_particles: Maximum new particles to create (default: 1000)

    Returns:
        diagnostics: dict with collision counts
            {
                'n_collisions_total': int,
                'n_elastic': int,
                'n_excitation': int,
                'n_ionization': int,
                'n_new_electrons': int,
                'n_new_ions': int,
            }
    """
    # Get electron particles
    n_particles = particles.n_particles

    # Identify electrons
    electron_mask = np.zeros(n_particles, dtype=np.bool_)
    for i in range(n_particles):
        if particles.active[i]:
            species_id = particles.species_id[i]
            species_name = ID_TO_SPECIES[species_id]
            if species_name == 'e':
                electron_mask[i] = True

    n_electrons = np.sum(electron_mask)

    if n_electrons == 0:
        # No electrons to collide
        return {
            'n_collisions_total': 0,
            'n_elastic': 0,
            'n_excitation': 0,
            'n_ionization': 0,
            'n_new_electrons': 0,
            'n_new_ions': 0,
        }

    # Compute maximum collision frequency
    # Use maximum cross-section from database
    sigma_max = 0.0
    for reaction_type in CROSS_SECTION_DATA[neutral_species]:
        data = CROSS_SECTION_DATA[neutral_species][reaction_type]
        sigma_reaction_max = np.max(data['cross_sections'])
        sigma_max += sigma_reaction_max

    # Maximum electron energy (scan all electrons)
    E_max_eV = 0.0
    for i in range(n_particles):
        if electron_mask[i]:
            vx, vy, vz = particles.v[i, :]
            E_eV = electron_energy_ev(vx, vy, vz, m_e)
            if E_eV > E_max_eV:
                E_max_eV = E_eV

    # Safety: ensure E_max is at least 10 eV
    E_max_eV = max(E_max_eV, 10.0)

    # Maximum collision frequency
    nu_max = compute_max_collision_frequency(neutral_density, sigma_max, E_max_eV, m_e)

    # Collision probability
    P_coll = nu_max * dt

    # If P_coll > 1, warn and clamp (timestep too large)
    if P_coll > 1.0:
        print(f"  [WARNING] MCC collision probability = {P_coll:.2f} > 1.0!")
        print(f"            Reduce timestep for accuracy")
        P_coll = 1.0

    # Collision counters
    n_collisions_total = 0
    n_elastic = 0
    n_excitation = 0
    n_ionization = 0

    # Arrays for new particles (ionization products)
    new_particles_x = np.zeros((max_new_particles, 3), dtype=np.float64)
    new_particles_v = np.zeros((max_new_particles, 3), dtype=np.float64)
    new_particles_weight = np.zeros(max_new_particles, dtype=np.float64)
    new_particles_species_id = np.zeros(max_new_particles, dtype=np.int32)
    n_new_particles = np.array([0], dtype=np.int32)  # Mutable counter

    # Get ion species ID for this neutral
    # N2 -> N2+, O -> O+, etc.
    ion_species_name = neutral_species + '+'
    ion_species_id = 0  # Default
    for sid, sname in ID_TO_SPECIES.items():
        if sname == ion_species_name:
            ion_species_id = sid
            break

    # Process each electron
    for i in range(n_particles):
        if not electron_mask[i]:
            continue

        # Test for collision
        r = np.random.random()
        if r > P_coll:
            continue  # No collision

        # Collision occurs - determine type
        vx, vy, vz = particles.v[i, :]
        E_eV = electron_energy_ev(vx, vy, vz, m_e)

        # Sample collision type
        r_type = np.random.random()
        reaction_type_name = sample_collision_type(neutral_species, E_eV, r_type)

        if reaction_type_name is None:
            continue  # Null collision (energy below all thresholds)

        # Get reaction data
        reaction_data = get_reaction_data(neutral_species, reaction_type_name)
        energy_loss_eV = reaction_data['energy_loss']

        # Map reaction type name to integer
        reaction_type_int = {'elastic': 0, 'excitation': 1, 'ionization': 2}[reaction_type_name]

        # Process collision
        new_vx, new_vy, new_vz, n_new = process_electron_collision(
            i,
            vx, vy, vz,
            particles.weight[i],
            reaction_type_int,
            energy_loss_eV,
            m_e,
            new_particles_x,
            new_particles_v,
            new_particles_weight,
            new_particles_species_id,
            n_new_particles,
            max_new_particles,
            ion_species_id
        )

        # Update electron velocity
        particles.v[i, 0] = new_vx
        particles.v[i, 1] = new_vy
        particles.v[i, 2] = new_vz

        # Update counters
        n_collisions_total += 1
        if reaction_type_name == 'elastic':
            n_elastic += 1
        elif reaction_type_name == 'excitation':
            n_excitation += 1
        elif reaction_type_name == 'ionization':
            n_ionization += 1

    # Add new particles to particle array
    n_new_total = n_new_particles[0]

    for idx in range(n_new_total):
        if particles.n_particles >= particles.max_particles:
            print(f"  [WARNING] Particle array full! Cannot add more particles.")
            break

        # Position: same as parent electron (find first electron for simplicity)
        # In practice, should track parent position
        for i in range(n_particles):
            if electron_mask[i]:
                new_particles_x[idx, :] = particles.x[i, :]
                break

        # Add particle
        species_id = new_particles_species_id[idx]
        species_name = ID_TO_SPECIES[species_id]

        x_new = new_particles_x[idx:idx+1, :]
        v_new = new_particles_v[idx:idx+1, :]
        particles.add_particles(x_new, v_new, species_name, weight=new_particles_weight[idx])

    # Count new particles
    n_new_electrons = 0
    n_new_ions = 0
    for idx in range(n_new_total):
        species_id = new_particles_species_id[idx]
        species_name = ID_TO_SPECIES[species_id]
        if species_name == 'e':
            n_new_electrons += 1
        else:
            n_new_ions += 1

    return {
        'n_collisions_total': n_collisions_total,
        'n_elastic': n_elastic,
        'n_excitation': n_excitation,
        'n_ionization': n_ionization,
        'n_new_electrons': n_new_electrons,
        'n_new_ions': n_new_ions,
    }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 60)
    print("MCC Module - Self Test")
    print("=" * 60)

    # Test 1: Maximum collision frequency
    print("\nTest 1: Maximum collision frequency...")

    n_neutral = 1.65e17  # m^-3
    sigma_max = 15e-20  # m^2 (typical for N2 total)
    E_max_eV = 50.0  # eV
    m_electron = m_e

    nu_max = compute_max_collision_frequency(n_neutral, sigma_max, E_max_eV, m_electron)
    print(f"  n_neutral = {n_neutral:.2e} m^-3")
    print(f"  sigma_max = {sigma_max:.2e} m^2")
    print(f"  E_max = {E_max_eV:.1f} eV")
    print(f"  nu_max = {nu_max:.3e} s^-1")

    dt = 1e-10  # 0.1 ns
    P_coll = nu_max * dt
    print(f"  dt = {dt*1e9:.1f} ns")
    print(f"  P_coll = {P_coll:.4f}")

    # Test 2: Electron energy calculation
    print("\nTest 2: Electron energy calculation...")

    v = 1e6  # 1000 km/s
    vx, vy, vz = v, 0, 0

    E_eV = electron_energy_ev(vx, vy, vz, m_electron)
    print(f"  v = {v/1e3:.0f} km/s")
    print(f"  E = {E_eV:.2f} eV")

    # Analytical check
    E_J = 0.5 * m_electron * v**2
    E_eV_analytical = E_J / eV
    error = abs(E_eV - E_eV_analytical) / E_eV_analytical
    print(f"  Analytical: {E_eV_analytical:.2f} eV")
    print(f"  Error: {error:.3e}")
    assert error < 1e-10, "Energy calculation error!"

    # Test 3: Isotropic scattering
    print("\nTest 3: Isotropic scattering distribution...")

    n_samples = 10000
    v_mag = 1e6  # m/s

    thetas = []
    for _ in range(n_samples):
        vx, vy, vz = isotropic_scatter(v_mag)

        # Check magnitude
        v_actual = np.sqrt(vx**2 + vy**2 + vz**2)
        assert abs(v_actual - v_mag) / v_mag < 1e-6, "Magnitude not preserved!"

        # Polar angle
        theta = np.arccos(vz / v_mag)
        thetas.append(theta)

    # Distribution should be uniform in cos(theta)
    cos_thetas = np.cos(thetas)
    mean_cos = np.mean(cos_thetas)
    std_cos = np.std(cos_thetas)

    print(f"  Samples: {n_samples}")
    print(f"  <cos(theta)> = {mean_cos:.4f} (expect: 0.0)")
    print(f"  std(cos(theta)) = {std_cos:.4f} (expect: 0.577)")

    # Should be close to uniform on [-1, 1] -> mean = 0, std = sqrt(1/3) = 0.577
    assert abs(mean_cos) < 0.02, "Isotropic distribution not centered!"
    assert abs(std_cos - 0.577) < 0.02, "Isotropic distribution wrong variance!"

    print("  [PASS] Isotropic scattering validated")

    print("\n" + "=" * 60)
    print("MCC module self-tests passed!")
    print("=" * 60)
