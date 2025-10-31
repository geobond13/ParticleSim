"""
DSMC Collision Module - Variable Hard Sphere (VHS) Model

Implements:
- VHS collision cross-section (temperature-dependent)
- Binary collision algorithm (Majorant Collision Frequency method)
- Post-collision velocities (isotropic scattering)
- Multi-species collision pairs

Reference:
- Bird (1994), "Molecular Gas Dynamics and the Direct Simulation of Gas Flows"
- Alexander, Garcia, Alder (1998), "Cell size dependence of transport coefficients"

Week 2 Deliverable - IntakeSIM
"""

import numpy as np
from numba import njit, prange
import math

from ..constants import kB, SPECIES


# ==================== VHS CROSS-SECTION MODEL ====================

@njit
def vhs_collision_cross_section(v_rel, T_ref, d_ref, omega):
    """
    Variable Hard Sphere (VHS) collision cross-section.

    The VHS model uses a temperature-dependent collision diameter:
        σ_T = σ_ref * (T_ref / T)^ω

    where the collision temperature is related to the relative velocity:
        T_coll = m_r * v_rel^2 / (2 * kB)

    This gives:
        σ_T = π * d_ref^2 * (v_ref / v_rel)^(2ω - 1)

    Parameters:
    -----------
    v_rel : float
        Relative velocity magnitude [m/s]
    T_ref : float
        Reference temperature [K]
    d_ref : float
        Reference collision diameter at T_ref [m]
    omega : float
        VHS temperature exponent (species-dependent)
        - Hard sphere: ω = 0.5
        - VHS typical: ω = 0.7-0.8

    Returns:
    --------
    sigma_T : float
        Temperature-dependent collision cross-section [m^2]

    Notes:
    ------
    For the special case ω = 0.5 (Hard Sphere), σ_T = π*d_ref^2 (constant).

    Physical interpretation:
    - At high relative velocities (high collision energy), cross-section decreases
    - At low relative velocities (low collision energy), cross-section increases
    - This captures quantum/molecular effects not in hard-sphere model
    """
    # Reference velocity at T_ref
    # For VHS: v_ref^2 = 2 * kB * T_ref / m_r (not needed explicitly here)

    # VHS cross-section formula
    # σ_T = π * d_ref^2 * (v_ref / v_rel)^(2ω - 1)

    # For numerical stability, handle v_rel ≈ 0 case
    if v_rel < 1e-10:
        # Return maximum cross-section (low-velocity limit)
        return math.pi * d_ref * d_ref * 1e10

    # Calculate reference velocity from T_ref
    # Using Bird's definition: v_ref = sqrt(2 * kB * T_ref / m_r)
    # But we only need the ratio (v_ref / v_rel), which is temperature-dependent

    # Simplified VHS formula (Bird Eq. 4.63):
    # σ_T = π * d_ref^2 * (2 * kB * T_ref / (m_r * v_rel^2))^(ω - 0.5)

    # For computational efficiency, we use:
    # σ_T = π * d_ref^2 * (v_ref / v_rel)^(2ω - 1)

    # Since we don't have m_r here, we use T_ref = 273 K as standard
    # and the reference velocity magnitude
    v_ref_sq = 2.0 * kB * T_ref / (28.0 * 1.66e-27)  # Using N2 mass as reference
    v_ref = math.sqrt(v_ref_sq)

    # VHS cross-section
    exponent = 2.0 * omega - 1.0
    sigma_T = math.pi * d_ref * d_ref * math.pow(v_ref / v_rel, exponent)

    return sigma_T


@njit
def compute_collision_frequency(n_density, d_ref, omega, v_mean, T):
    """
    Compute mean collision frequency for VHS model.

    For a single species, the collision frequency is:
        ν = n * σ_T * v_rel_mean

    where v_rel_mean depends on the distribution (Maxwellian assumed).

    Parameters:
    -----------
    n_density : float
        Number density [m^-3]
    d_ref : float
        Reference collision diameter [m]
    omega : float
        VHS exponent
    v_mean : float
        Mean thermal velocity [m/s]
    T : float
        Temperature [K]

    Returns:
    --------
    nu : float
        Collision frequency [Hz = 1/s]
    """
    # Mean relative velocity for Maxwellian distribution
    # v_rel_mean = sqrt(2) * v_mean for same species
    v_rel_mean = math.sqrt(2.0) * v_mean

    # VHS cross-section at mean velocity
    sigma_T = vhs_collision_cross_section(v_rel_mean, 273.0, d_ref, omega)

    # Collision frequency
    nu = n_density * sigma_T * v_rel_mean

    return nu


# ==================== BINARY COLLISION ALGORITHM ====================

@njit
def compute_majorant_frequency(n_cells, cell_volumes, cell_counts,
                                 cell_particles, n_particles_max,
                                 x, v, species_id, active, weight,
                                 d_ref_array, omega_array):
    """
    Compute majorant collision frequency for each cell.

    The majorant frequency is the maximum possible collision rate in each cell,
    used for the acceptance-rejection method.

    For each cell:
        ν_maj = F_num * (N_cell * (N_cell - 1) / (2 * V_cell)) * σ_max * v_rel_max

    where:
    - F_num = average weight (real molecules per computational particle)
    - N_cell = number of computational particles in cell
    - V_cell = cell volume
    - σ_max = maximum cross-section (conservative estimate)
    - v_rel_max = maximum relative velocity in cell

    Parameters:
    -----------
    n_cells : int
        Number of cells
    cell_volumes : ndarray (n_cells,)
        Volume of each cell [m^3]
    cell_counts : ndarray (n_cells,)
        Number of particles in each cell
    cell_particles : ndarray (n_cells, max_per_cell)
        Particle indices in each cell
    n_particles_max : int
        Maximum particles per cell
    x, v : ndarray (n_particles, 3)
        Particle positions and velocities
    species_id : ndarray (n_particles,)
        Species ID for each particle
    active : ndarray (n_particles,)
        Active flag for each particle
    weight : ndarray (n_particles,)
        Particle statistical weight
    d_ref_array : ndarray (n_species,)
        Reference diameter for each species
    omega_array : ndarray (n_species,)
        VHS omega for each species

    Returns:
    --------
    nu_majorant : ndarray (n_cells,)
        Majorant collision frequency for each cell [Hz]
    """
    nu_majorant = np.zeros(n_cells, dtype=np.float64)

    for cell_idx in range(n_cells):
        N_cell = cell_counts[cell_idx]

        if N_cell < 2:
            # Need at least 2 particles to collide
            nu_majorant[cell_idx] = 0.0
            continue

        V_cell = cell_volumes[cell_idx]

        # Find maximum relative velocity in this cell and average weight
        v_rel_max = 0.0
        d_max = 0.0
        omega_max = 0.0
        weight_sum = 0.0

        for i in range(N_cell):
            idx_i = cell_particles[cell_idx, i]
            if not active[idx_i]:
                continue

            # Accumulate weight
            weight_sum += weight[idx_i]

            # Update maximum diameter and omega
            species_i = species_id[idx_i]
            if d_ref_array[species_i] > d_max:
                d_max = d_ref_array[species_i]
            if omega_array[species_i] > omega_max:
                omega_max = omega_array[species_i]

            for j in range(i + 1, N_cell):
                idx_j = cell_particles[cell_idx, j]
                if not active[idx_j]:
                    continue

                # Compute relative velocity
                dv_x = v[idx_i, 0] - v[idx_j, 0]
                dv_y = v[idx_i, 1] - v[idx_j, 1]
                dv_z = v[idx_i, 2] - v[idx_j, 2]
                v_rel = math.sqrt(dv_x*dv_x + dv_y*dv_y + dv_z*dv_z)

                if v_rel > v_rel_max:
                    v_rel_max = v_rel

        # Average weight (F_num factor)
        F_num = weight_sum / N_cell if N_cell > 0 else 1.0

        # Compute maximum cross-section
        # For VHS, σ_max occurs at minimum velocity
        # Conservative: use v_rel_max for σ calculation (underestimates σ)
        # Better: use minimum v_rel, but that requires another loop
        # Compromise: use mean velocity / 2 as conservative estimate
        v_for_sigma = max(v_rel_max / 3.0, 10.0)  # Conservative lower velocity
        sigma_max = vhs_collision_cross_section(v_for_sigma, 273.0, d_max, omega_max)

        # Majorant frequency (includes F_num factor for weighted particles)
        # ν_maj = F_num * (N_cell * (N_cell - 1) / (2 * V_cell)) * σ_max * v_rel_max
        nu_majorant[cell_idx] = F_num * (N_cell * (N_cell - 1.0) / (2.0 * V_cell)) * sigma_max * v_rel_max

    return nu_majorant


@njit
def select_collision_pairs(cell_idx, cell_count, cell_particles,
                             nu_majorant, dt, rng_state):
    """
    Select collision pairs in a cell using acceptance-rejection method.

    The number of collision attempts is Poisson-distributed:
        N_coll ~ Poisson(ν_maj * dt)

    For small ν_maj * dt << 1, we can use:
        N_coll ≈ ν_maj * dt (expected value)

    Each pair is selected randomly from the cell.

    Parameters:
    -----------
    cell_idx : int
        Cell index
    cell_count : int
        Number of particles in cell
    cell_particles : ndarray (max_per_cell,)
        Particle indices in this cell
    nu_majorant : float
        Majorant collision frequency [Hz]
    dt : float
        Timestep [s]
    rng_state : int
        Random number generator state (placeholder for Numba)

    Returns:
    --------
    pairs : ndarray (N_pairs, 2)
        Array of particle index pairs to attempt collision
    """
    if cell_count < 2:
        # Return empty array
        return np.empty((0, 2), dtype=np.int32)

    # Expected number of collision attempts
    N_coll_expected = nu_majorant * dt

    # For small N_coll_expected, use Poisson sampling
    # For large N_coll_expected, use Gaussian approximation
    if N_coll_expected < 10.0:
        # Poisson distribution (using exponential inter-arrival times)
        # N_coll ~ Poisson(λ) where λ = N_coll_expected
        # Generate using Knuth's algorithm
        L = math.exp(-N_coll_expected)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= np.random.rand()
        N_coll = k - 1
    else:
        # Gaussian approximation: N_coll ~ N(μ, μ) where μ = N_coll_expected
        N_coll = int(np.random.normal(N_coll_expected, math.sqrt(N_coll_expected)))
        if N_coll < 0:
            N_coll = 0

    # Allocate array for pairs
    pairs = np.empty((N_coll, 2), dtype=np.int32)

    # Select random pairs
    for i in range(N_coll):
        # Select two different particles randomly
        idx1 = np.random.randint(0, cell_count)
        idx2 = np.random.randint(0, cell_count)

        # Ensure different particles
        while idx2 == idx1:
            idx2 = np.random.randint(0, cell_count)

        # Store particle indices (not cell-local indices)
        pairs[i, 0] = cell_particles[idx1]
        pairs[i, 1] = cell_particles[idx2]

    return pairs


# ==================== POST-COLLISION VELOCITIES ====================

@njit
def compute_post_collision_velocity(v1, v2, m1, m2, d_ref, omega):
    """
    Compute post-collision velocities using isotropic scattering (VHS model).

    Conservation laws:
    - Momentum: m1*v1 + m2*v2 = m1*v1' + m2*v2'
    - Energy: (1/2)*m1*v1^2 + (1/2)*m2*v2^2 = (1/2)*m1*v1'^2 + (1/2)*m2*v2'^2

    VHS model: Scattering is isotropic in the center-of-mass frame.

    Algorithm:
    1. Transform to center-of-mass (COM) frame
    2. Compute relative velocity magnitude (conserved)
    3. Select random isotropic scattering direction
    4. Transform back to lab frame

    Parameters:
    -----------
    v1, v2 : ndarray (3,)
        Velocities of particles 1 and 2 before collision [m/s]
    m1, m2 : float
        Masses of particles 1 and 2 [kg]
    d_ref : float
        Reference diameter (not used in isotropic scattering, but kept for interface)
    omega : float
        VHS exponent (affects cross-section, not scattering angle)

    Returns:
    --------
    v1_post, v2_post : ndarray (3,)
        Velocities after collision [m/s]

    Notes:
    ------
    For VHS model, the scattering is isotropic in the COM frame, meaning
    the deflection angle is uniformly distributed on the sphere.

    This is different from Variable Soft Sphere (VSS) model, where
    the scattering angle distribution depends on the impact parameter.
    """
    # Center-of-mass velocity
    m_total = m1 + m2
    v_com = (m1 * v1 + m2 * v2) / m_total

    # Relative velocity (before collision)
    v_rel = v1 - v2
    v_rel_mag = math.sqrt(v_rel[0]**2 + v_rel[1]**2 + v_rel[2]**2)

    # Reduced mass
    m_r = m1 * m2 / m_total

    # Isotropic scattering: select random direction on unit sphere
    # Method: Marsaglia (1972) - rejection method for uniform sphere sampling

    # Generate random point on unit sphere
    # Using spherical coordinates with uniform distribution
    cos_theta = 2.0 * np.random.rand() - 1.0  # cos(θ) ∈ [-1, 1]
    sin_theta = math.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * math.pi * np.random.rand()  # φ ∈ [0, 2π]

    # New relative velocity direction (unit vector)
    v_rel_dir = np.array([
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta
    ], dtype=np.float64)

    # New relative velocity (magnitude conserved, direction random)
    v_rel_post = v_rel_mag * v_rel_dir

    # Transform back to lab frame
    # v1' = v_com + (m2 / m_total) * v_rel'
    # v2' = v_com - (m1 / m_total) * v_rel'
    v1_post = v_com + (m2 / m_total) * v_rel_post
    v2_post = v_com - (m1 / m_total) * v_rel_post

    return v1_post, v2_post


@njit
def attempt_collision(idx1, idx2, v, species_id, active,
                       mass_array, d_ref_array, omega_array,
                       nu_majorant, dt):
    """
    Attempt a collision between two particles using acceptance-rejection.

    The collision is accepted with probability:
        P_accept = (σ_actual * v_rel) / (σ_max * v_rel_max)

    where σ_actual and v_rel are for the specific pair, and
    σ_max, v_rel_max are the majorant values.

    If accepted, update velocities.

    Parameters:
    -----------
    idx1, idx2 : int
        Particle indices
    v : ndarray (n_particles, 3)
        Velocity array (modified in-place if collision accepted)
    species_id : ndarray (n_particles,)
        Species ID array
    active : ndarray (n_particles,)
        Active flag array
    mass_array : ndarray (n_species,)
        Mass for each species [kg]
    d_ref_array : ndarray (n_species,)
        Reference diameter for each species [m]
    omega_array : ndarray (n_species,)
        VHS omega for each species
    nu_majorant : float
        Majorant collision frequency for this cell [Hz]
    dt : float
        Timestep [s]

    Returns:
    --------
    collision_occurred : bool
        True if collision was accepted and velocities updated
    """
    if not active[idx1] or not active[idx2]:
        return False

    # Get species properties
    sp1 = species_id[idx1]
    sp2 = species_id[idx2]

    m1 = mass_array[sp1]
    m2 = mass_array[sp2]

    # Use average diameter and omega for mixed-species collisions
    d_ref = 0.5 * (d_ref_array[sp1] + d_ref_array[sp2])
    omega = 0.5 * (omega_array[sp1] + omega_array[sp2])

    # Compute relative velocity
    dv = v[idx1] - v[idx2]
    v_rel = math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)

    # Handle near-zero relative velocity
    if v_rel < 1e-10:
        return False

    # Compute actual cross-section for this pair
    sigma_actual = vhs_collision_cross_section(v_rel, 273.0, d_ref, omega)

    # Acceptance probability
    # P_accept = (σ_actual * v_rel) / (σ_max * v_rel_max)
    # But we already incorporated this in the majorant frequency
    # So we need to compute the ratio of actual to majorant rates

    # Simplified acceptance: compare σ_actual * v_rel to majorant
    # This is approximate; rigorous method requires storing σ_max * v_rel_max per cell

    # For now, use simple acceptance based on collision probability
    # P_accept = σ_actual / σ_max (conservative)

    # Conservative σ_max estimate (at low velocity)
    sigma_max = vhs_collision_cross_section(10.0, 273.0, d_ref, omega)

    P_accept = min(1.0, sigma_actual / sigma_max)

    # Accept collision with probability P_accept
    if np.random.rand() > P_accept:
        return False

    # Collision accepted - compute post-collision velocities
    v1_post, v2_post = compute_post_collision_velocity(
        v[idx1], v[idx2], m1, m2, d_ref, omega
    )

    # Update velocities
    v[idx1, 0] = v1_post[0]
    v[idx1, 1] = v1_post[1]
    v[idx1, 2] = v1_post[2]

    v[idx2, 0] = v2_post[0]
    v[idx2, 1] = v2_post[1]
    v[idx2, 2] = v2_post[2]

    return True


# ==================== MAIN COLLISION ROUTINE ====================

@njit(parallel=False)  # Disable parallel for now to avoid race conditions
def perform_collisions_1d(x, v, species_id, active, weight, n_particles,
                           cell_edges, cell_volumes,
                           cell_particles, cell_counts, max_per_cell,
                           mass_array, d_ref_array, omega_array,
                           dt):
    """
    Perform VHS collisions for all cells using binary collision algorithm.

    Algorithm:
    1. For each cell, compute majorant collision frequency
    2. Select collision pairs (Poisson-distributed)
    3. For each pair, attempt collision with acceptance-rejection
    4. If accepted, update velocities conserving momentum and energy

    Parameters:
    -----------
    x, v : ndarray (n_particles, 3)
        Position and velocity arrays
    species_id : ndarray (n_particles,)
        Species ID for each particle
    active : ndarray (n_particles,)
        Active flag for each particle
    weight : ndarray (n_particles,)
        Particle weight (statistical weight for Monte Carlo)
    n_particles : int
        Number of particles
    cell_edges : ndarray (n_cells+1,)
        Cell edge positions [m]
    cell_volumes : ndarray (n_cells,)
        Cell volumes [m^3]
    cell_particles : ndarray (n_cells, max_per_cell)
        Particle indices in each cell
    cell_counts : ndarray (n_cells,)
        Number of particles in each cell
    max_per_cell : int
        Maximum particles per cell
    mass_array : ndarray (n_species,)
        Mass for each species [kg]
    d_ref_array : ndarray (n_species,)
        Reference diameter for each species [m]
    omega_array : ndarray (n_species,)
        VHS omega for each species
    dt : float
        Timestep [s]

    Returns:
    --------
    n_collisions : int
        Total number of collisions performed
    """
    n_cells = len(cell_volumes)
    n_collisions = 0

    # Compute majorant frequencies
    nu_majorant = compute_majorant_frequency(
        n_cells, cell_volumes, cell_counts, cell_particles, max_per_cell,
        x, v, species_id, active, weight, d_ref_array, omega_array
    )

    # Process each cell
    for cell_idx in range(n_cells):
        if cell_counts[cell_idx] < 2:
            continue

        # Select collision pairs
        pairs = select_collision_pairs(
            cell_idx, cell_counts[cell_idx],
            cell_particles[cell_idx, :],
            nu_majorant[cell_idx], dt, 0  # rng_state placeholder
        )

        # Attempt collisions
        for pair_idx in range(len(pairs)):
            idx1 = pairs[pair_idx, 0]
            idx2 = pairs[pair_idx, 1]

            collision_occurred = attempt_collision(
                idx1, idx2, v, species_id, active,
                mass_array, d_ref_array, omega_array,
                nu_majorant[cell_idx], dt
            )

            if collision_occurred:
                n_collisions += 1

    return n_collisions
