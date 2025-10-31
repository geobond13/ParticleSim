"""
DSMC Surface Interaction Module

Implements:
- Cercignani-Lampis-Lord (CLL) gas-surface reflection model
- Catalytic recombination (O + O → O₂)
- Temperature-dependent accommodation coefficients
- Atomic oxygen fluence tracking for surface aging

References:
- Lord (1991), "Some Extensions to the Cercignani-Lampis Gas-Surface Scattering Kernel"
- Bird (1994), "Molecular Gas Dynamics", Ch. 11
- Cacciatore & Rutigliano (2009), "Catalytic Recombination on Spacecraft Materials"

Week 3 Deliverable - IntakeSIM
"""

import numpy as np
from numba import njit
import math

from ..constants import kB


# ==================== CLL SURFACE REFLECTION MODEL ====================

@njit
def cll_reflect_particle(v_incident, v_wall, m, T_wall, alpha_n, alpha_t):
    """
    Cercignani-Lampis-Lord (CLL) gas-surface reflection model.

    **LEGACY FUNCTION - ASSUMES Z-AXIS WALL NORMAL**
    For multi-channel honeycomb geometries with arbitrary wall orientations,
    use cll_reflect_particle_general() instead.

    The CLL model provides independent accommodation of normal and tangential
    momentum components, making it more realistic than simple Maxwell
    (fully diffuse) or specular models.

    Parameters:
    -----------
    v_incident : ndarray (3,)
        Incident velocity in lab frame [m/s]
    v_wall : ndarray (3,)
        Wall velocity (usually zero for stationary wall) [m/s]
    m : float
        Particle mass [kg]
    T_wall : float
        Wall temperature [K]
    alpha_n : float
        Normal accommodation coefficient [0, 1]
        0 = specular, 1 = fully diffuse
    alpha_t : float
        Tangential accommodation coefficient [0, 1]
        0 = no tangential accommodation, 1 = full accommodation

    Returns:
    --------
    v_reflected : ndarray (3,)
        Reflected velocity in lab frame [m/s]

    Notes:
    ------
    The CLL model is implemented following Lord (1991):

    Normal component:
    - Specular part: v_n,spec = -v_n,incident
    - Diffuse part: sampled from half-Maxwellian at T_wall
    - Mixture: v_n,refl = sqrt(1 - α_n) * v_n,spec + sqrt(α_n) * v_n,diff

    Tangential component:
    - Similar mixture but with retention of incident tangential momentum
    - v_t,refl = sqrt(1 - α_t) * v_t,incident + sqrt(α_t) * v_t,diff

    For fully diffuse reflection (α_n = α_t = 1), this reduces to Maxwell model.
    For specular reflection (α_n = α_t = 0), this gives perfect reflection.
    """
    # Relative velocity (particle - wall)
    v_rel = v_incident - v_wall

    # Most common case: wall at rest (v_wall = 0), so v_rel = v_incident
    # But we keep general form for moving walls

    # Assume wall normal is in +z direction (z = 0 plane, normal = [0, 0, 1])
    # For general wall orientations, would need to transform to wall frame

    # Normal component (z-direction, assuming wall normal = +z)
    v_n_incident = v_rel[2]

    # Tangential components
    v_t_incident = np.array([v_rel[0], v_rel[1]], dtype=np.float64)

    # Thermal velocity scale at wall temperature
    v_thermal = math.sqrt(2.0 * kB * T_wall / m)

    # ========== NORMAL COMPONENT (CLL model) ==========

    # For particles incident from above (v_n < 0), reflect
    # Note: In DSMC, we only reflect particles hitting the wall (v_n < 0)
    if v_n_incident >= 0:
        # Particle moving away from wall - no reflection
        return v_incident

    # Specular reflection: v_n_spec = -v_n_incident
    v_n_spec = -v_n_incident

    # Diffuse component: sample from half-Maxwellian
    # For diffuse reflection, v_n follows distribution:
    # f(v_n) ∝ v_n * exp(-v_n^2 / v_thermal^2) for v_n > 0
    # Sampling: v_n = v_thermal * sqrt(-log(random))
    u1 = np.random.rand()
    v_n_diff = v_thermal * math.sqrt(-math.log(u1))

    # CLL mixture for normal component
    v_n_reflected = math.sqrt(1.0 - alpha_n) * v_n_spec + math.sqrt(alpha_n) * v_n_diff

    # ========== TANGENTIAL COMPONENTS (CLL model) ==========

    # For each tangential direction (x, y)
    v_t_reflected = np.zeros(2, dtype=np.float64)

    for i in range(2):
        # Specular: v_t_spec = v_t_incident (no change in tangential)
        v_t_spec = v_t_incident[i]

        # Diffuse component: sample from Maxwellian (can be positive or negative)
        # f(v_t) ∝ exp(-v_t^2 / v_thermal^2)
        # Sampling: v_t = v_thermal * randn() (Box-Muller or similar)
        u2 = np.random.rand()
        u3 = np.random.rand()
        v_t_diff = v_thermal * math.sqrt(-2.0 * math.log(u2)) * math.cos(2.0 * math.pi * u3)

        # CLL mixture for tangential component
        v_t_reflected[i] = math.sqrt(1.0 - alpha_t) * v_t_spec + math.sqrt(alpha_t) * v_t_diff

    # ========== ASSEMBLE REFLECTED VELOCITY ==========

    v_reflected = np.array([
        v_t_reflected[0],
        v_t_reflected[1],
        v_n_reflected
    ], dtype=np.float64)

    # Add wall velocity back (for moving walls)
    v_reflected += v_wall

    return v_reflected


@njit
def cll_reflect_particle_general(v_incident, v_wall, m, T_wall, alpha_n, alpha_t, wall_normal):
    """
    Generalized Cercignani-Lampis-Lord (CLL) reflection for arbitrary wall orientations.

    This extends the CLL model to work with any wall normal direction, not just the z-axis.
    Critical for multi-channel honeycomb geometries where each channel has a radially
    outward normal.

    Parameters:
    -----------
    v_incident : ndarray (3,)
        Incident velocity in lab frame [m/s]
    v_wall : ndarray (3,)
        Wall velocity (usually zero for stationary wall) [m/s]
    m : float
        Particle mass [kg]
    T_wall : float
        Wall temperature [K]
    alpha_n : float
        Normal accommodation coefficient [0, 1]
        0 = specular, 1 = fully diffuse
    alpha_t : float
        Tangential accommodation coefficient [0, 1]
        0 = no tangential accommodation, 1 = full accommodation
    wall_normal : ndarray (3,)
        Unit normal vector pointing INTO DOMAIN (away from wall)
        For cylindrical channels: radially outward from centerline

    Returns:
    --------
    v_reflected : ndarray (3,)
        Reflected velocity in lab frame [m/s]

    Notes:
    ------
    Algorithm:
    1. Compute relative velocity: v_rel = v_incident - v_wall
    2. Decompose into normal and tangential components using dot product:
       - v_n = (v_rel · n)  [scalar magnitude]
       - v_t = v_rel - v_n * n  [vector]
    3. Apply CLL sampling (same physics as cll_reflect_particle):
       - Normal: mixture of specular and half-Maxwellian
       - Tangential: mixture of incident and Maxwellian
    4. Reconstruct: v_reflected = v_n_refl * n + v_t_refl + v_wall

    The key difference from cll_reflect_particle() is the use of vector projection
    instead of hardcoded [x,y,z] indexing, allowing arbitrary wall orientations.

    Reference: Lord (1991), Bird (1994) Ch. 11
    Week 2 Deliverable - IntakeSIM Multi-Channel Geometry Phase II
    """
    # Relative velocity (particle - wall)
    v_rel = v_incident - v_wall

    # ========== DECOMPOSE INTO NORMAL AND TANGENTIAL ==========

    # Normal component (scalar): v_n = v_rel · wall_normal
    v_n_incident = v_rel[0]*wall_normal[0] + v_rel[1]*wall_normal[1] + v_rel[2]*wall_normal[2]

    # Check if particle moving away from wall
    if v_n_incident >= 0:
        return v_incident  # No reflection needed

    # Tangential velocity vector: v_t = v_rel - (v_rel · n) * n
    # This projects v_rel onto the plane perpendicular to wall_normal
    v_t_incident = np.array([
        v_rel[0] - v_n_incident * wall_normal[0],
        v_rel[1] - v_n_incident * wall_normal[1],
        v_rel[2] - v_n_incident * wall_normal[2]
    ], dtype=np.float64)

    # Thermal velocity scale at wall temperature
    v_thermal = math.sqrt(2.0 * kB * T_wall / m)

    # ========== NORMAL COMPONENT (CLL model) ==========

    # Specular reflection: reverse normal component
    v_n_spec = -v_n_incident

    # Diffuse component: half-Maxwellian (always positive, leaving wall)
    # f(v_n) ∝ v_n * exp(-v_n^2 / v_thermal^2) for v_n > 0
    # Sampling: v_n = v_thermal * sqrt(-log(random))
    u1 = np.random.rand()
    v_n_diff = v_thermal * math.sqrt(-math.log(u1))

    # CLL mixture for normal component
    v_n_reflected = math.sqrt(1.0 - alpha_n) * v_n_spec + math.sqrt(alpha_n) * v_n_diff

    # ========== TANGENTIAL COMPONENT (CLL model) ==========

    # For tangential, we need to sample a 3D Gaussian velocity and project it
    # onto the tangent plane (perpendicular to wall_normal)

    # Sample 3D Maxwellian velocity components
    # Component 1
    u2 = np.random.rand()
    u3 = np.random.rand()
    v_diff_1 = v_thermal * math.sqrt(-2.0 * math.log(u2)) * math.cos(2.0 * math.pi * u3)

    # Component 2
    u4 = np.random.rand()
    u5 = np.random.rand()
    v_diff_2 = v_thermal * math.sqrt(-2.0 * math.log(u4)) * math.cos(2.0 * math.pi * u5)

    # Component 3
    u6 = np.random.rand()
    u7 = np.random.rand()
    v_diff_3 = v_thermal * math.sqrt(-2.0 * math.log(u6)) * math.cos(2.0 * math.pi * u7)

    # Construct diffuse velocity vector
    v_diff_3d = np.array([v_diff_1, v_diff_2, v_diff_3], dtype=np.float64)

    # Project to tangent plane: v_t_diff = v_diff_3d - (v_diff_3d · n) * n
    v_diff_dot_n = v_diff_3d[0]*wall_normal[0] + v_diff_3d[1]*wall_normal[1] + v_diff_3d[2]*wall_normal[2]
    v_t_diff = np.array([
        v_diff_3d[0] - v_diff_dot_n * wall_normal[0],
        v_diff_3d[1] - v_diff_dot_n * wall_normal[1],
        v_diff_3d[2] - v_diff_dot_n * wall_normal[2]
    ], dtype=np.float64)

    # CLL mixture for tangential component (element-wise)
    v_t_reflected = np.array([
        math.sqrt(1.0 - alpha_t) * v_t_incident[0] + math.sqrt(alpha_t) * v_t_diff[0],
        math.sqrt(1.0 - alpha_t) * v_t_incident[1] + math.sqrt(alpha_t) * v_t_diff[1],
        math.sqrt(1.0 - alpha_t) * v_t_incident[2] + math.sqrt(alpha_t) * v_t_diff[2]
    ], dtype=np.float64)

    # ========== RECONSTRUCT REFLECTED VELOCITY ==========

    # v_reflected = v_n_reflected * wall_normal + v_t_reflected + v_wall
    v_reflected = np.array([
        v_n_reflected * wall_normal[0] + v_t_reflected[0] + v_wall[0],
        v_n_reflected * wall_normal[1] + v_t_reflected[1] + v_wall[1],
        v_n_reflected * wall_normal[2] + v_t_reflected[2] + v_wall[2]
    ], dtype=np.float64)

    return v_reflected


@njit
def apply_cll_surface_bc(x, v, active, m, wall_position, wall_normal,
                          T_wall, alpha_n, alpha_t, n_particles):
    """
    Apply CLL surface boundary conditions to particles hitting a wall.

    This function checks which particles have crossed the wall boundary
    and reflects them using the CLL model.

    Parameters:
    -----------
    x, v : ndarray (n_particles, 3)
        Position and velocity arrays
    active : ndarray (n_particles,)
        Active particle flags
    m : float
        Particle mass [kg]
    wall_position : float
        Wall position coordinate (e.g., z = 0) [m]
    wall_normal : ndarray (3,)
        Wall normal vector (unit vector pointing into domain)
    T_wall : float
        Wall temperature [K]
    alpha_n, alpha_t : float
        Normal and tangential accommodation coefficients
    n_particles : int
        Number of particles

    Returns:
    --------
    n_reflected : int
        Number of particles reflected

    Notes:
    ------
    This function assumes a planar wall perpendicular to one coordinate axis.
    For complex geometries, would need ray-tracing to find wall intersections.
    """
    n_reflected = 0
    v_wall = np.zeros(3, dtype=np.float64)  # Stationary wall

    # Determine which coordinate axis is the wall normal
    # Assume wall_normal is [0, 0, 1] for simplicity (z-wall)
    # For general case, would need to transform coordinates

    for i in range(n_particles):
        if not active[i]:
            continue

        # Check if particle crossed wall (z < wall_position for wall at z=0 with normal +z)
        if x[i, 2] < wall_position:
            # Particle has crossed wall - reflect it

            # Move particle back to wall surface
            x[i, 2] = wall_position

            # Reflect velocity using CLL model
            v_reflected = cll_reflect_particle(
                v[i, :], v_wall, m, T_wall, alpha_n, alpha_t
            )

            v[i, :] = v_reflected
            n_reflected += 1

    return n_reflected


# ==================== SPECIAL CASES (ANALYTICAL) ====================

@njit
def specular_reflect_particle(v_incident, wall_normal):
    """
    Specular (mirror) reflection.

    This is a special case of CLL with alpha_n = alpha_t = 0.
    Included for performance when accommodation is zero.

    Parameters:
    -----------
    v_incident : ndarray (3,)
        Incident velocity [m/s]
    wall_normal : ndarray (3,)
        Wall normal vector (unit, pointing into domain)

    Returns:
    --------
    v_reflected : ndarray (3,)
        Reflected velocity [m/s]

    Notes:
    ------
    Specular reflection: v_refl = v_incident - 2 * (v_incident · n) * n
    """
    # Dot product: v · n
    v_dot_n = v_incident[0] * wall_normal[0] + \
              v_incident[1] * wall_normal[1] + \
              v_incident[2] * wall_normal[2]

    # Reflected velocity
    v_reflected = v_incident - 2.0 * v_dot_n * wall_normal

    return v_reflected


@njit
def diffuse_reflect_particle(m, T_wall):
    """
    Fully diffuse (Maxwell) reflection.

    This is a special case of CLL with alpha_n = alpha_t = 1.
    Particles are emitted from the wall with Maxwellian distribution at T_wall.

    Parameters:
    -----------
    m : float
        Particle mass [kg]
    T_wall : float
        Wall temperature [K]

    Returns:
    --------
    v_reflected : ndarray (3,)
        Reflected velocity [m/s]

    Notes:
    ------
    Normal component: v_n ~ sqrt(-log(u)) * v_thermal (half-Maxwellian, v_n > 0)
    Tangential: v_t ~ randn() * v_thermal / sqrt(2) (full Maxwellian)

    This gives a cosine distribution of reflection angles (Lambert's law).
    """
    v_thermal = math.sqrt(2.0 * kB * T_wall / m)

    # Normal component (always positive, leaving wall)
    u1 = np.random.rand()
    v_n = v_thermal * math.sqrt(-math.log(u1))

    # Tangential components (Gaussian)
    u2 = np.random.rand()
    u3 = np.random.rand()
    v_t1 = v_thermal * math.sqrt(-2.0 * math.log(u2)) * math.cos(2.0 * math.pi * u3)

    u4 = np.random.rand()
    u5 = np.random.rand()
    v_t2 = v_thermal * math.sqrt(-2.0 * math.log(u4)) * math.cos(2.0 * math.pi * u5)

    v_reflected = np.array([v_t1, v_t2, v_n], dtype=np.float64)

    return v_reflected


# ==================== CATALYTIC RECOMBINATION ====================

@njit
def catalytic_recombination_probability(T_wall, species_name_hash):
    """
    Compute recombination probability for atomic oxygen on surface.

    The recombination coefficient γ depends on:
    - Surface temperature (Arrhenius-type)
    - Surface material and oxidation state
    - Atomic oxygen fluence (surface aging)

    Parameters:
    -----------
    T_wall : float
        Surface temperature [K]
    species_name_hash : int
        Hash of species name (0 for O, others not reactive)

    Returns:
    --------
    gamma : float
        Recombination probability per collision [0, 1]

    Notes:
    ------
    For atomic oxygen recombination on oxidized surfaces:
    - γ_0 ~ 0.01-0.03 (fresh SiO2, Al2O3)
    - E_a ~ 0.1-0.2 eV (activation energy)
    - Temperature dependence: γ(T) = γ_0 * exp(-E_a / (k*T))

    For this implementation, we use a simple Arrhenius model:
    γ(T) = 0.02 * exp(-2000 K / T) for 200 K < T < 1000 K

    Reference: Cacciatore & Rutigliano (2009)
    """
    # Only atomic oxygen (species hash 0 by convention) recombines
    if species_name_hash != 0:
        return 0.0

    # Recombination parameters (for oxidized SiO2-like surface)
    gamma_0 = 0.02  # Pre-exponential factor
    E_a_over_k = 2000.0  # Activation temperature [K]

    # Arrhenius expression
    gamma = gamma_0 * math.exp(-E_a_over_k / T_wall)

    # Clamp to physical range [0, 1]
    if gamma > 1.0:
        gamma = 1.0
    if gamma < 0.0:
        gamma = 0.0

    return gamma


@njit
def attempt_catalytic_recombination(v_incident, m_O, T_wall, species_id, gamma):
    """
    Attempt catalytic recombination: O + O(surface) → O₂.

    If recombination occurs, the incident O atom combines with a surface-adsorbed
    O atom to form O₂. The product molecule is emitted with:
    - Translational energy from exothermic reaction (5.1 eV)
    - Thermal energy from wall temperature
    - Cosine angular distribution

    Parameters:
    -----------
    v_incident : ndarray (3,)
        Incident O atom velocity [m/s]
    m_O : float
        Atomic oxygen mass [kg]
    T_wall : float
        Surface temperature [K]
    species_id : int
        Species ID of incident particle
    gamma : float
        Recombination probability

    Returns:
    --------
    recombined : bool
        True if recombination occurred
    v_product : ndarray (3,)
        Product O₂ velocity [m/s] (if recombined=True)
    new_species_id : int
        Species ID of product (O₂) if recombined, else original species_id

    Notes:
    ------
    Recombination releases 5.1 eV per O₂ formed.
    This energy is partitioned between:
    - Translational (emitted O₂ molecule)
    - Vibrational/rotational (internal modes)
    - Surface phonons (lattice heating)

    Typical partition: ~30% translational, ~40% internal, ~30% to surface
    We assume 30% goes to kinetic energy of emitted O₂.
    """
    # Random test for recombination
    if np.random.rand() > gamma:
        # No recombination - return incident particle unchanged
        return False, v_incident, species_id

    # Recombination occurs!
    # Form O₂ molecule and emit it

    m_O2 = 2.0 * m_O  # O₂ mass

    # Energy release: 5.1 eV = 5.1 * 1.602e-19 J
    E_recomb = 5.1 * 1.602176634e-19  # Joules

    # Assume 30% goes to translational kinetic energy of O₂
    E_trans = 0.3 * E_recomb

    # Additional thermal energy from wall
    E_thermal = 1.5 * kB * T_wall

    # Total kinetic energy of emitted O₂
    E_kinetic_total = E_trans + E_thermal

    # Velocity magnitude
    v_mag = math.sqrt(2.0 * E_kinetic_total / m_O2)

    # Direction: cosine distribution (diffuse emission)
    # Sample angle from Lambert's law: P(θ) ∝ cos(θ) for θ ∈ [0, π/2]
    # This gives: cos(θ) = sqrt(random())
    u1 = np.random.rand()
    cos_theta = math.sqrt(u1)
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

    # Azimuthal angle (uniform)
    phi = 2.0 * math.pi * np.random.rand()

    # Velocity components (assuming wall normal is +z)
    v_product = np.array([
        v_mag * sin_theta * math.cos(phi),  # x
        v_mag * sin_theta * math.sin(phi),  # y
        v_mag * cos_theta                    # z (normal to wall)
    ], dtype=np.float64)

    # Species ID for O₂ (assume O₂ is species 2 in our database)
    new_species_id = 2  # TODO: Make this configurable

    return True, v_product, new_species_id


# ==================== ENERGY ACCOMMODATION ====================

@njit
def compute_energy_accommodation(v_incident, v_reflected, m, T_wall):
    """
    Compute energy accommodation coefficient from incident/reflected velocities.

    The energy accommodation coefficient is defined as:
        α_E = (E_i - E_r) / (E_i - E_wall)

    where:
    - E_i = incident kinetic energy in wall frame
    - E_r = reflected kinetic energy in wall frame
    - E_wall = energy corresponding to wall temperature (3/2 * k * T_wall)

    Parameters:
    -----------
    v_incident, v_reflected : ndarray (3,)
        Incident and reflected velocities [m/s]
    m : float
        Particle mass [kg]
    T_wall : float
        Wall temperature [K]

    Returns:
    --------
    alpha_E : float
        Energy accommodation coefficient [0, 1]

    Notes:
    ------
    α_E = 0: No energy exchange (specular reflection)
    α_E = 1: Full thermal accommodation
    """
    # Kinetic energies
    E_i = 0.5 * m * np.dot(v_incident, v_incident)
    E_r = 0.5 * m * np.dot(v_reflected, v_reflected)

    # Energy corresponding to wall temperature
    E_wall = 1.5 * kB * T_wall

    # Accommodation coefficient
    if E_i > E_wall:
        alpha_E = (E_i - E_r) / (E_i - E_wall)
    else:
        # Particle cooler than wall - use different definition
        alpha_E = (E_r - E_i) / (E_wall - E_i)

    # Clamp to [0, 1]
    if alpha_E > 1.0:
        alpha_E = 1.0
    if alpha_E < 0.0:
        alpha_E = 0.0

    return alpha_E
