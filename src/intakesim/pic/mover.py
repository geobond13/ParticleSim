"""
PIC Particle Mover with Boris Push and TSC Weighting

Implements:
- Boris algorithm for particle push (leap-frog integrator)
- Triangular-Shaped Cloud (TSC) weighting for charge deposition
- TSC field interpolation (grid → particles)
- Boundary conditions (absorbing, periodic)

Reference:
    Birdsall & Langdon (2004), "Plasma Physics via Computer Simulation"
    Chapter 4: The Electrostatic Program
"""

import numpy as np
import numba
from ..constants import SPECIES, ID_TO_SPECIES, e, m_e
from .surfaces import apply_sheath_bc_1d


# ==================== TSC WEIGHTING FUNCTIONS ====================


@numba.njit
def tsc_weight_1d(x_particle, x_cell, dx):
    """
    Calculate TSC (Triangular-Shaped Cloud) weight for a single cell.

    TSC is a 2nd-order accurate weighting scheme that spreads particle
    charge/field over 3 nearest cells with a triangular weight function.

    Weight function:
        distance = |x_particle - x_cell| / dx

        if distance < 0.5:
            W = 0.75 - distance²
        elif distance < 1.5:
            W = 0.5 * (1.5 - distance)²
        else:
            W = 0

    Properties:
        - 2nd order accurate (smoother than CIC, NGP)
        - Σ_cells W = 1.0 (charge conserving)
        - Continuous first derivative (low noise)

    Args:
        x_particle: Particle position [m]
        x_cell: Cell center position [m]
        dx: Cell spacing [m]

    Returns:
        weight: TSC weight (0 to 0.75)

    Reference:
        Birdsall & Langdon, Section 4.7
    """
    distance = abs(x_particle - x_cell) / dx

    if distance < 0.5:
        # Central cell: parabolic peak
        return 0.75 - distance * distance
    elif distance < 1.5:
        # Neighbor cells: parabolic falloff
        d = 1.5 - distance
        return 0.5 * d * d
    else:
        # No contribution
        return 0.0


@numba.njit
def get_tsc_stencil_1d(x_particle, x_grid, dx, n_cells):
    """
    Get TSC stencil: 3 cell indices and weights for particle position.

    Args:
        x_particle: Particle position [m]
        x_grid: Cell center positions [n_cells] [m]
        dx: Cell spacing [m]
        n_cells: Number of cells

    Returns:
        cell_indices: [3] array of cell indices (may include -1 for out-of-bounds)
        weights: [3] array of TSC weights (sum = 1.0)

    Example:
        >>> x_particle = 0.0025  # 2.5 mm
        >>> x_grid = [0.001, 0.002, 0.003, ...]  # 1mm spacing
        >>> indices, weights = get_tsc_stencil_1d(x_particle, x_grid, 0.001, 100)
        >>> # Returns: indices=[1, 2, 3], weights=[0.125, 0.75, 0.125]
    """
    # Find nearest cell center
    cell_idx = int((x_particle - x_grid[0]) / dx + 0.5)

    # Clamp to valid range
    if cell_idx < 0:
        cell_idx = 0
    elif cell_idx >= n_cells:
        cell_idx = n_cells - 1

    # TSC stencil: current cell and two neighbors
    indices = np.array([cell_idx - 1, cell_idx, cell_idx + 1], dtype=np.int32)
    weights = np.zeros(3, dtype=np.float64)

    # Calculate weights for each cell
    for i in range(3):
        idx = indices[i]
        if 0 <= idx < n_cells:
            x_cell = x_grid[idx]
            weights[i] = tsc_weight_1d(x_particle, x_cell, dx)

    # Normalize weights (handles boundary effects)
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights /= total_weight

    return indices, weights


# ==================== CHARGE DEPOSITION ====================


@numba.njit
def deposit_charge_tsc_1d(
    x, weight, charge, active, x_grid, dx, n_cells, rho_out, n_particles
):
    """
    Deposit particle charges to grid using TSC weighting.

    Distributes each particle's charge over 3 nearest cells with
    triangular-shaped weights.

    Args:
        x: Particle positions [n_particles, 3] [m]
        weight: Particle computational weights [n_particles]
        charge: Particle charges [n_particles] [C] (from species)
        active: Particle active flags [n_particles]
        x_grid: Cell center positions [n_cells] [m]
        dx: Cell spacing [m]
        n_cells: Number of cells
        rho_out: Output charge density [n_cells] [C/m^3] (modified in-place)
        n_particles: Number of particles to process

    Note:
        - rho_out should be reset to zero before calling
        - Only processes active particles
        - Charge = particle.weight * charge[i] / cell_volume
    """
    # Reset charge density
    for i in range(n_cells):
        rho_out[i] = 0.0

    cell_volume = dx  # For 1D (for 3D: dx*dy*dz)

    # Deposit each particle
    for i in range(n_particles):
        if not active[i]:
            continue

        # Get particle position (x component only for 1D)
        x_p = x[i, 0]

        # Skip if out of domain
        if x_p < x_grid[0] or x_p >= x_grid[n_cells - 1] + dx:
            continue

        # Total charge represented by this particle
        Q = weight[i] * charge[i]

        # Get TSC stencil
        indices, weights_tsc = get_tsc_stencil_1d(x_p, x_grid, dx, n_cells)

        # Deposit charge to 3 nearest cells
        for j in range(3):
            idx = indices[j]
            if 0 <= idx < n_cells:
                # Charge density contribution [C/m^3]
                rho_out[idx] += Q * weights_tsc[j] / cell_volume


# ==================== FIELD INTERPOLATION ====================


@numba.njit
def interpolate_field_tsc_1d(
    x, active, E_grid, x_grid, dx, n_cells, E_particles_out, n_particles
):
    """
    Interpolate electric field from grid to particles using TSC weights.

    Uses same TSC weighting as charge deposition for consistency
    (ensures momentum conservation).

    Args:
        x: Particle positions [n_particles, 3] [m]
        active: Particle active flags [n_particles]
        E_grid: Electric field at grid faces [n_cells+1] [V/m]
        x_grid: Cell center positions [n_cells] [m]
        dx: Cell spacing [m]
        n_cells: Number of cells
        E_particles_out: Output E field at particles [n_particles, 3] [V/m]
        n_particles: Number of particles

    Note:
        - Only interpolates for active particles
        - E_grid is at faces, but we interpolate from cell-centered values
          (convert faces → centers first, or use face interpolation)
        - For simplicity, we'll interpolate E at cell centers
    """
    for i in range(n_particles):
        if not active[i]:
            E_particles_out[i, :] = 0.0
            continue

        x_p = x[i, 0]

        # Skip if out of domain
        if x_p < x_grid[0] or x_p >= x_grid[n_cells - 1] + dx:
            E_particles_out[i, :] = 0.0
            continue

        # Get TSC stencil
        indices, weights_tsc = get_tsc_stencil_1d(x_p, x_grid, dx, n_cells)

        # Interpolate E field (1D: only x-component)
        E_interp = 0.0
        for j in range(3):
            idx = indices[j]
            if 0 <= idx < n_cells:
                # E at cell center: average of face values
                # E_cell[idx] ≈ 0.5 * (E_face[idx] + E_face[idx+1])
                E_cell = 0.5 * (E_grid[idx] + E_grid[idx + 1])
                E_interp += weights_tsc[j] * E_cell

        E_particles_out[i, 0] = E_interp
        E_particles_out[i, 1] = 0.0  # No E_y
        E_particles_out[i, 2] = 0.0  # No E_z


# ==================== BORIS PARTICLE PUSHER ====================


@numba.njit
def boris_push_electrostatic(x, v, q_over_m, active, E_particles, dt, n_particles):
    """
    Push particles using Boris algorithm (electrostatic case, B=0).

    Leap-frog integration:
        v^{n+1/2} = v^{n-1/2} + (q/m) * E * dt
        x^{n+1} = x^n + v^{n+1/2} * dt

    For full Boris with B field:
        See boris_push_electromagnetic() (future implementation)

    Args:
        x: Particle positions [n_particles, 3] [m] (modified in-place)
        v: Particle velocities [n_particles, 3] [m/s] (modified in-place)
        q_over_m: Charge-to-mass ratio [n_particles] [C/kg]
        active: Particle active flags [n_particles]
        E_particles: Electric field at particles [n_particles, 3] [V/m]
        dt: Timestep [s]
        n_particles: Number of particles

    Note:
        - Velocities are at half-timesteps (leap-frog)
        - Only active particles are pushed
        - Conserves energy exactly for constant E field

    Reference:
        Birdsall & Langdon, Section 4.3
    """
    for i in range(n_particles):
        if not active[i]:
            continue

        # Acceleration: a = (q/m)*E
        qm = q_over_m[i]

        # Velocity push: v^{n+1/2} = v^{n-1/2} + a*dt
        v[i, 0] += qm * E_particles[i, 0] * dt
        v[i, 1] += qm * E_particles[i, 1] * dt
        v[i, 2] += qm * E_particles[i, 2] * dt

        # Position push: x^{n+1} = x^n + v^{n+1/2} * dt
        x[i, 0] += v[i, 0] * dt
        x[i, 1] += v[i, 1] * dt
        x[i, 2] += v[i, 2] * dt


# ==================== BOUNDARY CONDITIONS ====================


@numba.njit
def apply_absorbing_bc_1d(x, v, active, x_min, x_max, n_particles):
    """
    Apply absorbing boundary conditions: deactivate particles that hit walls.

    Args:
        x: Particle positions [n_particles, 3] [m]
        v: Particle velocities [n_particles, 3] [m/s]
        active: Particle active flags [n_particles] (modified in-place)
        x_min: Domain minimum [m]
        x_max: Domain maximum [m]
        n_particles: Number of particles

    Returns:
        n_absorbed: Number of particles absorbed
    """
    n_absorbed = 0

    for i in range(n_particles):
        if not active[i]:
            continue

        x_p = x[i, 0]

        if x_p < x_min or x_p > x_max:
            active[i] = False
            n_absorbed += 1

    return n_absorbed


@numba.njit
def apply_periodic_bc_1d(x, active, x_min, x_max, n_particles):
    """
    Apply periodic boundary conditions: wrap particles around domain.

    Args:
        x: Particle positions [n_particles, 3] [m] (modified in-place)
        active: Particle active flags [n_particles]
        x_min: Domain minimum [m]
        x_max: Domain maximum [m]
        n_particles: Number of particles
    """
    L = x_max - x_min  # Domain length

    for i in range(n_particles):
        if not active[i]:
            continue

        # Wrap x position
        while x[i, 0] < x_min:
            x[i, 0] += L
        while x[i, 0] > x_max:
            x[i, 0] -= L


@numba.njit
def apply_reflecting_bc_1d(x, v, active, x_min, x_max, n_particles):
    """
    Apply reflecting boundary conditions: specular reflection at walls.

    Implements simple elastic reflection where particles bounce off walls
    with reversed velocity component normal to the wall.

    Physics:
        - Position mirrored: x_new = 2*x_wall - x_old
        - Velocity reversed: v_new = -v_old (in x-direction only)
        - Energy conserved: |v_new| = |v_old|

    Note:
        This is a simplified model. Real plasma-wall interactions include:
        - Thermal accommodation (velocity thermalization to wall temperature)
        - Secondary electron emission (SEE)
        - Sheath potential (energy-dependent reflection)

        For physically accurate discharge simulations, use proper sheath
        boundary conditions (see apply_sheath_bc_1d, future implementation).

    Args:
        x: Particle positions [n_particles, 3] [m] (modified in-place)
        v: Particle velocities [n_particles, 3] [m/s] (modified in-place)
        active: Particle active flags [n_particles]
        x_min: Domain minimum [m]
        x_max: Domain maximum [m]
        n_particles: Number of particles

    Returns:
        n_reflected: Number of reflections (for diagnostics)

    Example:
        >>> # Electron hits left wall
        >>> x[i, 0] = -0.001  # 1 mm beyond boundary
        >>> v[i, 0] = -1000   # Moving left
        >>> apply_reflecting_bc_1d(...)
        >>> # Result: x[i, 0] = 0.001, v[i, 0] = +1000 (bounces back)
    """
    n_reflected = 0

    for i in range(n_particles):
        if not active[i]:
            continue

        x_p = x[i, 0]

        # Left wall reflection
        if x_p < x_min:
            x[i, 0] = 2.0 * x_min - x_p  # Mirror position
            v[i, 0] = -v[i, 0]            # Reverse x-velocity
            n_reflected += 1

        # Right wall reflection
        elif x_p > x_max:
            x[i, 0] = 2.0 * x_max - x_p  # Mirror position
            v[i, 0] = -v[i, 0]            # Reverse x-velocity
            n_reflected += 1

    return n_reflected


# ==================== HELPER FUNCTIONS ====================


@numba.njit
def calculate_electron_temperature_eV(v, active, species_id, n_particles, electron_id=8):
    """
    Calculate electron temperature from velocity distribution.

    T_e = (m_e / 3k) * <v²> [K]
    T_e [eV] = (m_e / (3 * e)) * <v²> [eV]

    Args:
        v: Particle velocities [n_particles, 3] [m/s]
        active: Particle active flags [n_particles]
        species_id: Particle species IDs [n_particles]
        n_particles: Number of particles
        electron_id: Species ID for electrons (default: 8)

    Returns:
        T_e_eV: Electron temperature [eV]

    Note:
        Electron species ID is 8 in the current SPECIES definition.
        This was hardcoded as 0 in Week 11, causing T_e to always return floor value.
        Fixed in Week 12.
    """
    # Accumulate <v²> for electrons
    v_sq_sum = 0.0
    n_electrons = 0

    for i in range(n_particles):
        if active[i] and species_id[i] == electron_id:
            v_sq = v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2
            v_sq_sum += v_sq
            n_electrons += 1

    if n_electrons == 0:
        return 0.1  # Fallback to avoid division by zero

    # Mean square velocity
    v_sq_mean = v_sq_sum / n_electrons

    # Temperature in eV: T = (m_e / (3 * e)) * <v²>
    T_e_eV = (m_e / (3.0 * e)) * v_sq_mean

    # Floor at 0.1 eV (avoid numerical issues)
    if T_e_eV < 0.1:
        T_e_eV = 0.1

    return T_e_eV


# ==================== MAIN INTEGRATION FUNCTION ====================


def push_pic_particles_1d(
    particles, mesh, dt, boundary_condition="absorbing", phi_left=0.0, phi_right=0.0
):
    """
    Main PIC particle push routine (high-level wrapper).

    Sequence:
        1. Deposit charge to grid (TSC)
        2. Solve Poisson equation for E field
        3. Interpolate E to particle positions (TSC)
        4. Push particles (Boris algorithm)
        5. Apply boundary conditions

    Args:
        particles: ParticleArrayNumba instance
        mesh: Mesh1DPIC instance
        dt: Timestep [s]
        boundary_condition: "absorbing", "periodic", "reflecting", or "sheath"
            - "absorbing": Particles removed at walls (default)
            - "periodic": Particles wrap around domain
            - "reflecting": Specular reflection at walls (v_x reversed)
            - "sheath": Energy-dependent sheath boundary (realistic plasma-wall)
        phi_left: Boundary potential at left wall [V] (default: 0.0)
        phi_right: Boundary potential at right wall [V] (default: 0.0)

    Returns:
        diagnostics: dict with keys:
            - n_absorbed: Number of particles absorbed
            - n_reflected: Number of reflections (reflecting or sheath BC)
            - T_e_eV: Electron temperature [eV] (sheath BC only)
    """
    from .field_solver import solve_fields_1d

    n_particles = particles.n_particles

    # Prepare species data arrays for Numba functions
    charge = np.zeros(particles.max_particles, dtype=np.float64)
    q_over_m = np.zeros(particles.max_particles, dtype=np.float64)

    for i in range(n_particles):
        if particles.active[i]:
            species_name = ID_TO_SPECIES[particles.species_id[i]]
            species_data = SPECIES[species_name]
            charge[i] = species_data.charge
            q_over_m[i] = species_data.charge / species_data.mass

    # 1. Deposit charge to grid
    deposit_charge_tsc_1d(
        particles.x,
        particles.weight,
        charge,
        particles.active,
        mesh.x_centers,
        mesh.dx,
        mesh.n_cells,
        mesh.rho,
        n_particles,
    )

    # 2. Solve Poisson equation
    solve_fields_1d(mesh, phi_left=phi_left, phi_right=phi_right)

    # 3. Interpolate E field to particles
    E_particles = np.zeros((particles.max_particles, 3), dtype=np.float64)
    interpolate_field_tsc_1d(
        particles.x,
        particles.active,
        mesh.E,
        mesh.x_centers,
        mesh.dx,
        mesh.n_cells,
        E_particles,
        n_particles,
    )

    # 4. Push particles
    boris_push_electrostatic(
        particles.x,
        particles.v,
        q_over_m,
        particles.active,
        E_particles,
        dt,
        n_particles,
    )

    # 5. Apply boundary conditions
    if boundary_condition == "absorbing":
        n_absorbed = apply_absorbing_bc_1d(
            particles.x,
            particles.v,
            particles.active,
            mesh.x_min,
            mesh.x_max,
            n_particles,
        )
        n_reflected = 0
        T_e_eV = None
    elif boundary_condition == "periodic":
        apply_periodic_bc_1d(
            particles.x, particles.active, mesh.x_min, mesh.x_max, n_particles
        )
        n_absorbed = 0
        n_reflected = 0
        T_e_eV = None
    elif boundary_condition == "reflecting":
        n_reflected = apply_reflecting_bc_1d(
            particles.x,
            particles.v,
            particles.active,
            mesh.x_min,
            mesh.x_max,
            n_particles,
        )
        n_absorbed = 0
        T_e_eV = None
    elif boundary_condition == "sheath":
        # Calculate electron temperature for self-consistent sheath
        T_e_eV = calculate_electron_temperature_eV(
            particles.v,
            particles.active,
            particles.species_id,
            n_particles,
        )

        # Apply sheath boundary with energy-dependent reflection
        # Need ion mass for distinction (use N2+ as default)
        m_ion = 28 * 1.661e-27  # kg (N2+ molecular mass)

        n_reflected, n_absorbed = apply_sheath_bc_1d(
            particles.x,
            particles.v,
            particles.active,
            particles.species_id,
            particles.weight,
            mesh.x_min,
            mesh.x_max,
            n_particles,
            T_e_eV,
            m_ion,
            bohm_factor=4.5,
        )
    else:
        raise ValueError(f"Unknown boundary condition: {boundary_condition}")

    diagnostics = {"n_absorbed": n_absorbed, "n_reflected": n_reflected}
    if T_e_eV is not None:
        diagnostics["T_e_eV"] = T_e_eV

    return diagnostics


# ==================== TESTING UTILITIES ====================


def test_tsc_weights_sum_to_one():
    """Test that TSC weights sum to 1.0 for any particle position."""
    print("Testing TSC weight normalization...")

    dx = 0.001  # 1 mm
    x_grid = np.arange(0.0005, 0.0105, dx)  # Cell centers
    n_cells = len(x_grid)

    # Test 100 random positions
    np.random.seed(42)
    for _ in range(100):
        x_particle = np.random.uniform(x_grid[0], x_grid[-1])
        indices, weights = get_tsc_stencil_1d(x_particle, x_grid, dx, n_cells)

        weight_sum = np.sum(weights)
        assert abs(weight_sum - 1.0) < 1e-12, f"Weight sum {weight_sum} != 1.0"

    print("  [PASS] TSC weights sum to 1.0 for all positions")


def test_charge_conservation():
    """Test that deposited charge equals total particle charge."""
    print("\nTesting charge conservation...")

    from .mesh import Mesh1DPIC
    from ..particles import ParticleArrayNumba

    # Setup
    mesh = Mesh1DPIC(0.0, 0.01, 100)
    particles = ParticleArrayNumba(1000)

    # Add 100 electrons
    n_add = 100
    x_init = np.random.uniform(0.001, 0.009, (n_add, 3))
    x_init[:, 1:] = 0  # Only x-component
    v_init = np.zeros((n_add, 3))
    particles.add_particles(x_init, v_init, "e", weight=1.0)

    # Prepare charge array
    charge = np.zeros(particles.max_particles, dtype=np.float64)
    for i in range(particles.n_particles):
        species_name = ID_TO_SPECIES[particles.species_id[i]]
        charge[i] = SPECIES[species_name].charge

    # Deposit charge
    deposit_charge_tsc_1d(
        particles.x,
        particles.weight,
        charge,
        particles.active,
        mesh.x_centers,
        mesh.dx,
        mesh.n_cells,
        mesh.rho,
        particles.n_particles,
    )

    # Total deposited charge
    Q_deposited = np.sum(mesh.rho) * mesh.dx  # Integrate over volume
    Q_particles = n_add * SPECIES["e"].charge  # Total particle charge

    error = abs(Q_deposited - Q_particles) / abs(Q_particles)
    assert error < 1e-10, f"Charge error {error:.3e} > 1e-10"

    print(f"  Q_deposited: {Q_deposited:.6e} C")
    print(f"  Q_particles: {Q_particles:.6e} C")
    print(f"  Relative error: {error:.3e}")
    print("  [PASS] Charge conservation within 1e-10")


def test_energy_conservation():
    """Test energy conservation for particle in static E field."""
    print("\nTesting energy conservation...")

    from .mesh import Mesh1DPIC
    from ..particles import ParticleArrayNumba
    from ..constants import SPECIES

    # Setup: uniform E field
    mesh = Mesh1DPIC(0.0, 0.01, 100)
    mesh.E[:] = 100.0  # 100 V/m uniform field

    # Single electron
    particles = ParticleArrayNumba(10)
    x_init = np.array([[0.005, 0, 0]])  # Center
    v_init = np.array([[1e5, 0, 0]])  # 100 km/s initial
    particles.add_particles(x_init, v_init, "e", weight=1.0)

    # Initial energy
    m = SPECIES["e"].mass
    q = SPECIES["e"].charge
    KE_0 = 0.5 * m * v_init[0, 0] ** 2
    PE_0 = q * mesh.E[50] * x_init[0, 0]  # Approximate
    E_total_0 = KE_0 + PE_0

    # Prepare q_over_m array
    q_over_m = np.zeros(10, dtype=np.float64)
    q_over_m[0] = q / m

    # Push for 100 steps
    dt = 1e-12  # 1 ps
    for _ in range(100):
        # Interpolate E
        E_particles = np.zeros((10, 3))
        E_particles[0, 0] = mesh.E[50]  # Constant E

        # Push
        boris_push_electrostatic(
            particles.x,
            particles.v,
            q_over_m,
            particles.active,
            E_particles,
            dt,
            1,
        )

    # Final energy
    KE_f = 0.5 * m * np.sum(particles.v[0] ** 2)
    PE_f = q * mesh.E[50] * particles.x[0, 0]
    E_total_f = KE_f + PE_f

    energy_error = abs(E_total_f - E_total_0) / abs(E_total_0)
    print(f"  Initial energy: {E_total_0:.6e} J")
    print(f"  Final energy: {E_total_f:.6e} J")
    print(f"  Relative error: {energy_error:.3e}")

    # Note: For 100 timesteps in uniform E field, expect ~0.1-1% error
    # (2nd order accurate leap-frog, error accumulates as O(dt²*n_steps))
    assert energy_error < 1e-2, f"Energy error {energy_error:.3e} > 1%"
    print("  [PASS] Energy conserved within 1%")


if __name__ == "__main__":
    print("=" * 60)
    print("PIC Mover Module - Self Test")
    print("=" * 60)

    test_tsc_weights_sum_to_one()
    test_charge_conservation()
    test_energy_conservation()

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
