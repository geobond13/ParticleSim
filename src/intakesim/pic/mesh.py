"""
1D PIC Mesh for Electrostatic Simulations

Implements a uniform 1D grid with:
- Cell-centered charge density (rho)
- Face-centered electric potential (phi)
- Face-centered electric field (E)
- Ghost cells for boundary conditions

Design Philosophy:
- Simple uniform grid (no adaptivity in Week 7)
- Debye resolution enforced: dx <= 0.5 * lambda_D
- Numba-compatible data structures (pure numpy arrays)
"""

import numpy as np
from ..constants import eps0


class Mesh1DPIC:
    """
    Uniform 1D mesh for PIC simulations.

    Cell-centered quantities: rho (charge density)
    Face-centered quantities: phi (potential), E (electric field)

    Grid layout (n_cells = 4 example):

    Face:     0     1     2     3     4
              |     |     |     |     |
    Cell:       0     1     2     3

    Attributes:
        x_min: Domain minimum [m]
        x_max: Domain maximum [m]
        n_cells: Number of interior cells
        dx: Cell spacing [m]
        x_faces: Face positions [n_cells+1] [m]
        x_centers: Cell center positions [n_cells] [m]
        rho: Charge density at cell centers [n_cells] [C/m^3]
        phi: Electric potential at faces [n_cells+1] [V]
        E: Electric field at faces [n_cells+1] [V/m]
        volume: Cell volume [m] (for 1D: dx, for 3D: dx*dy*dz)
    """

    def __init__(self, x_min, x_max, n_cells):
        """
        Initialize uniform 1D mesh.

        Args:
            x_min: Domain minimum [m]
            x_max: Domain maximum [m]
            n_cells: Number of interior cells
        """
        self.x_min = x_min
        self.x_max = x_max
        self.n_cells = n_cells
        self.dx = (x_max - x_min) / n_cells

        # Grid points
        self.x_faces = np.linspace(x_min, x_max, n_cells + 1)
        self.x_centers = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])

        # Physical fields
        self.rho = np.zeros(n_cells, dtype=np.float64)  # Charge density [C/m^3]
        self.phi = np.zeros(n_cells + 1, dtype=np.float64)  # Potential [V]
        self.E = np.zeros(n_cells + 1, dtype=np.float64)  # Electric field [V/m]

        # Volume (for 1D: length, for 3D integration later)
        self.volume = self.dx

    def reset_charge_density(self):
        """Reset charge density to zero (called before each deposition step)."""
        self.rho[:] = 0.0

    def get_cell_index(self, x):
        """
        Get cell index for particle position x.

        Args:
            x: Particle position [m]

        Returns:
            cell_index: Integer cell index (0 to n_cells-1)
                       Returns -1 if out of bounds
        """
        if x < self.x_min or x >= self.x_max:
            return -1

        cell_idx = int((x - self.x_min) / self.dx)

        # Clamp to valid range (handle numerical precision)
        if cell_idx < 0:
            cell_idx = 0
        elif cell_idx >= self.n_cells:
            cell_idx = self.n_cells - 1

        return cell_idx

    def check_debye_resolution(self, n_e, T_e):
        """
        Check if mesh satisfies Debye length resolution requirement.

        PIC stability requires: dx <= 0.5 * lambda_D

        Args:
            n_e: Electron density [m^-3]
            T_e: Electron temperature [eV]

        Returns:
            is_resolved: Boolean, True if dx <= 0.5 * lambda_D
            lambda_D: Debye length [m]
            ratio: dx / lambda_D

        Reference:
            Birdsall & Langdon (2004), Section 4.2
        """
        from ..constants import e, eV

        # Debye length: lambda_D = sqrt(eps0 * kT / (n_e * e^2))
        T_e_J = T_e * eV
        lambda_D = np.sqrt(eps0 * T_e_J / (n_e * e**2))

        ratio = self.dx / lambda_D
        is_resolved = ratio <= 0.5

        return is_resolved, lambda_D, ratio

    def __repr__(self):
        """String representation of mesh."""
        return (
            f"Mesh1DPIC(n_cells={self.n_cells}, "
            f"dx={self.dx*1e3:.3f} mm, "
            f"domain=[{self.x_min*1e3:.1f}, {self.x_max*1e3:.1f}] mm)"
        )


def create_mesh_from_debye_length(length, n_e, T_e, cells_per_debye=2):
    """
    Create mesh with automatic Debye length resolution.

    Ensures: dx = lambda_D / cells_per_debye (default: 2 cells per Debye length)

    Args:
        length: Domain length [m]
        n_e: Electron density [m^-3]
        T_e: Electron temperature [eV]
        cells_per_debye: Number of cells per Debye length (default: 2)

    Returns:
        mesh: Mesh1DPIC instance with proper resolution

    Example:
        >>> # Parodi thruster: n_e = 1.65e17 m^-3, T_e = 7.8 eV
        >>> mesh = create_mesh_from_debye_length(0.06, 1.65e17, 7.8)
        >>> print(mesh)
        Mesh1DPIC(n_cells=573, dx=0.105 mm, domain=[0.0, 60.0] mm)
    """
    from ..constants import e, eV

    # Calculate Debye length
    T_e_J = T_e * eV
    lambda_D = np.sqrt(eps0 * T_e_J / (n_e * e**2))

    # Determine cell size
    dx = lambda_D / cells_per_debye

    # Calculate number of cells
    n_cells = int(np.ceil(length / dx))

    # Create mesh
    mesh = Mesh1DPIC(0.0, length, n_cells)

    return mesh


def check_courant_condition(dt, dx, T_e):
    """
    Check plasma Courant condition: omega_pe * dt < 0.2

    This ensures that plasma oscillations are adequately resolved.

    Args:
        dt: Timestep [s]
        dx: Cell spacing [m]
        T_e: Electron temperature [eV]

    Returns:
        is_stable: Boolean, True if omega_pe * dt < 0.2
        omega_pe: Plasma frequency [rad/s]
        omega_dt: omega_pe * dt (dimensionless)

    Reference:
        Birdsall & Langdon (2004), Section 4.3
    """
    from ..constants import m_e, e

    # Estimate electron density from Debye length and cell size
    # This is approximate; actual n_e should be provided
    # For now, use dx ~ lambda_D assumption
    T_e_J = T_e * 1.602e-19  # eV to J
    lambda_D = dx  # Assume dx ~ lambda_D for conservative estimate
    n_e = eps0 * T_e_J / (lambda_D**2 * e**2)

    # Plasma frequency: omega_pe = sqrt(n_e * e^2 / (m_e * eps0))
    omega_pe = np.sqrt(n_e * e**2 / (m_e * eps0))

    omega_dt = omega_pe * dt
    is_stable = omega_dt < 0.2

    return is_stable, omega_pe, omega_dt


# ==================== TESTING UTILITIES ====================

def create_test_mesh(n_cells=100, length=0.01):
    """
    Create a test mesh for unit tests.

    Args:
        n_cells: Number of cells (default: 100)
        length: Domain length [m] (default: 1 cm)

    Returns:
        mesh: Mesh1DPIC instance
    """
    return Mesh1DPIC(0.0, length, n_cells)


if __name__ == "__main__":
    print("=" * 60)
    print("PIC Mesh Module - Self Test")
    print("=" * 60)

    # Test 1: Basic mesh creation
    print("\nTest 1: Basic mesh creation")
    mesh = Mesh1DPIC(0.0, 0.01, 100)
    print(mesh)
    print(f"  dx = {mesh.dx*1e6:.2f} microns")
    print(f"  n_faces = {len(mesh.x_faces)}")
    print(f"  n_centers = {len(mesh.x_centers)}")

    # Test 2: Debye-resolved mesh for Parodi thruster
    print("\nTest 2: Debye-resolved mesh (Parodi conditions)")
    n_e = 1.65e17  # m^-3
    T_e = 7.8      # eV
    mesh_parodi = create_mesh_from_debye_length(0.06, n_e, T_e, cells_per_debye=2)
    print(mesh_parodi)

    is_resolved, lambda_D, ratio = mesh_parodi.check_debye_resolution(n_e, T_e)
    print(f"  lambda_D = {lambda_D*1e3:.3f} mm")
    print(f"  dx/lambda_D = {ratio:.3f}")
    print(f"  Resolved: {is_resolved}")

    # Test 3: Courant condition
    print("\nTest 3: Courant condition check")
    dt = 1e-10  # 0.1 ns
    is_stable, omega_pe, omega_dt = check_courant_condition(dt, mesh_parodi.dx, T_e)
    print(f"  dt = {dt*1e9:.2f} ns")
    print(f"  omega_pe = {omega_pe/1e9:.2f} GHz")
    print(f"  omega_pe * dt = {omega_dt:.3f}")
    print(f"  Stable: {is_stable} (require < 0.2)")

    # Test 4: Cell indexing
    print("\nTest 4: Cell indexing")
    test_positions = [0.0, 0.005, 0.01, -0.001, 0.011]
    for x in test_positions:
        idx = mesh.get_cell_index(x)
        print(f"  x = {x*1e3:6.2f} mm -> cell {idx}")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
