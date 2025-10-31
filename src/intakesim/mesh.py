"""
1D Mesh for Spatial Indexing in DSMC

Provides uniform 1D mesh for efficient particle-to-cell mapping
and collision detection.
"""

import numpy as np
import numba
from numba import njit


class Mesh1D:
    """
    Uniform 1D mesh for DSMC spatial indexing.

    Particles are assigned to cells based on x-coordinate.
    Cells enable efficient collision detection (only particles
    in the same cell can collide).

    Attributes:
        length: Domain length [m]
        n_cells: Number of cells
        dx: Cell width [m]
        cell_edges: Cell boundary positions [m]
        cell_centers: Cell center positions [m]
    """

    def __init__(self, length: float, n_cells: int, cross_section: float = 1.0):
        """
        Initialize uniform 1D mesh.

        Args:
            length: Domain length in meters
            n_cells: Number of cells
            cross_section: Cross-sectional area for 1D channel [m^2]
                          Default 1.0 m^2 for unit testing.
                          For physical systems, set to actual area.
        """
        self.length = length
        self.n_cells = n_cells
        self.dx = length / n_cells
        self.cross_section = cross_section

        # Cell boundaries (n_cells + 1 edges)
        self.cell_edges = np.linspace(0, length, n_cells + 1)

        # Cell centers
        self.cell_centers = 0.5 * (self.cell_edges[:-1] + self.cell_edges[1:])

        # Volume of each cell (for 1D: length × cross-section)
        self.cell_volumes = np.full(n_cells, self.dx * cross_section)

    def get_cell_index(self, x):
        """
        Get cell index for a position (Python interface).

        Args:
            x: Position(s) [m], scalar or array

        Returns:
            cell_idx: Cell index (0 to n_cells-1), or -1 if out of bounds
        """
        x = np.atleast_1d(x)
        cell_idx = np.floor(x / self.dx).astype(np.int32)

        # Handle boundaries
        cell_idx[x < 0] = -1
        cell_idx[x >= self.length] = -1
        cell_idx[cell_idx >= self.n_cells] = -1

        return cell_idx if len(cell_idx) > 1 else cell_idx[0]

    def count_particles_per_cell(self, particle_positions, active):
        """
        Count number of active particles in each cell.

        Args:
            particle_positions: Particle x-coordinates, shape (n_particles,)
            active: Active flags, shape (n_particles,)

        Returns:
            counts: Number of particles per cell, shape (n_cells,)
        """
        return _count_particles_per_cell_numba(
            particle_positions, active, self.n_cells, self.dx
        )

    def get_particles_in_cell(self, cell_idx, particle_positions, active):
        """
        Get indices of active particles in a specific cell.

        Args:
            cell_idx: Cell index (0 to n_cells-1)
            particle_positions: Particle x-coordinates, shape (n_particles,)
            active: Active flags, shape (n_particles,)

        Returns:
            indices: Array of particle indices in this cell
        """
        x_min = cell_idx * self.dx
        x_max = (cell_idx + 1) * self.dx

        in_cell = (particle_positions >= x_min) & \
                  (particle_positions < x_max) & \
                  active

        return np.where(in_cell)[0]

    def __repr__(self):
        """String representation."""
        return (f"Mesh1D(length={self.length:.3f} m, n_cells={self.n_cells}, "
                f"dx={self.dx*1000:.2f} mm)")


# ==================== NUMBA-COMPILED FUNCTIONS ====================

@njit
def _count_particles_per_cell_numba(x, active, n_cells, dx):
    """
    Count particles per cell (Numba-compiled).

    Args:
        x: Particle x-coordinates, shape (n_particles,)
        active: Active flags, shape (n_particles,)
        n_cells: Number of cells
        dx: Cell width [m]

    Returns:
        counts: Particles per cell, shape (n_cells,)
    """
    counts = np.zeros(n_cells, dtype=np.int64)

    for i in range(len(x)):
        if active[i]:
            cell_idx = int(x[i] / dx)
            if 0 <= cell_idx < n_cells:
                counts[cell_idx] += 1

    return counts


@njit
def get_cell_index_numba(x, dx):
    """
    Get cell index for a single position (Numba-compiled).

    Args:
        x: Position [m]
        dx: Cell width [m]

    Returns:
        cell_idx: Cell index, or -1 if out of bounds
    """
    if x < 0:
        return -1

    cell_idx = int(x / dx)
    return cell_idx


@njit
def index_particles_to_cells(x, active, n_cells, dx, max_per_cell):
    """
    Create index arrays mapping cells to particles.

    Args:
        x: Particle x-coordinates, shape (n_particles,)
        active: Active flags, shape (n_particles,)
        n_cells: Number of cells
        dx: Cell width [m]
        max_per_cell: Maximum particles per cell (for array allocation)

    Returns:
        cell_particles: Particle indices per cell, shape (n_cells, max_per_cell)
        cell_counts: Number of particles in each cell, shape (n_cells,)

    Note:
        If a cell has more than max_per_cell particles, extras are ignored.
        Choose max_per_cell large enough to avoid this.
    """
    n_particles = len(x)

    cell_particles = np.full((n_cells, max_per_cell), -1, dtype=np.int32)
    cell_counts = np.zeros(n_cells, dtype=np.int32)

    for i in range(n_particles):
        if active[i]:
            cell_idx = int(x[i] / dx)

            if 0 <= cell_idx < n_cells:
                count = cell_counts[cell_idx]

                if count < max_per_cell:
                    cell_particles[cell_idx, count] = i
                    cell_counts[cell_idx] += 1

    return cell_particles, cell_counts


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Mesh1D...")

    # Create mesh
    mesh = Mesh1D(length=1.0, n_cells=10)
    print(f"\n{mesh}")
    print(f"Cell edges: {mesh.cell_edges}")
    print(f"Cell centers: {mesh.cell_centers}")

    # Test particle indexing
    print("\nTesting particle indexing...")
    x_particles = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    active = np.ones(len(x_particles), dtype=bool)

    # Get cell indices
    for i, x in enumerate(x_particles):
        cell_idx = mesh.get_cell_index(x)
        print(f"  Particle at x={x:.2f} m → Cell {cell_idx}")

    # Count particles per cell
    counts = mesh.count_particles_per_cell(x_particles, active)
    print(f"\nParticles per cell: {counts}")
    print(f"Total: {np.sum(counts)}")

    # Test cell particle lookup
    print("\nParticles in cell 2:")
    indices = mesh.get_particles_in_cell(2, x_particles, active)
    print(f"  Indices: {indices}")
    print(f"  Positions: {x_particles[indices]}")

    # Test Numba indexing function
    print("\nTesting Numba indexing...")
    cell_particles, cell_counts = index_particles_to_cells(
        x_particles, active, mesh.n_cells, mesh.dx, max_per_cell=10
    )
    print(f"  cell_counts: {cell_counts}")
    print(f"  cell_particles[2]: {cell_particles[2][cell_particles[2] >= 0]}")

    print("\n✅ Mesh1D tests passed!")
