"""
Particle-in-Cell (PIC) Module

Implements electrostatic/electromagnetic PIC with Monte Carlo Collisions
for plasma simulation in the ABEP thruster.

Components:
- mesh: 1D mesh with Debye resolution
- field_solver: Poisson solver (Thomas algorithm)
- mover: Boris pusher with TSC weighting
"""

from .mesh import Mesh1DPIC, create_mesh_from_debye_length, check_courant_condition
from .field_solver import solve_fields_1d, solve_poisson_1d_dirichlet, compute_electric_field_1d
from .mover import (
    push_pic_particles_1d,
    boris_push_electrostatic,
    deposit_charge_tsc_1d,
    interpolate_field_tsc_1d,
    apply_absorbing_bc_1d,
    apply_periodic_bc_1d,
)

__all__ = [
    # Mesh
    "Mesh1DPIC",
    "create_mesh_from_debye_length",
    "check_courant_condition",
    # Field solver
    "solve_fields_1d",
    "solve_poisson_1d_dirichlet",
    "compute_electric_field_1d",
    # Mover
    "push_pic_particles_1d",
    "boris_push_electrostatic",
    "deposit_charge_tsc_1d",
    "interpolate_field_tsc_1d",
    "apply_absorbing_bc_1d",
    "apply_periodic_bc_1d",
]
