"""
Geometry module for IntakeSIM.

Provides geometric models for ABEP intake structures.
"""

from .intake import (
    clausing_factor_analytical,
    transmission_probability_angle,
    HoneycombIntake,
    sample_freestream_velocity,
    apply_attitude_jitter,
    compute_compression_ratio,
    compute_hexagonal_channel_centers,
    get_channel_id_from_position,
    get_radial_distance_from_channel_center,
    get_wall_normal_at_position,
)

__all__ = [
    'clausing_factor_analytical',
    'transmission_probability_angle',
    'HoneycombIntake',
    'sample_freestream_velocity',
    'apply_attitude_jitter',
    'compute_compression_ratio',
    'compute_hexagonal_channel_centers',
    'get_channel_id_from_position',
    'get_radial_distance_from_channel_center',
    'get_wall_normal_at_position',
]
