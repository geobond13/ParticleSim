"""
IntakeSIM: Air-Breathing Electric Propulsion Particle Simulation

Particle-based simulation toolkit combining DSMC and PIC methods
for ABEP system validation and performance prediction.

Authors: AeriSat Systems CTO Office
Version: 0.1.0 (Week 1 - Ballistic Motion Prototype)
Date: November 2025
"""

__version__ = "0.1.0"
__author__ = "AeriSat Systems"

# Import key classes for convenient access
from .constants import *
from .particles import ParticleArrayNumba
from .mesh import Mesh1D

__all__ = [
    "ParticleArrayNumba",
    "Mesh1D",
]
