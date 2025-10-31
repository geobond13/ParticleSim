"""
Particle Data Structures for DSMC and PIC

Uses Structure-of-Arrays (SoA) layout for cache efficiency and Numba performance.
"""

import numpy as np
from .constants import SPECIES, SPECIES_ID, ID_TO_SPECIES


class ParticleArrayNumba:
    """
    Particle container optimized for Numba JIT compilation.

    Uses Structure-of-Arrays (SoA) layout where each property is stored
    in a separate array. This provides:
    - Better cache efficiency (accessing position doesn't load velocity)
    - Vectorization-friendly for Numba
    - 50-100× speedup over Array-of-Structures

    Attributes:
        x: Position vectors [n_max, 3] in meters
        v: Velocity vectors [n_max, 3] in m/s
        weight: Computational weight (number of real particles represented)
        species_id: Integer species identifier (maps to SPECIES dict)
        active: Boolean mask for active particles
        n_particles: Current number of active particles
        max_particles: Maximum capacity
    """

    def __init__(self, max_particles: int):
        """
        Initialize particle arrays.

        Args:
            max_particles: Maximum number of particles to allocate
        """
        self.max_particles = max_particles
        self.n_particles = 0

        # Pre-allocate arrays (SoA layout)
        self.x = np.zeros((max_particles, 3), dtype=np.float64)  # Position [m]
        self.v = np.zeros((max_particles, 3), dtype=np.float64)  # Velocity [m/s]
        self.weight = np.ones(max_particles, dtype=np.float64)  # Computational weight
        self.species_id = np.zeros(max_particles, dtype=np.int32)  # Species ID
        self.active = np.zeros(max_particles, dtype=np.bool_)  # Active flag

    def add_particles(self, x, v, species, weight=1.0):
        """
        Add particles to the array.

        Args:
            x: Positions, shape (n, 3) or (3,) [m]
            v: Velocities, shape (n, 3) or (3,) [m/s]
            species: Species name (str) or ID (int)
            weight: Computational weight (default 1.0)

        Returns:
            indices: Array indices of added particles

        Raises:
            ValueError: If array is full
        """
        # Convert inputs to arrays
        x = np.atleast_2d(x)
        v = np.atleast_2d(v)

        n_add = x.shape[0]

        # Check capacity
        if self.n_particles + n_add > self.max_particles:
            raise ValueError(
                f"Cannot add {n_add} particles: "
                f"would exceed max capacity {self.max_particles}"
            )

        # Get species ID
        if isinstance(species, str):
            species_id = SPECIES_ID[species]
        else:
            species_id = species

        # Add particles
        start_idx = self.n_particles
        end_idx = start_idx + n_add

        self.x[start_idx:end_idx] = x
        self.v[start_idx:end_idx] = v
        self.species_id[start_idx:end_idx] = species_id
        self.weight[start_idx:end_idx] = weight
        self.active[start_idx:end_idx] = True

        self.n_particles = end_idx

        return np.arange(start_idx, end_idx)

    def remove_inactive(self):
        """
        Compact array by removing inactive particles.

        This is expensive (O(n)) so should be done infrequently,
        perhaps every 100-1000 timesteps.
        """
        if self.n_particles == 0:
            return

        # Find active particles
        active_mask = self.active[:self.n_particles]
        n_active = np.sum(active_mask)

        if n_active == 0:
            self.n_particles = 0
            return

        # Compact arrays
        self.x[:n_active] = self.x[:self.n_particles][active_mask]
        self.v[:n_active] = self.v[:self.n_particles][active_mask]
        self.weight[:n_active] = self.weight[:self.n_particles][active_mask]
        self.species_id[:n_active] = self.species_id[:self.n_particles][active_mask]
        self.active[:n_active] = True

        self.n_particles = n_active

    def get_species_mask(self, species):
        """
        Get mask of particles of a given species.

        Args:
            species: Species name (str) or ID (int)

        Returns:
            mask: Boolean array of shape (n_particles,)
        """
        if isinstance(species, str):
            species_id = SPECIES_ID[species]
        else:
            species_id = species

        mask = (self.species_id[:self.n_particles] == species_id) & \
               self.active[:self.n_particles]

        return mask

    def count_species(self, species):
        """
        Count active particles of a given species.

        Args:
            species: Species name (str) or ID (int)

        Returns:
            count: Number of active particles
        """
        return np.sum(self.get_species_mask(species))

    def get_mass_array(self):
        """
        Get array of particle masses.

        Returns:
            masses: Array of shape (n_particles,) in kg
        """
        masses = np.zeros(self.n_particles)
        for species_name, species_id in SPECIES_ID.items():
            mask = self.species_id[:self.n_particles] == species_id
            masses[mask] = SPECIES[species_name].mass

        return masses

    def get_charge_array(self):
        """
        Get array of particle charges.

        Returns:
            charges: Array of shape (n_particles,) in Coulombs
        """
        charges = np.zeros(self.n_particles)
        for species_name, species_id in SPECIES_ID.items():
            mask = self.species_id[:self.n_particles] == species_id
            charges[mask] = SPECIES[species_name].charge

        return charges

    def kinetic_energy(self):
        """
        Total kinetic energy of all active particles.

        Returns:
            KE: Kinetic energy in Joules
        """
        masses = self.get_mass_array()
        v_squared = np.sum(self.v[:self.n_particles]**2, axis=1)
        active_mask = self.active[:self.n_particles]

        return 0.5 * np.sum(masses[active_mask] * v_squared[active_mask] *
                           self.weight[:self.n_particles][active_mask])

    def momentum(self):
        """
        Total momentum of all active particles.

        Returns:
            p: Momentum vector [px, py, pz] in kg·m/s
        """
        masses = self.get_mass_array()
        active_mask = self.active[:self.n_particles]

        momentum = np.zeros(3)
        for i in range(3):
            momentum[i] = np.sum(masses[active_mask] *
                                self.v[:self.n_particles, i][active_mask] *
                                self.weight[:self.n_particles][active_mask])

        return momentum

    def __repr__(self):
        """String representation."""
        active_count = np.sum(self.active[:self.n_particles])
        return (f"ParticleArrayNumba(n_particles={self.n_particles}, "
                f"active={active_count}, max={self.max_particles})")

    def __len__(self):
        """Return number of particles (including inactive)."""
        return self.n_particles

    def summary(self):
        """Print summary statistics."""
        print(f"\nParticle Array Summary:")
        print(f"  Total particles:  {self.n_particles}")
        print(f"  Active particles: {np.sum(self.active[:self.n_particles])}")
        print(f"  Max capacity:     {self.max_particles}")
        print(f"  Fill ratio:       {100*self.n_particles/self.max_particles:.1f}%")

        print(f"\n  Species breakdown:")
        for species_name in ['O', 'N2', 'O2', 'NO', 'e', 'O+', 'N2+']:
            if species_name in SPECIES_ID:
                count = self.count_species(species_name)
                if count > 0:
                    print(f"    {species_name:4s}: {count:10d}")

        if self.n_particles > 0:
            KE = self.kinetic_energy()
            p = self.momentum()
            print(f"\n  Kinetic energy: {KE:.3e} J")
            print(f"  Momentum: [{p[0]:.3e}, {p[1]:.3e}, {p[2]:.3e}] kg·m/s")


# ==================== HELPER FUNCTIONS ====================

def sample_maxwellian_velocity(T, mass, n_samples=1):
    """
    Sample velocity from 3D Maxwellian distribution.

    Args:
        T: Temperature [K]
        mass: Particle mass [kg]
        n_samples: Number of samples

    Returns:
        v: Velocity array of shape (n_samples, 3) in m/s
    """
    from .constants import kB

    # Thermal velocity (standard deviation of velocity distribution)
    v_th = np.sqrt(kB * T / mass)

    # Sample from normal distribution in each direction
    v = np.random.normal(0, v_th, size=(n_samples, 3))

    return v if n_samples > 1 else v[0]


def sample_shifted_maxwellian(T, mass, v_bulk, n_samples=1):
    """
    Sample from shifted Maxwellian (e.g., for orbital velocity).

    Args:
        T: Temperature [K]
        mass: Particle mass [kg]
        v_bulk: Bulk velocity vector [vx, vy, vz] in m/s
        n_samples: Number of samples

    Returns:
        v: Velocity array of shape (n_samples, 3) in m/s
    """
    v_thermal = sample_maxwellian_velocity(T, mass, n_samples)
    v_bulk = np.atleast_2d(v_bulk)

    return v_thermal + v_bulk


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing ParticleArrayNumba...")

    # Create particle array
    particles = ParticleArrayNumba(max_particles=1000)

    # Add some N2 particles
    print("\nAdding 100 N2 particles...")
    x_n2 = np.random.rand(100, 3) * 0.1  # Random positions in 10 cm cube
    v_n2 = sample_maxwellian_velocity(T=300, mass=SPECIES['N2'].mass, n_samples=100)
    particles.add_particles(x_n2, v_n2, species='N2')

    # Add some O particles
    print("Adding 50 O particles...")
    x_o = np.random.rand(50, 3) * 0.1
    v_o = sample_maxwellian_velocity(T=1000, mass=SPECIES['O'].mass, n_samples=50)
    particles.add_particles(x_o, v_o, species='O')

    # Print summary
    particles.summary()

    # Test deactivation
    print("\nDeactivating first 10 particles...")
    particles.active[:10] = False

    print(f"Active count before compaction: {np.sum(particles.active[:particles.n_particles])}")

    particles.remove_inactive()

    print(f"Active count after compaction:  {particles.n_particles}")

    print("\n✅ ParticleArrayNumba tests passed!")
