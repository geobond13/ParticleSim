"""
Unit tests for particle data structures
"""

import pytest
import numpy as np
from intakesim.particles import (
    ParticleArrayNumba,
    sample_maxwellian_velocity,
    sample_shifted_maxwellian,
)
from intakesim.constants import SPECIES, kB


class TestParticleArrayNumba:
    """Test ParticleArrayNumba class."""

    def test_initialization(self):
        """Test particle array initialization."""
        particles = ParticleArrayNumba(max_particles=1000)

        assert particles.max_particles == 1000
        assert particles.n_particles == 0
        assert particles.x.shape == (1000, 3)
        assert particles.v.shape == (1000, 3)
        assert len(particles.weight) == 1000
        assert len(particles.species_id) == 1000
        assert len(particles.active) == 1000

    def test_add_particles_single(self):
        """Test adding a single particle."""
        particles = ParticleArrayNumba(100)

        indices = particles.add_particles(
            x=[0.1, 0.2, 0.3],
            v=[100, 0, 0],
            species='N2',
            weight=1.0
        )

        assert particles.n_particles == 1
        assert len(indices) == 1
        np.testing.assert_array_almost_equal(particles.x[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(particles.v[0], [100, 0, 0])
        assert particles.active[0] == True

    def test_add_particles_multiple(self):
        """Test adding multiple particles."""
        particles = ParticleArrayNumba(100)

        x = np.random.rand(50, 3)
        v = np.random.rand(50, 3)

        indices = particles.add_particles(x, v, species='O', weight=2.0)

        assert particles.n_particles == 50
        assert len(indices) == 50
        np.testing.assert_array_almost_equal(particles.x[:50], x)
        np.testing.assert_array_almost_equal(particles.v[:50], v)
        assert np.all(particles.weight[:50] == 2.0)
        assert np.all(particles.active[:50])

    def test_capacity_exceeded(self):
        """Test that adding too many particles raises error."""
        particles = ParticleArrayNumba(10)

        x = np.random.rand(5, 3)
        v = np.random.rand(5, 3)

        particles.add_particles(x, v, species='N2')  # OK
        assert particles.n_particles == 5

        # Try to add 10 more (total would be 15 > capacity 10)
        x_new = np.random.rand(10, 3)
        v_new = np.random.rand(10, 3)

        with pytest.raises(ValueError, match="exceed max capacity"):
            particles.add_particles(x_new, v_new, species='N2')

    def test_species_counting(self):
        """Test species counting and masking."""
        particles = ParticleArrayNumba(100)

        # Add N2 particles
        x_n2 = np.random.rand(30, 3)
        v_n2 = np.random.rand(30, 3)
        particles.add_particles(x_n2, v_n2, species='N2')

        # Add O particles
        x_o = np.random.rand(20, 3)
        v_o = np.random.rand(20, 3)
        particles.add_particles(x_o, v_o, species='O')

        assert particles.count_species('N2') == 30
        assert particles.count_species('O') == 20
        assert particles.n_particles == 50

    def test_remove_inactive(self):
        """Test compaction by removing inactive particles."""
        particles = ParticleArrayNumba(100)

        # Add 50 particles
        x = np.random.rand(50, 3)
        v = np.random.rand(50, 3)
        particles.add_particles(x, v, species='N2')

        # Deactivate first 10
        particles.active[:10] = False

        # Compact
        particles.remove_inactive()

        assert particles.n_particles == 40
        assert np.all(particles.active[:40])

    def test_kinetic_energy(self):
        """Test kinetic energy calculation."""
        particles = ParticleArrayNumba(10)

        # Add particle with known KE
        v = np.array([[100, 0, 0]])  # 100 m/s in x
        particles.add_particles(x=[[0, 0, 0]], v=v, species='N2', weight=1.0)

        KE = particles.kinetic_energy()

        # Expected: KE = 0.5 * m * v^2
        m_N2 = SPECIES['N2'].mass
        KE_expected = 0.5 * m_N2 * 100**2

        assert abs(KE - KE_expected) / KE_expected < 1e-6

    def test_momentum_conservation(self):
        """Test momentum calculation."""
        particles = ParticleArrayNumba(100)

        # Add particles with net zero momentum
        n = 50
        x = np.random.rand(n, 3)
        v = np.random.randn(n, 3) * 100
        particles.add_particles(x, v, species='N2', weight=1.0)

        # Add equal and opposite velocities
        particles.add_particles(x, -v, species='N2', weight=1.0)

        # Total momentum should be ~zero
        p = particles.momentum()
        m_N2 = SPECIES['N2'].mass

        # Allow for numerical error
        assert np.abs(p[0]) < 1e-10 * m_N2 * 100 * n
        assert np.abs(p[1]) < 1e-10 * m_N2 * 100 * n
        assert np.abs(p[2]) < 1e-10 * m_N2 * 100 * n


class TestVelocitySampling:
    """Test velocity distribution sampling."""

    def test_maxwellian_shape(self):
        """Test Maxwellian velocity sampling shape."""
        v = sample_maxwellian_velocity(T=300, mass=SPECIES['N2'].mass, n_samples=100)

        assert v.shape == (100, 3)

    def test_maxwellian_temperature(self):
        """Test that sampled velocities match temperature."""
        T = 300  # K
        mass = SPECIES['N2'].mass
        n_samples = 10000

        v = sample_maxwellian_velocity(T, mass, n_samples)

        # Compute temperature from kinetic energy
        v_squared = np.sum(v**2, axis=1)
        KE_per_particle = 0.5 * mass * np.mean(v_squared)

        # Expected: <KE> = (3/2) kB T
        T_measured = (2.0 / 3.0) * KE_per_particle / kB

        # Should match within 5% for 10k samples
        assert abs(T_measured - T) / T < 0.05

    def test_shifted_maxwellian(self):
        """Test shifted Maxwellian has correct bulk velocity."""
        T = 300
        mass = SPECIES['O'].mass
        v_bulk = np.array([1000, 0, 0])  # 1 km/s in x
        n_samples = 10000

        v = sample_shifted_maxwellian(T, mass, v_bulk, n_samples)

        # Mean velocity should equal bulk velocity
        v_mean = np.mean(v, axis=0)

        np.testing.assert_array_almost_equal(v_mean, v_bulk, decimal=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
