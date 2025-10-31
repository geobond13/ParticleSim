"""
Tests for diagnostic module.

Week 5: Validate diagnostic calculations.
"""

import numpy as np
import pytest

from intakesim.diagnostics import (
    compute_density_profile,
    compute_velocity_distribution,
    compute_temperature_profile,
    compute_compression_ratio,
    check_mass_conservation,
    check_energy_conservation,
    check_momentum_conservation,
    DiagnosticTracker,
)
from intakesim.constants import SPECIES, kB


class TestDensityProfile:
    """Test density profile calculation."""

    def test_uniform_distribution(self):
        """Uniform particle distribution gives constant density."""
        n_particles = 1000
        length = 1.0

        # Uniform distribution
        x = np.random.rand(n_particles, 3) * length
        x[:, 1:] = 0.0
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)

        # Bins
        z_bins = np.linspace(0, length, 11)

        density = compute_density_profile(x, active, weight, n_particles, z_bins)

        # Should be approximately uniform
        mean_density = np.mean(density)
        std_density = np.std(density)

        assert std_density / mean_density < 0.3, "Density should be approximately uniform"

    def test_empty_bins(self):
        """Empty bins should have zero density."""
        n_particles = 100
        x = np.zeros((n_particles, 3))
        x[:, 0] = 0.5  # All particles at z=0.5
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)

        z_bins = np.linspace(0, 1, 11)

        density = compute_density_profile(x, active, weight, n_particles, z_bins)

        # Only one bin should be non-zero
        non_zero_bins = np.sum(density > 0)
        assert non_zero_bins == 1, "Only one bin should contain particles"


class TestVelocityDistribution:
    """Test velocity distribution calculation."""

    def test_single_velocity(self):
        """All particles with same velocity."""
        n_particles = 100
        v = np.ones((n_particles, 3)) * 1000.0  # 1 km/s
        active = np.ones(n_particles, dtype=np.bool_)

        v_bins = np.linspace(0, 2000, 21)

        hist = compute_velocity_distribution(v, active, n_particles, v_bins)

        # All particles should be in one bin
        assert np.sum(hist) == n_particles, "All particles should be counted"
        assert np.max(hist) == n_particles, "All particles in one bin"


class TestTemperatureProfile:
    """Test temperature calculation."""

    def test_temperature_from_velocity(self):
        """Temperature calculated correctly from velocity."""
        n_particles = 1000
        T_target = 300.0  # K
        mass = SPECIES['N2'].mass

        # Sample Maxwell-Boltzmann velocities
        from intakesim.particles import sample_maxwellian_velocity
        v = sample_maxwellian_velocity(T_target, mass, n_particles)

        x = np.random.rand(n_particles, 3)
        active = np.ones(n_particles, dtype=np.bool_)

        z_bins = np.array([0.0, 1.0])  # Single bin

        temperature = compute_temperature_profile(v, mass, active, n_particles, z_bins, x)

        # Should recover target temperature within ~10%
        assert abs(temperature[0] - T_target) / T_target < 0.1, \
            f"Temperature {temperature[0]:.1f} K should match target {T_target} K"


class TestCompressionRatio:
    """Test compression ratio calculation."""

    def test_no_compression(self):
        """Equal density at inlet and outlet gives CR=1."""
        n_particles = 1000

        # Uniform distribution
        x = np.random.rand(n_particles, 3)
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)

        CR, n_in, n_out = compute_compression_ratio(
            x, active, weight, n_particles,
            z_inlet=0.1, z_outlet=0.9, dz_sample=0.05
        )

        # CR should be close to 1.0 (allow variance from random sampling)
        assert abs(CR - 1.0) < 0.7, f"CR={CR:.2f} should be close to 1.0 for uniform distribution"

    def test_compression(self):
        """Higher density at outlet gives CR > 1."""
        n_particles = 1000

        # More particles near outlet
        x = np.random.rand(n_particles, 3)
        x[:, 0] = 0.9 + np.random.rand(n_particles) * 0.05  # Concentrate near outlet
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)

        # Add some particles at inlet
        n_inlet_particles = 100
        x[:n_inlet_particles, 0] = 0.1 + np.random.rand(n_inlet_particles) * 0.05

        CR, n_in, n_out = compute_compression_ratio(
            x, active, weight, n_particles,
            z_inlet=0.1, z_outlet=0.9, dz_sample=0.05
        )

        # CR should be > 1
        assert CR > 1.0, f"CR={CR:.2f} should be > 1.0 for concentrated outlet"
        assert n_out > n_in, "Outlet density should be higher than inlet"


class TestConservation:
    """Test conservation law checkers."""

    def test_mass_conservation_exact(self):
        """Exact mass conservation gives zero error."""
        error, is_conserved = check_mass_conservation(
            n_particles_initial=100,
            n_particles_final=150,
            n_injected=50,
            n_removed=0
        )

        assert error == 0.0, "Exact conservation should give zero error"
        assert is_conserved, "Should report as conserved"

    def test_energy_conservation(self):
        """Energy conservation check."""
        n_particles = 100
        mass = SPECIES['N2'].mass

        # Create velocities
        v = np.random.randn(n_particles, 3) * 500.0
        active = np.ones(n_particles, dtype=np.bool_)

        # Compute initial energy
        E_initial = 0.0
        for i in range(n_particles):
            v_sq = v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2
            E_initial += 0.5 * mass * v_sq

        # Check conservation (should be exact)
        E_final, error = check_energy_conservation(v, mass, active, n_particles, E_initial)

        assert error < 1e-10, "Energy should be exactly conserved"
        assert abs(E_final - E_initial) / E_initial < 1e-10, "Energies should match"


class TestDiagnosticTracker:
    """Test DiagnosticTracker class."""

    def test_tracker_initialization(self):
        """Tracker initializes correctly."""
        tracker = DiagnosticTracker(n_steps=1000, output_interval=10)

        assert tracker.n_outputs == 101, "Should have 101 output slots"
        assert tracker.output_idx == 0, "Should start at index 0"
        assert len(tracker.time) == 101, "Arrays should be sized correctly"

    def test_tracker_record(self):
        """Tracker records data correctly."""
        tracker = DiagnosticTracker(n_steps=100, output_interval=10)

        # Create dummy particle data
        n_particles = 100
        x = np.random.rand(n_particles, 3)
        v = np.random.randn(n_particles, 3) * 1000.0
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)
        mass = SPECIES['O'].mass

        # Record
        tracker.record(
            step=0,
            time=0.0,
            x=x,
            v=v,
            active=active,
            weight=weight,
            n_particles=n_particles,
            mass=mass,
            z_inlet=0.1,
            z_outlet=0.9
        )

        assert tracker.output_idx == 1, "Should have recorded one entry"
        assert tracker.n_particles[0] == n_particles, "Particle count should be recorded"
        assert tracker.compression_ratio[0] >= 0, "CR should be non-negative"

    def test_tracker_save_csv(self, tmp_path):
        """Tracker saves CSV correctly."""
        tracker = DiagnosticTracker(n_steps=100, output_interval=10)

        # Record some dummy data
        n_particles = 100
        x = np.random.rand(n_particles, 3)
        v = np.random.randn(n_particles, 3) * 1000.0
        active = np.ones(n_particles, dtype=np.bool_)
        weight = np.ones(n_particles, dtype=np.float64)
        mass = SPECIES['O'].mass

        for i in range(5):
            tracker.record(
                step=i*10,
                time=i*1e-5,
                x=x,
                v=v,
                active=active,
                weight=weight,
                n_particles=n_particles,
                mass=mass,
                z_inlet=0.1,
                z_outlet=0.9
            )

        # Save
        csv_file = tmp_path / "test_diagnostics.csv"
        tracker.save_csv(str(csv_file))

        # Verify file exists
        assert csv_file.exists(), "CSV file should be created"

        # Verify contents
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 6, "Should have header + 5 data rows"
        assert rows[0][0] == 'step', "First column should be 'step'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
