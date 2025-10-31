"""
Tests for PIC sheath boundary conditions (energy-dependent reflection/absorption)
"""

import pytest
import numpy as np
from intakesim.particles import ParticleArrayNumba
from intakesim.pic.mesh import Mesh1DPIC
from intakesim.pic.surfaces import calculate_sheath_potential, apply_sheath_bc_1d
from intakesim.pic.mover import calculate_electron_temperature_eV
from intakesim.constants import e, m_e


class TestSheathPotentialCalculation:
    """Test suite for sheath potential calculation"""

    def test_sheath_potential_scaling(self):
        """Test that V_sheath scales linearly with T_e"""
        T_e_values = [1.0, 5.0, 10.0, 20.0]  # eV
        bohm_factor = 4.5

        for T_e in T_e_values:
            V_sheath = calculate_sheath_potential(T_e, bohm_factor=bohm_factor)
            expected = bohm_factor * T_e
            assert abs(V_sheath - expected) < 1e-12, f"V_sheath calculation incorrect for T_e={T_e} eV"

    def test_sheath_potential_typical_values(self):
        """Test typical discharge conditions"""
        # Typical CCP discharge: T_e ~ 3-5 eV
        T_e = 4.0  # eV
        V_sheath = calculate_sheath_potential(T_e, bohm_factor=4.5)

        # Expect V_sheath ~ 18 V
        assert 15.0 < V_sheath < 25.0, f"Sheath potential {V_sheath} V unrealistic for T_e={T_e} eV"

    def test_sheath_potential_high_temperature(self):
        """Test high temperature case (inductive discharge)"""
        # ICP discharge: T_e ~ 5-8 eV
        T_e = 7.0  # eV
        V_sheath = calculate_sheath_potential(T_e, bohm_factor=4.5)

        # Expect V_sheath ~ 31.5 V
        assert 25.0 < V_sheath < 40.0, f"Sheath potential {V_sheath} V unrealistic for T_e={T_e} eV"


class TestEnergyDependentBoundary:
    """Test suite for energy-dependent sheath boundary"""

    def test_low_energy_electron_reflects(self):
        """Test that low-energy electrons are reflected by sheath"""
        # Setup
        n_particles = 10
        x = np.zeros((n_particles, 3))
        v = np.zeros((n_particles, 3))
        active = np.ones(n_particles, dtype=bool)
        species_id = np.zeros(n_particles, dtype=np.int32)  # All electrons
        weight = np.ones(n_particles)

        # Electron with E = 1 eV (low energy)
        E_eV = 1.0
        v_mag = np.sqrt(2.0 * E_eV * e / m_e)
        x[0, 0] = -0.001  # Beyond left wall
        v[0, 0] = -v_mag  # Moving left (into wall)

        x_min, x_max = 0.0, 0.01
        T_e_eV = 5.0  # V_sheath = 22.5 V >> 1 eV → should reflect
        m_ion = 28 * 1.661e-27  # kg

        # Apply sheath BC
        n_reflected, n_absorbed = apply_sheath_bc_1d(
            x, v, active, species_id, weight, x_min, x_max, n_particles, T_e_eV, m_ion
        )

        # Verify reflection
        assert n_reflected == 1, "Low-energy electron should be reflected"
        assert n_absorbed == 0, "No particles should be absorbed"
        assert x[0, 0] > x_min, "Electron should be back in domain"
        assert v[0, 0] > 0, "Velocity should reverse (now moving right)"

    def test_high_energy_electron_absorbed(self):
        """Test that high-energy electrons overcome sheath and are absorbed"""
        # Setup
        n_particles = 10
        x = np.zeros((n_particles, 3))
        v = np.zeros((n_particles, 3))
        active = np.ones(n_particles, dtype=bool)
        species_id = np.zeros(n_particles, dtype=np.int32)  # All electrons
        weight = np.ones(n_particles)

        # Electron with E = 50 eV (high energy)
        E_eV = 50.0
        v_mag = np.sqrt(2.0 * E_eV * e / m_e)
        x[0, 0] = -0.001  # Beyond left wall
        v[0, 0] = -v_mag  # Moving left

        x_min, x_max = 0.0, 0.01
        T_e_eV = 5.0  # V_sheath = 22.5 V << 50 eV → should absorb
        m_ion = 28 * 1.661e-27  # kg

        # Apply sheath BC
        n_reflected, n_absorbed = apply_sheath_bc_1d(
            x, v, active, species_id, weight, x_min, x_max, n_particles, T_e_eV, m_ion
        )

        # Verify absorption
        assert n_reflected == 0, "High-energy electron should not be reflected"
        assert n_absorbed == 1, "High-energy electron should be absorbed"
        assert not active[0], "Absorbed particle should be inactive"

    def test_threshold_energy_behavior(self):
        """Test behavior near the threshold energy (E ~ e*V_sheath)"""
        # Setup
        n_particles = 100
        x = np.zeros((n_particles, 3))
        v = np.zeros((n_particles, 3))
        active = np.ones(n_particles, dtype=bool)
        species_id = np.zeros(n_particles, dtype=np.int32)  # All electrons
        weight = np.ones(n_particles)

        x_min, x_max = 0.0, 0.01
        T_e_eV = 5.0  # V_sheath = 22.5 V
        m_ion = 28 * 1.661e-27

        # Test energies around threshold
        # Below threshold: should reflect
        E_below = 20.0  # eV (< 22.5 V)
        v_below = np.sqrt(2.0 * E_below * e / m_e)
        x[0, 0] = -0.001
        v[0, 0] = -v_below

        # Above threshold: should absorb
        E_above = 25.0  # eV (> 22.5 V)
        v_above = np.sqrt(2.0 * E_above * e / m_e)
        x[1, 0] = -0.001
        v[1, 0] = -v_above

        # Apply sheath BC
        n_reflected, n_absorbed = apply_sheath_bc_1d(
            x, v, active, species_id, weight, x_min, x_max, n_particles, T_e_eV, m_ion
        )

        # Verify correct behavior
        assert active[0], "Below-threshold electron should be active (reflected)"
        assert not active[1], "Above-threshold electron should be absorbed"

    def test_ions_always_absorbed(self):
        """Test that all ions are absorbed regardless of energy"""
        # Setup
        n_particles = 10
        x = np.zeros((n_particles, 3))
        v = np.zeros((n_particles, 3))
        active = np.ones(n_particles, dtype=bool)
        species_id = np.ones(n_particles, dtype=np.int32)  # All ions (species 1)
        weight = np.ones(n_particles)

        # Low-energy ion (E = 1 eV)
        E_eV_low = 1.0
        m_ion = 28 * 1.661e-27  # kg (N2+)
        v_mag_low = np.sqrt(2.0 * E_eV_low * e / m_ion)
        x[0, 0] = -0.001
        v[0, 0] = -v_mag_low

        # High-energy ion (E = 100 eV)
        E_eV_high = 100.0
        v_mag_high = np.sqrt(2.0 * E_eV_high * e / m_ion)
        x[1, 0] = 0.012  # Beyond right wall
        v[1, 0] = v_mag_high

        x_min, x_max = 0.0, 0.01
        T_e_eV = 5.0

        # Apply sheath BC
        n_reflected, n_absorbed = apply_sheath_bc_1d(
            x, v, active, species_id, weight, x_min, x_max, n_particles, T_e_eV, m_ion
        )

        # Verify both ions absorbed
        assert n_reflected == 0, "Ions should never be reflected"
        assert n_absorbed == 2, "Both ions should be absorbed"
        assert not active[0], "Low-energy ion should be absorbed"
        assert not active[1], "High-energy ion should be absorbed"

    def test_interior_particles_unaffected(self):
        """Test that particles inside domain are not affected"""
        # Setup
        n_particles = 50
        x = np.random.uniform(0.001, 0.009, (n_particles, 3))
        x[:, 1:] = 0
        v = np.random.uniform(-1e6, 1e6, (n_particles, 3))
        v[:, 1:] = 0
        active = np.ones(n_particles, dtype=bool)
        species_id = np.zeros(n_particles, dtype=np.int32)  # Electrons
        weight = np.ones(n_particles)

        # Store initial state
        x_initial = x.copy()
        v_initial = v.copy()

        x_min, x_max = 0.0, 0.01
        T_e_eV = 5.0
        m_ion = 28 * 1.661e-27

        # Apply sheath BC
        n_reflected, n_absorbed = apply_sheath_bc_1d(
            x, v, active, species_id, weight, x_min, x_max, n_particles, T_e_eV, m_ion
        )

        # Verify no interaction
        assert n_reflected == 0, "Interior particles should not be reflected"
        assert n_absorbed == 0, "Interior particles should not be absorbed"
        np.testing.assert_array_equal(x, x_initial, err_msg="Positions should not change")
        np.testing.assert_array_equal(v, v_initial, err_msg="Velocities should not change")


class TestElectronTemperatureCalculation:
    """Test suite for electron temperature calculation"""

    def test_temperature_from_thermal_distribution(self):
        """Test temperature calculation from known Maxwell-Boltzmann distribution"""
        # Create electrons with known temperature
        n_electrons = 10000
        T_e_eV_input = 5.0  # eV

        # Thermal velocity distribution
        # <v²> = 3kT/m = 3eT/m (in SI units)
        v_thermal = np.sqrt(3.0 * T_e_eV_input * e / m_e)

        # Sample velocities from normal distribution
        np.random.seed(42)
        v = np.random.normal(0, v_thermal / np.sqrt(3), (n_electrons, 3))

        active = np.ones(n_electrons, dtype=bool)
        species_id = np.zeros(n_electrons, dtype=np.int32)  # All electrons

        # Calculate temperature
        T_e_eV_calculated = calculate_electron_temperature_eV(v, active, species_id, n_electrons)

        # Should match input within ~5% (Monte Carlo sampling error)
        relative_error = abs(T_e_eV_calculated - T_e_eV_input) / T_e_eV_input
        assert relative_error < 0.05, f"Temperature calculation error {relative_error:.2%} > 5%"

    def test_temperature_with_mixed_species(self):
        """Test that only electrons contribute to T_e calculation"""
        # Setup: mix of electrons and ions
        n_particles = 100
        v = np.random.uniform(-1e6, 1e6, (n_particles, 3))
        active = np.ones(n_particles, dtype=bool)

        # Half electrons, half ions
        species_id = np.zeros(n_particles, dtype=np.int32)
        species_id[50:] = 1  # Second half are ions

        # Calculate T_e
        T_e_eV = calculate_electron_temperature_eV(v, active, species_id, n_particles)

        # Should only use electron velocities (first 50 particles)
        # Calculate manually
        v_sq_sum = 0.0
        for i in range(50):  # Only electrons
            v_sq_sum += v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2
        v_sq_mean = v_sq_sum / 50
        T_e_expected = (m_e / (3.0 * e)) * v_sq_mean

        assert abs(T_e_eV - T_e_expected) < 1e-6, "Should only count electrons"

    def test_temperature_floor(self):
        """Test that temperature has a minimum floor"""
        # Setup: very low velocity electrons
        n_particles = 10
        v = np.zeros((n_particles, 3))  # Zero velocity → T = 0
        active = np.ones(n_particles, dtype=bool)
        species_id = np.zeros(n_particles, dtype=np.int32)

        # Calculate T_e
        T_e_eV = calculate_electron_temperature_eV(v, active, species_id, n_particles)

        # Should have floor at 0.1 eV
        assert T_e_eV >= 0.1, f"Temperature {T_e_eV} eV should have floor at 0.1 eV"

    def test_temperature_with_inactive_particles(self):
        """Test that inactive particles are ignored"""
        # Setup
        n_particles = 100
        v = np.random.uniform(-1e6, 1e6, (n_particles, 3))
        active = np.ones(n_particles, dtype=bool)
        species_id = np.zeros(n_particles, dtype=np.int32)

        # Deactivate half
        active[50:] = False

        # Calculate T_e
        T_e_eV = calculate_electron_temperature_eV(v, active, species_id, n_particles)

        # Calculate expected (only first 50)
        v_sq_sum = 0.0
        for i in range(50):
            if active[i]:
                v_sq_sum += v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2
        v_sq_mean = v_sq_sum / 50
        T_e_expected = max(0.1, (m_e / (3.0 * e)) * v_sq_mean)

        assert abs(T_e_eV - T_e_expected) < 1e-6, "Should only count active particles"


class TestSheathSelfConsistency:
    """Test self-consistent behavior of sheath boundaries"""

    def test_sheath_stabilizes_temperature(self):
        """Test that sheath BC prevents temperature runaway"""
        # This is an integration test showing the physical mechanism

        # Initial condition: electrons with high initial temperature
        n_particles = 1000
        T_e_initial_eV = 100.0  # Very hot (unrealistic)

        v_thermal = np.sqrt(3.0 * T_e_initial_eV * e / m_e)
        np.random.seed(42)
        v = np.random.normal(0, v_thermal / np.sqrt(3), (n_particles, 3))

        x = np.random.uniform(0.001, 0.009, (n_particles, 3))
        x[:, 1:] = 0

        active = np.ones(n_particles, dtype=bool)
        species_id = np.zeros(n_particles, dtype=np.int32)
        weight = np.ones(n_particles)

        x_min, x_max = 0.0, 0.01
        m_ion = 28 * 1.661e-27

        # Move particles toward walls
        x[:, 0] += v[:, 0] * 1e-9  # Small timestep

        # Apply sheath BC
        T_e_eV = calculate_electron_temperature_eV(v, active, species_id, n_particles)
        n_reflected, n_absorbed = apply_sheath_bc_1d(
            x, v, active, species_id, weight, x_min, x_max, n_particles, T_e_eV, m_ion
        )

        # High T_e → large V_sheath → only very high energy electrons escape
        # Most electrons should be reflected, a few absorbed
        # In real simulation, this selective absorption cools the distribution

        # Check that mechanism is working
        assert n_absorbed > 0, "Some high-energy electrons should be absorbed"
        # Note: n_reflected may be 0 if no particles reached walls in this test setup

        # Calculate new T_e after absorption
        T_e_after = calculate_electron_temperature_eV(v, active, species_id, n_particles)

        # Temperature should decrease (high-energy tail removed)
        # Note: This requires multiple timesteps in real simulation
        # Here we just verify the mechanism is present
        assert T_e_after <= T_e_initial_eV, "Absorption should remove high-energy electrons"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
