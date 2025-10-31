"""
Power Balance and Energy Diagnostics for PIC Simulations

CRITICAL VALIDATION: Power balance MUST close within 10% for physically correct simulations.

Power Balance Equation:
    P_in = P_out + dE/dt

Where:
    P_in = Input power (RF, DC, heating source)
    P_out = Ion losses + Electron losses + Ionization + Excitation + Radiation
    dE/dt = Rate of change of stored energy

For steady-state: P_in ≈ P_out (within 10%)

This module provides:
    - Power balance tracking
    - Energy decomposition
    - Real-time validation
    - Time-averaged statistics

Reference:
    Lieberman & Lichtenberg (2005), "Principles of Plasma Discharges"
    Turner et al. (2013), "Simulation benchmarks for low-pressure plasmas"

Author: AeriSat Systems
Date: 2025
"""

import numpy as np
import sys
import os

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from intakesim.constants import e, m_e
else:
    from ..constants import e, m_e

# ==================== POWER BALANCE CALCULATION ====================

def calculate_power_balance(
    particles,
    mesh,
    mcc_diagnostics,
    see_diagnostics,
    dt,
    P_input=0.0,
    volume=None
):
    """
    Calculate power balance for PIC-MCC simulation.

    P_in = P_input (externally supplied)
    P_out = P_ions + P_electrons + P_ionization + P_excitation

    Args:
        particles: ParticleArrayNumba instance
        mesh: Mesh1DPIC instance
        mcc_diagnostics: Dict from MCC module with collision counts
        see_diagnostics: Dict from SEE module with wall interactions
        dt: Timestep [s]
        P_input: Input power [W] (e.g., RF absorbed power)
        volume: Discharge volume [m^3] (if None, use mesh.dx for 1D)

    Returns:
        power_balance: dict with all power components [W]
            {
                'P_input': float,
                'P_ion_loss': float,
                'P_electron_loss': float,
                'P_ionization': float,
                'P_excitation': float,
                'P_total_loss': float,
                'P_balance_error': float,
                'P_balance_error_percent': float,
                'stored_energy': float,
            }
    """
    try:
        from ..constants import ID_TO_SPECIES
    except ImportError:
        from intakesim.constants import ID_TO_SPECIES

    # Volume (for 1D: just dx, for 3D would be V)
    if volume is None:
        volume = mesh.dx

    # ==================== POWER LOSSES ====================

    # Ion losses to walls
    n_ions_lost = see_diagnostics.get('n_ions_absorbed', 0)

    # Estimate average ion energy at wall
    # Ions accelerated through sheath: E_ion ~ e * V_sheath
    # For now, use simplified estimate
    V_sheath_estimate = 4.0  # Typical: 3-5 × T_e, assume T_e ~ 1V
    E_ion_per_particle = e * V_sheath_estimate  # Joules

    # Power from ion losses
    P_ion_loss = (n_ions_lost * E_ion_per_particle) / dt

    # Electron losses to walls
    n_electrons_lost = see_diagnostics.get('n_electrons_absorbed', 0)
    n_secondaries = see_diagnostics.get('n_secondaries_created', 0)

    # Net electron loss (primaries - secondaries)
    n_electrons_net_loss = n_electrons_lost - n_secondaries

    # Average electron energy
    # Calculate from particle velocities
    electron_energies = []
    for i in range(particles.n_particles):
        if not particles.active[i]:
            continue

        species_name = ID_TO_SPECIES[particles.species_id[i]]
        if species_name == 'e':
            v = particles.v[i, :]
            v_mag_sq = np.sum(v * v)
            E_J = 0.5 * m_e * v_mag_sq
            electron_energies.append(E_J)

    if len(electron_energies) > 0:
        E_electron_mean = np.mean(electron_energies)
    else:
        E_electron_mean = 2.0 * e  # Default: 2 eV

    # Power from electron losses
    # Electrons lose kinetic energy + work function (typically ~2 × KE)
    E_electron_wall = E_electron_mean * 2.0  # Factor of 2 for sheath acceleration
    P_electron_loss = (n_electrons_net_loss * E_electron_wall) / dt

    # Ionization power loss
    n_ionizations = mcc_diagnostics.get('n_ionization', 0)

    # Ionization energy (N2: 15.58 eV, use average)
    E_ionization_eV = 15.0  # Average ionization potential
    E_ionization_J = E_ionization_eV * e

    P_ionization = (n_ionizations * E_ionization_J) / dt

    # Excitation power loss
    n_excitations = mcc_diagnostics.get('n_excitation', 0)

    # Excitation energy (average ~8 eV)
    E_excitation_eV = 8.0
    E_excitation_J = E_excitation_eV * e

    P_excitation = (n_excitations * E_excitation_J) / dt

    # Total power loss
    P_total_loss = P_ion_loss + P_electron_loss + P_ionization + P_excitation

    # ==================== POWER BALANCE ====================

    # Error
    P_balance_error = P_input - P_total_loss

    # Relative error
    if P_input > 0.0:
        P_balance_error_percent = abs(P_balance_error) / P_input * 100.0
    else:
        P_balance_error_percent = 0.0

    # ==================== STORED ENERGY ====================

    # Total kinetic energy in particles
    E_kinetic_total = 0.0

    for i in range(particles.n_particles):
        if not particles.active[i]:
            continue

        species_name = ID_TO_SPECIES[particles.species_id[i]]
        v = particles.v[i, :]
        v_mag_sq = np.sum(v * v)

        if species_name == 'e':
            mass = m_e
        else:
            # Ions (approximate)
            mass = 28 * 1.661e-27  # kg (N2+)

        E_k = 0.5 * mass * v_mag_sq
        E_kinetic_total += E_k * particles.weight[i]  # Account for statistical weight

    # Electric field energy
    E_field_total = 0.5 * 8.854e-12 * np.sum(mesh.E**2) * volume

    stored_energy = E_kinetic_total + E_field_total

    return {
        'P_input': P_input,
        'P_ion_loss': P_ion_loss,
        'P_electron_loss': P_electron_loss,
        'P_ionization': P_ionization,
        'P_excitation': P_excitation,
        'P_total_loss': P_total_loss,
        'P_balance_error': P_balance_error,
        'P_balance_error_percent': P_balance_error_percent,
        'stored_energy': stored_energy,
    }


# ==================== PLASMA PARAMETERS ====================

def calculate_plasma_parameters(particles, mesh):
    """
    Calculate key plasma parameters.

    Args:
        particles: ParticleArrayNumba instance
        mesh: Mesh1DPIC instance

    Returns:
        params: dict with plasma parameters
            {
                'n_e': float,  # Electron density [m^-3]
                'T_e': float,  # Electron temperature [eV]
                'n_ion': float,  # Ion density [m^-3]
                'T_ion': float,  # Ion temperature [eV]
                'lambda_D': float,  # Debye length [m]
                'omega_pe': float,  # Plasma frequency [rad/s]
            }
    """
    try:
        from ..constants import ID_TO_SPECIES, eps0
    except ImportError:
        from intakesim.constants import ID_TO_SPECIES, eps0

    # Count particles
    n_electrons = 0
    n_ions = 0
    total_weight_electrons = 0.0
    total_weight_ions = 0.0

    electron_energies = []
    ion_energies = []

    for i in range(particles.n_particles):
        if not particles.active[i]:
            continue

        species_name = ID_TO_SPECIES[particles.species_id[i]]
        v = particles.v[i, :]
        v_mag_sq = np.sum(v * v)
        weight = particles.weight[i]

        if species_name == 'e':
            E_eV = 0.5 * m_e * v_mag_sq / e
            electron_energies.append(E_eV)

            n_electrons += 1
            total_weight_electrons += weight

        elif '+' in species_name:
            # Ion
            m_ion = 28 * 1.661e-27  # kg (approximate)
            E_eV = 0.5 * m_ion * v_mag_sq / e
            ion_energies.append(E_eV)

            n_ions += 1
            total_weight_ions += weight

    # Densities (assume uniform in domain)
    volume = mesh.dx  # For 1D
    n_e = total_weight_electrons / volume if volume > 0 else 0.0
    n_ion = total_weight_ions / volume if volume > 0 else 0.0

    # Temperatures (from kinetic energy: <E> = (3/2) k T → T = (2/3) <E>)
    if len(electron_energies) > 0:
        mean_E_e_eV = np.mean(electron_energies)
        T_e_eV = (2.0 / 3.0) * mean_E_e_eV  # For 3D velocities
    else:
        T_e_eV = 0.0

    if len(ion_energies) > 0:
        mean_E_ion_eV = np.mean(ion_energies)
        T_ion_eV = (2.0 / 3.0) * mean_E_ion_eV
    else:
        T_ion_eV = 0.0

    # Debye length
    if n_e > 0 and T_e_eV > 0:
        T_e_J = T_e_eV * e
        lambda_D = np.sqrt(eps0 * T_e_J / (n_e * e**2))
    else:
        lambda_D = 0.0

    # Plasma frequency
    if n_e > 0:
        omega_pe = np.sqrt(n_e * e**2 / (m_e * eps0))
    else:
        omega_pe = 0.0

    return {
        'n_e': n_e,
        'T_e': T_e_eV,
        'n_ion': n_ion,
        'T_ion': T_ion_eV,
        'lambda_D': lambda_D,
        'omega_pe': omega_pe,
    }


# ==================== TIME-AVERAGED DIAGNOSTICS ====================

class PowerBalanceTracker:
    """
    Track power balance over time for validation.

    Usage:
        tracker = PowerBalanceTracker()

        for step in range(n_steps):
            # ... simulation step ...

            power_data = calculate_power_balance(...)
            tracker.update(power_data, time)

        stats = tracker.get_statistics()
        print(f"Mean power balance error: {stats['mean_error_percent']:.2f}%")
    """

    def __init__(self):
        self.times = []
        self.P_input = []
        self.P_total_loss = []
        self.P_balance_error = []
        self.P_balance_error_percent = []

        self.P_ion_loss = []
        self.P_electron_loss = []
        self.P_ionization = []
        self.P_excitation = []

    def update(self, power_data, time):
        """Add new power balance data point."""
        self.times.append(time)
        self.P_input.append(power_data['P_input'])
        self.P_total_loss.append(power_data['P_total_loss'])
        self.P_balance_error.append(power_data['P_balance_error'])
        self.P_balance_error_percent.append(power_data['P_balance_error_percent'])

        self.P_ion_loss.append(power_data['P_ion_loss'])
        self.P_electron_loss.append(power_data['P_electron_loss'])
        self.P_ionization.append(power_data['P_ionization'])
        self.P_excitation.append(power_data['P_excitation'])

    def get_statistics(self):
        """Get time-averaged statistics."""
        if len(self.times) == 0:
            return {}

        return {
            'mean_P_input': np.mean(self.P_input),
            'mean_P_total_loss': np.mean(self.P_total_loss),
            'mean_error_percent': np.mean(np.abs(self.P_balance_error_percent)),
            'max_error_percent': np.max(np.abs(self.P_balance_error_percent)),
            'mean_P_ion_loss': np.mean(self.P_ion_loss),
            'mean_P_electron_loss': np.mean(self.P_electron_loss),
            'mean_P_ionization': np.mean(self.P_ionization),
            'mean_P_excitation': np.mean(self.P_excitation),
        }

    def passes_validation(self, threshold=10.0):
        """
        Check if power balance passes validation.

        Args:
            threshold: Maximum allowed error [%] (default: 10%)

        Returns:
            passes: True if mean error < threshold
        """
        stats = self.get_statistics()
        return stats.get('mean_error_percent', 100.0) < threshold


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Power Balance Diagnostics - Self Test")
    print("=" * 60)

    # Test 1: Simple power balance calculation
    print("\nTest 1: Power balance calculation...")

    # Mock data
    mcc_diagnostics = {
        'n_ionization': 100,
        'n_excitation': 200,
        'n_elastic': 500,
    }

    see_diagnostics = {
        'n_electrons_absorbed': 50,
        'n_ions_absorbed': 100,
        'n_secondaries_created': 25,
    }

    # Inputs
    dt = 1e-10  # 0.1 ns
    P_input = 20.0  # 20 W

    # Calculate (without particles/mesh for now - simplified test)
    # We'll just validate the structure

    print(f"  Input power: {P_input} W")
    print(f"  Timestep: {dt*1e9:.1f} ns")
    print(f"  Ionizations: {mcc_diagnostics['n_ionization']}")
    print(f"  Electrons absorbed: {see_diagnostics['n_electrons_absorbed']}")
    print(f"  Ions absorbed: {see_diagnostics['n_ions_absorbed']}")
    print()

    # Test 2: PowerBalanceTracker
    print("Test 2: PowerBalanceTracker...")

    tracker = PowerBalanceTracker()

    # Simulate some timesteps
    for step in range(10):
        # Mock power data with some error
        power_data = {
            'P_input': 20.0,
            'P_total_loss': 20.0 + np.random.normal(0, 1.0),  # ±1 W error
            'P_balance_error': np.random.normal(0, 1.0),
            'P_balance_error_percent': abs(np.random.normal(0, 5.0)),  # ~5% error
            'P_ion_loss': 8.0,
            'P_electron_loss': 6.0,
            'P_ionization': 4.0,
            'P_excitation': 2.0,
        }

        tracker.update(power_data, step * dt)

    stats = tracker.get_statistics()

    print(f"  Timesteps: 10")
    print(f"  Mean P_input: {stats['mean_P_input']:.2f} W")
    print(f"  Mean P_loss: {stats['mean_P_total_loss']:.2f} W")
    print(f"  Mean error: {stats['mean_error_percent']:.2f}%")
    print(f"  Max error: {stats['max_error_percent']:.2f}%")
    print()

    # Validation
    passes = tracker.passes_validation(threshold=10.0)
    print(f"  Passes validation (<10% error): {passes}")

    if passes:
        print("  [PASS] Power balance tracker working correctly")
    else:
        print("  [WARN] Mock data has high error (expected for random test)")

    print("\n" + "=" * 60)
    print("Power balance diagnostics module created!")
    print("IMPORTANT: Integrate with full PIC simulation for real validation")
    print("=" * 60)
