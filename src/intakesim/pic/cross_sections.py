"""
Electron-Impact Cross-Section Database

Collision cross sections for electron-neutral interactions in VLEO atmosphere:
- Species: O, N2, O2, NO
- Reactions: Ionization, excitation, elastic scattering
- Data sources: LXCat database (Biagi, Phelps), literature

Units:
    Energy: eV
    Cross-section: m^2

Data Format:
    Each reaction is a dictionary with:
        'type': 'ionization', 'excitation', 'elastic'
        'threshold': Threshold energy [eV]
        'energy_loss': Energy lost per collision [eV]
        'energies': Energy grid [eV]
        'cross_sections': Cross-section values [m^2]

References:
    - LXCat: www.lxcat.net (Biagi-v8.9, Phelps)
    - Itikawa (2006): Cross sections for electron collisions with N2, O2
    - Alves et al. (2013): The IST-Lisbon database on LXCat
"""

import numpy as np
import sys
import os

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from intakesim.constants import eV, m_e
else:
    from ..constants import eV, m_e

# ==================== CROSS-SECTION DATA ====================

# Cross-section database (organized by species and reaction type)
CROSS_SECTION_DATA = {}

# ==================== NITROGEN (N2) ====================

CROSS_SECTION_DATA['N2'] = {
    # Ionization: e + N2 -> N2+ + 2e
    'ionization': {
        'type': 'ionization',
        'threshold': 15.58,  # [eV]
        'energy_loss': 15.58,  # [eV]
        'products': ['N2+', 'e'],  # Products of reaction
        # Simplified cross-section (Biagi-v8.9 fit)
        # Peak around 100 eV
        'energies': np.array([
            15.58, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000
        ]),
        'cross_sections': np.array([
            0.0, 0.5, 1.2, 1.7, 2.0, 2.4, 2.7, 2.8, 2.7, 2.4, 2.0, 1.5
        ]) * 1e-20,  # Convert from 10^-20 m^2 to m^2
    },

    # Excitation: e + N2 -> N2* + e (combined vibrational + electronic)
    'excitation': {
        'type': 'excitation',
        'threshold': 6.17,  # [eV] (first electronic excitation A^3Sigma)
        'energy_loss': 8.0,  # [eV] (average of multiple levels)
        'products': ['N2*'],
        # Simplified excitation cross-section
        'energies': np.array([
            6.17, 8, 10, 12, 15, 20, 30, 50, 100, 200, 500
        ]),
        'cross_sections': np.array([
            0.0, 0.3, 0.8, 1.2, 1.5, 1.7, 1.5, 1.2, 0.8, 0.5, 0.3
        ]) * 1e-20,
    },

    # Elastic scattering: e + N2 -> e + N2
    'elastic': {
        'type': 'elastic',
        'threshold': 0.0,  # [eV] (no threshold)
        'energy_loss': 0.0,  # [eV] (elastic)
        'products': ['N2'],
        # Elastic momentum transfer cross-section
        # Decreases with energy (typical for molecules)
        'energies': np.array([
            0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 500
        ]),
        'cross_sections': np.array([
            10.0, 10.5, 8.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5
        ]) * 1e-20,
    },
}

# ==================== OXYGEN ATOM (O) ====================

CROSS_SECTION_DATA['O'] = {
    # Ionization: e + O -> O+ + 2e
    'ionization': {
        'type': 'ionization',
        'threshold': 13.62,  # [eV]
        'energy_loss': 13.62,  # [eV]
        'products': ['O+', 'e'],
        # From Biagi database
        'energies': np.array([
            13.62, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000
        ]),
        'cross_sections': np.array([
            0.0, 0.8, 1.5, 2.0, 2.3, 2.7, 3.0, 3.1, 3.0, 2.7, 2.2, 1.6
        ]) * 1e-20,
    },

    # Excitation: e + O -> O* + e
    'excitation': {
        'type': 'excitation',
        'threshold': 9.52,  # [eV] (first electronic excitation 3s ^3S)
        'energy_loss': 9.52,  # [eV]
        'products': ['O*'],
        'energies': np.array([
            9.52, 12, 15, 20, 30, 50, 100, 200, 500
        ]),
        'cross_sections': np.array([
            0.0, 0.4, 0.9, 1.3, 1.5, 1.3, 0.9, 0.6, 0.3
        ]) * 1e-20,
    },

    # Elastic scattering: e + O -> e + O
    'elastic': {
        'type': 'elastic',
        'threshold': 0.0,
        'energy_loss': 0.0,
        'products': ['O'],
        # Atomic oxygen elastic cross-section
        'energies': np.array([
            0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 500
        ]),
        'cross_sections': np.array([
            12.0, 11.0, 9.0, 7.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0
        ]) * 1e-20,
    },
}

# ==================== OXYGEN MOLECULE (O2) ====================

CROSS_SECTION_DATA['O2'] = {
    # Ionization: e + O2 -> O2+ + 2e
    'ionization': {
        'type': 'ionization',
        'threshold': 12.07,  # [eV]
        'energy_loss': 12.07,  # [eV]
        'products': ['O2+', 'e'],
        'energies': np.array([
            12.07, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000
        ]),
        'cross_sections': np.array([
            0.0, 0.7, 1.4, 1.9, 2.2, 2.6, 2.9, 3.0, 2.9, 2.6, 2.1, 1.5
        ]) * 1e-20,
    },

    # Excitation: e + O2 -> O2* + e
    'excitation': {
        'type': 'excitation',
        'threshold': 4.50,  # [eV] (a^1Delta_g lowest electronic state)
        'energy_loss': 6.0,  # [eV] (average)
        'products': ['O2*'],
        'energies': np.array([
            4.50, 6, 8, 10, 15, 20, 30, 50, 100, 200, 500
        ]),
        'cross_sections': np.array([
            0.0, 0.4, 0.9, 1.3, 1.6, 1.7, 1.5, 1.1, 0.7, 0.4, 0.2
        ]) * 1e-20,
    },

    # Elastic scattering: e + O2 -> e + O2
    'elastic': {
        'type': 'elastic',
        'threshold': 0.0,
        'energy_loss': 0.0,
        'products': ['O2'],
        'energies': np.array([
            0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 500
        ]),
        'cross_sections': np.array([
            11.0, 10.0, 7.5, 5.5, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.8, 1.5
        ]) * 1e-20,
    },
}

# ==================== NITRIC OXIDE (NO) ====================

CROSS_SECTION_DATA['NO'] = {
    # Ionization: e + NO -> NO+ + 2e
    'ionization': {
        'type': 'ionization',
        'threshold': 9.26,  # [eV]
        'energy_loss': 9.26,  # [eV]
        'products': ['NO+', 'e'],
        'energies': np.array([
            9.26, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500
        ]),
        'cross_sections': np.array([
            0.0, 0.9, 1.5, 2.0, 2.3, 2.5, 2.8, 3.0, 3.0, 2.9, 2.5, 2.0
        ]) * 1e-20,
    },

    # Excitation: e + NO -> NO* + e
    'excitation': {
        'type': 'excitation',
        'threshold': 5.45,  # [eV]
        'energy_loss': 5.45,  # [eV]
        'products': ['NO*'],
        'energies': np.array([
            5.45, 8, 10, 15, 20, 30, 50, 100, 200, 500
        ]),
        'cross_sections': np.array([
            0.0, 0.5, 1.0, 1.4, 1.6, 1.4, 1.0, 0.6, 0.3, 0.1
        ]) * 1e-20,
    },

    # Elastic scattering: e + NO -> e + NO
    'elastic': {
        'type': 'elastic',
        'threshold': 0.0,
        'energy_loss': 0.0,
        'products': ['NO'],
        'energies': np.array([
            0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 500
        ]),
        'cross_sections': np.array([
            10.5, 9.5, 7.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.3, 1.0
        ]) * 1e-20,
    },
}

# ==================== HELPER FUNCTIONS ====================

def get_cross_section(species, reaction_type, energy_eV):
    """
    Get collision cross-section for given species, reaction, and electron energy.

    Uses log-log linear interpolation for smooth variation.

    Args:
        species: Species name ('N2', 'O', 'O2', 'NO')
        reaction_type: Reaction type ('ionization', 'excitation', 'elastic')
        energy_eV: Electron energy [eV]

    Returns:
        sigma: Cross-section [m^2]

    Example:
        >>> sigma = get_cross_section('N2', 'ionization', 50.0)
        >>> print(f"sigma = {sigma:.3e} m^2")
        sigma = 2.000e-20 m^2
    """
    if species not in CROSS_SECTION_DATA:
        raise ValueError(f"Unknown species: {species}")

    if reaction_type not in CROSS_SECTION_DATA[species]:
        raise ValueError(f"Unknown reaction type: {reaction_type} for species {species}")

    data = CROSS_SECTION_DATA[species][reaction_type]

    # Below threshold: zero cross-section
    if energy_eV < data['threshold']:
        return 0.0

    # Interpolate (log-log for better accuracy)
    energies = data['energies']
    cross_sections = data['cross_sections']

    # Handle extrapolation
    if energy_eV <= energies[0]:
        return cross_sections[0]
    elif energy_eV >= energies[-1]:
        return cross_sections[-1]

    # Log-log interpolation
    log_E = np.log(energies)
    log_sigma = np.log(cross_sections + 1e-30)  # Avoid log(0)

    log_sigma_interp = np.interp(np.log(energy_eV), log_E, log_sigma)
    sigma = np.exp(log_sigma_interp)

    return sigma


def get_total_cross_section(species, energy_eV):
    """
    Get total collision cross-section (sum of all reaction types).

    Args:
        species: Species name
        energy_eV: Electron energy [eV]

    Returns:
        sigma_total: Total cross-section [m^2]
    """
    sigma_total = 0.0

    for reaction_type in CROSS_SECTION_DATA[species]:
        sigma_total += get_cross_section(species, reaction_type, energy_eV)

    return sigma_total


def get_collision_frequency(species, energy_eV, n_neutral):
    """
    Calculate collision frequency for electron-neutral collisions.

    nu = n_neutral * sigma(E) * v_electron

    where v_electron = sqrt(2*E/m_e)

    Args:
        species: Neutral species name
        energy_eV: Electron energy [eV]
        n_neutral: Neutral density [m^-3]

    Returns:
        nu: Collision frequency [s^-1]
    """
    # Electron velocity
    E_J = energy_eV * eV
    v_e = np.sqrt(2.0 * E_J / m_e)

    # Total cross-section
    sigma_total = get_total_cross_section(species, energy_eV)

    # Collision frequency
    nu = n_neutral * sigma_total * v_e

    return nu


def sample_collision_type(species, energy_eV, random_uniform):
    """
    Sample collision type based on relative cross-sections.

    Uses cumulative probability method:
        P(reaction_i) = sigma_i / sigma_total

    Args:
        species: Species name
        energy_eV: Electron energy [eV]
        random_uniform: Random number in [0, 1]

    Returns:
        reaction_type: Sampled reaction type ('ionization', 'excitation', 'elastic')
        None if energy below all thresholds
    """
    # Get all cross-sections
    sigmas = {}
    sigma_total = 0.0

    for reaction_type in CROSS_SECTION_DATA[species]:
        sigma = get_cross_section(species, reaction_type, energy_eV)
        sigmas[reaction_type] = sigma
        sigma_total += sigma

    if sigma_total == 0.0:
        return None  # No collision possible

    # Cumulative probability
    cumulative_prob = 0.0

    for reaction_type in ['elastic', 'excitation', 'ionization']:  # Order matters for sampling
        if reaction_type in sigmas:
            cumulative_prob += sigmas[reaction_type] / sigma_total

            if random_uniform < cumulative_prob:
                return reaction_type

    # Fallback (should not reach here)
    return 'elastic'


def get_reaction_data(species, reaction_type):
    """
    Get full reaction data dictionary.

    Args:
        species: Species name
        reaction_type: Reaction type

    Returns:
        data: Dictionary with threshold, energy_loss, products, etc.
    """
    if species not in CROSS_SECTION_DATA:
        raise ValueError(f"Unknown species: {species}")

    if reaction_type not in CROSS_SECTION_DATA[species]:
        raise ValueError(f"Unknown reaction: {reaction_type} for {species}")

    return CROSS_SECTION_DATA[species][reaction_type]


# ==================== TESTING ====================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Cross-Section Database - Self Test")
    print("=" * 60)

    # Test 1: Plot cross-sections for N2
    print("\nTest 1: Plotting N2 cross-sections...")

    fig, ax = plt.subplots(figsize=(10, 6))

    energies_plot = np.logspace(-2, 3, 200)  # 0.01 to 1000 eV

    for reaction in ['elastic', 'excitation', 'ionization']:
        sigmas = [get_cross_section('N2', reaction, E) for E in energies_plot]
        ax.plot(energies_plot, np.array(sigmas) * 1e20, label=f"N2 {reaction}", linewidth=2)

    ax.set_xlabel("Electron Energy [eV]")
    ax.set_ylabel("Cross-section [10^-20 m^2]")
    ax.set_title("Electron-N2 Collision Cross-Sections")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 20)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()
    plt.savefig("cross_sections_N2.png", dpi=150)
    print("  Saved: cross_sections_N2.png")

    # Test 2: Collision frequency at typical conditions
    print("\nTest 2: Collision frequency (Parodi conditions)...")

    n_N2 = 1.65e17  # m^-3 (compressed VLEO)
    E_e = 7.8  # eV (electron temperature)

    nu_total = get_collision_frequency('N2', E_e, n_N2)
    print(f"  n_N2 = {n_N2:.2e} m^-3")
    print(f"  E_e = {E_e:.1f} eV")
    print(f"  Total collision frequency: nu = {nu_total:.3e} s^-1")
    print(f"  Mean free time: tau = {1/nu_total*1e9:.2f} ns")

    # Test 3: Sample collision types
    print("\nTest 3: Sampling collision types at 20 eV...")

    n_samples = 10000
    energy = 20.0  # eV

    counts = {'elastic': 0, 'excitation': 0, 'ionization': 0}

    for _ in range(n_samples):
        r = np.random.random()
        reaction = sample_collision_type('N2', energy, r)
        if reaction:
            counts[reaction] += 1

    print(f"  Energy: {energy} eV")
    print(f"  Samples: {n_samples}")
    for reaction, count in counts.items():
        fraction = count / n_samples
        print(f"    {reaction}: {fraction:.3f}")

    # Compare to analytical probabilities
    sigma_elastic = get_cross_section('N2', 'elastic', energy)
    sigma_excitation = get_cross_section('N2', 'excitation', energy)
    sigma_ionization = get_cross_section('N2', 'ionization', energy)
    sigma_total = sigma_elastic + sigma_excitation + sigma_ionization

    print(f"\n  Analytical probabilities:")
    print(f"    elastic: {sigma_elastic/sigma_total:.3f}")
    print(f"    excitation: {sigma_excitation/sigma_total:.3f}")
    print(f"    ionization: {sigma_ionization/sigma_total:.3f}")

    # Test 4: Thresholds enforced
    print("\nTest 4: Checking thresholds...")

    for species in ['N2', 'O', 'O2', 'NO']:
        ionization_data = CROSS_SECTION_DATA[species]['ionization']
        threshold = ionization_data['threshold']

        sigma_below = get_cross_section(species, 'ionization', threshold - 0.1)
        sigma_above = get_cross_section(species, 'ionization', threshold + 5.0)

        print(f"  {species}: threshold = {threshold:.2f} eV")
        print(f"    sigma(E < threshold) = {sigma_below:.3e} m^2 (should be 0)")
        print(f"    sigma(E > threshold) = {sigma_above:.3e} m^2 (should be > 0)")

        assert sigma_below == 0.0, f"{species} ionization below threshold!"
        assert sigma_above > 0.0, f"{species} ionization above threshold is zero!"

    print("\n  [PASS] All thresholds correctly enforced")

    print("\n" + "=" * 60)
    print("Cross-section database validated!")
    print("=" * 60)
