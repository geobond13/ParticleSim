"""
Physical Constants and Species Properties

All units in SI unless otherwise noted.
"""

import numpy as np

# ==================== FUNDAMENTAL CONSTANTS ====================

# Basic constants
e = 1.602176634e-19  # Elementary charge [C]
m_e = 9.1093837015e-31  # Electron mass [kg]
m_p = 1.67262192369e-27  # Proton mass [kg]
eps0 = 8.8541878128e-12  # Vacuum permittivity [F/m]
mu0 = 1.25663706212e-6  # Vacuum permeability [H/m]
c = 299792458.0  # Speed of light [m/s]
kB = 1.380649e-23  # Boltzmann constant [J/K]
h = 6.62607015e-34  # Planck constant [J·s]
AMU = 1.66053906660e-27  # Atomic mass unit [kg]
eV = e  # 1 eV in Joules [J]

# Derived constants
alpha_fine = e**2 / (4 * np.pi * eps0 * c * h)  # Fine structure constant
a0 = 4 * np.pi * eps0 * h**2 / (m_e * e**2)  # Bohr radius [m]

# ==================== SPECIES DATABASE ====================

class SpeciesData:
    """
    Species properties for VLEO atmospheric and plasma particles.

    Attributes:
        mass: Particle mass [kg]
        diameter: VHS reference diameter [m]
        omega: VHS temperature exponent (dimensionless)
        charge: Particle charge [C] (0 for neutrals)
        ionization_energy: Ionization threshold [eV] (None for ions)
    """

    def __init__(self, mass, diameter, omega=0.74, charge=0, ionization_energy=None):
        self.mass = mass
        self.diameter = diameter
        self.omega = omega  # Temperature exponent for VHS
        self.charge = charge
        self.ionization_energy = ionization_energy


# Neutral species
SPECIES = {
    # Atomic oxygen
    'O': SpeciesData(
        mass=16.0 * AMU,
        diameter=3.0e-10,  # 3.0 Angstroms
        omega=0.80,  # VHS exponent
        charge=0,
        ionization_energy=13.618  # eV
    ),

    # Molecular nitrogen
    'N2': SpeciesData(
        mass=28.014 * AMU,
        diameter=4.17e-10,  # 4.17 Angstroms
        omega=0.74,
        charge=0,
        ionization_energy=15.58  # eV
    ),

    # Molecular oxygen
    'O2': SpeciesData(
        mass=32.0 * AMU,
        diameter=4.01e-10,  # 4.01 Angstroms
        omega=0.77,
        charge=0,
        ionization_energy=12.07  # eV
    ),

    # Nitric oxide
    'NO': SpeciesData(
        mass=30.006 * AMU,
        diameter=3.5e-10,  # Estimate
        omega=0.76,
        charge=0,
        ionization_energy=9.26  # eV
    ),

    # Ions
    'O+': SpeciesData(
        mass=16.0 * AMU,
        diameter=2.5e-10,
        omega=0.80,
        charge=e,
        ionization_energy=None  # Already ionized
    ),

    'N2+': SpeciesData(
        mass=28.014 * AMU,
        diameter=4.0e-10,
        omega=0.74,
        charge=e,
        ionization_energy=None
    ),

    'O2+': SpeciesData(
        mass=32.0 * AMU,
        diameter=3.8e-10,
        omega=0.77,
        charge=e,
        ionization_energy=None
    ),

    'NO+': SpeciesData(
        mass=30.006 * AMU,
        diameter=3.3e-10,
        omega=0.76,
        charge=e,
        ionization_energy=None
    ),

    # Electrons
    'e': SpeciesData(
        mass=m_e,
        diameter=1e-15,  # Classical electron radius (negligible for collisions)
        omega=0.5,  # Not used for electrons
        charge=-e,
        ionization_energy=None
    ),
}

# Create integer species IDs for efficient array indexing
SPECIES_ID = {name: i for i, name in enumerate(SPECIES.keys())}
ID_TO_SPECIES = {i: name for name, i in SPECIES_ID.items()}

# ==================== ATMOSPHERIC COMPOSITION ====================

# VLEO atmospheric composition at 200 km
# Source: NRLMSISE-00 model
ATMOSPHERE_200KM = {
    'altitude': 200e3,  # [m]
    'temperature': 1000.0,  # [K] (varies with solar activity)
    'total_density': 4.2e17,  # [m^-3] total number density
    'composition': {
        'O': 0.83,  # 83% atomic oxygen
        'N2': 0.14,  # 14% molecular nitrogen
        'O2': 0.02,  # 2% molecular oxygen
        'NO': 0.01,  # 1% nitric oxide
    }
}

# Compute number densities
for species, fraction in ATMOSPHERE_200KM['composition'].items():
    ATMOSPHERE_200KM[f'n_{species}'] = fraction * ATMOSPHERE_200KM['total_density']

# Orbital velocity at 200 km
ATMOSPHERE_200KM['v_orbital'] = 7780.0  # [m/s]

# ==================== REFERENCE VALUES ====================

# Reference conditions for non-dimensionalization
T_REF = 300.0  # [K]
n_REF = 1e20  # [m^-3]
L_REF = 0.01  # [m] - 1 cm

# Reference derived quantities
v_thermal_ref = np.sqrt(2 * kB * T_REF / SPECIES['N2'].mass)  # [m/s]
lambda_mfp_ref = 1 / (np.sqrt(2) * np.pi * SPECIES['N2'].diameter**2 * n_REF)  # [m]
tau_coll_ref = lambda_mfp_ref / v_thermal_ref  # [s]

# ==================== PLASMA PARAMETERS ====================

def debye_length(n_e, T_e):
    """
    Electron Debye length.

    Args:
        n_e: Electron density [m^-3]
        T_e: Electron temperature [eV]

    Returns:
        lambda_D: Debye length [m]
    """
    T_e_J = T_e * eV
    return np.sqrt(eps0 * T_e_J / (n_e * e**2))


def plasma_frequency(n_e):
    """
    Electron plasma frequency.

    Args:
        n_e: Electron density [m^-3]

    Returns:
        omega_pe: Plasma frequency [rad/s]
    """
    return np.sqrt(n_e * e**2 / (m_e * eps0))


def electron_thermal_velocity(T_e):
    """
    Electron thermal velocity.

    Args:
        T_e: Electron temperature [eV]

    Returns:
        v_th: Thermal velocity [m/s]
    """
    T_e_J = T_e * eV
    return np.sqrt(2 * T_e_J / m_e)


# ==================== UTILITY FUNCTIONS ====================

def mean_free_path(n, diameter):
    """
    Mean free path for hard-sphere collisions.

    Args:
        n: Number density [m^-3]
        diameter: Collision diameter [m]

    Returns:
        lambda_mfp: Mean free path [m]
    """
    sigma = np.pi * diameter**2
    return 1 / (np.sqrt(2) * sigma * n)


def collision_frequency(n, diameter, v_relative):
    """
    Collision frequency for hard-sphere model.

    Args:
        n: Number density [m^-3]
        diameter: Collision diameter [m]
        v_relative: Relative velocity [m/s]

    Returns:
        nu: Collision frequency [Hz]
    """
    sigma = np.pi * diameter**2
    return n * sigma * v_relative


def thermal_velocity(T, mass):
    """
    Most probable thermal velocity (Maxwellian).

    Args:
        T: Temperature [K]
        mass: Particle mass [kg]

    Returns:
        v_th: Thermal velocity [m/s]
    """
    return np.sqrt(2 * kB * T / mass)


# ==================== CONSTANTS SUMMARY ====================

if __name__ == "__main__":
    print("=" * 60)
    print("IntakeSIM Physical Constants")
    print("=" * 60)

    print("\nFundamental Constants:")
    print(f"  Elementary charge:     e = {e:.6e} C")
    print(f"  Electron mass:         m_e = {m_e:.6e} kg")
    print(f"  Boltzmann constant:    kB = {kB:.6e} J/K")
    print(f"  Atomic mass unit:      AMU = {AMU:.6e} kg")

    print("\nSpecies Database:")
    for name, data in SPECIES.items():
        print(f"  {name:4s}: m = {data.mass/AMU:6.2f} AMU, "
              f"d = {data.diameter*1e10:4.2f} Å, "
              f"q = {data.charge/e:+2.0f}e")

    print("\nVLEO Atmosphere (200 km):")
    print(f"  Total density: {ATMOSPHERE_200KM['total_density']:.2e} m^-3")
    print(f"  Temperature:   {ATMOSPHERE_200KM['temperature']:.0f} K")
    print(f"  Composition:")
    for species, fraction in ATMOSPHERE_200KM['composition'].items():
        n = ATMOSPHERE_200KM[f'n_{species}']
        print(f"    {species:4s}: {fraction*100:5.1f}%  ({n:.2e} m^-3)")

    print("\nReference Values:")
    print(f"  v_thermal (N2 @ 300 K): {v_thermal_ref:.1f} m/s")
    print(f"  lambda_mfp (n=1e20):    {lambda_mfp_ref*1e3:.2f} mm")
    print(f"  tau_coll:               {tau_coll_ref*1e6:.2f} μs")

    print("\nPlasma Parameters (example: n_e=1e17 m^-3, T_e=8 eV):")
    print(f"  Debye length:     {debye_length(1e17, 8)*1e3:.3f} mm")
    print(f"  Plasma frequency: {plasma_frequency(1e17)/1e9:.2f} GHz")
    print(f"  Electron v_th:    {electron_thermal_velocity(8)/1e6:.2f} km/s")
    print("=" * 60)
