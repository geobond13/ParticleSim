"""
Experimental Data from Cifali et al. (2011)
"Experimental characterization of HET and RIT with atmospheric propellants"
IEPC-2011-224

Extracted from: whitepapers/experimental characterization of HET and RIT with atmospheric propellants.pdf

This data provides experimental context for IntakeSIM validation.
NOT direct validation (different scale, power, geometry) - for reference only.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExperimentalDataPoint:
    """Single experimental measurement."""
    voltage_V: float
    current_A: float
    thrust_mN: float
    mass_flow_mg_s: float
    power_W: float
    propellant: str
    notes: str = ""


# ============================================================================
# HET (PPS1350-TSD) Experimental Data
# ============================================================================

# Pure Nitrogen Tests (from Figure 5 and Figure 6)
HET_N2_OPERATING_POINTS = [
    # Operating point of interest (RAM-EP application)
    ExperimentalDataPoint(
        voltage_V=305,
        current_A=3.48,
        thrust_mN=20.0,  # 19-21 mN range
        mass_flow_mg_s=2.65,
        power_W=305 * 3.48,  # ~1061 W
        propellant='N2',
        notes='Optimal RAM-EP point, 41.6 W/mN, 10hr stable operation'
    ),
    # Additional characterization points (approximate from graphs)
    ExperimentalDataPoint(
        voltage_V=305,
        current_A=3.0,
        thrust_mN=19.0,
        mass_flow_mg_s=2.65,
        power_W=305 * 3.0,
        propellant='N2',
        notes='Long-firing test point'
    ),
]

# N2/O2 Mixture Tests (1.27N2 + O2, representative of 200km atmosphere)
HET_N2O2_OPERATING_POINTS = [
    ExperimentalDataPoint(
        voltage_V=305,
        current_A=3.0,  # Approximate
        thrust_mN=24.0,
        mass_flow_mg_s=2.75,
        power_W=305 * 3.0,  # Approximate
        propellant='N2/O2 (1.27:1)',
        notes='Higher thrust than pure N2, 10hr stable operation'
    ),
]

# Summary statistics for HET
HET_N2_SUMMARY = {
    'thrust_range_mN': (19, 21),
    'power_to_thrust_W_per_mN': 41.6,
    'mass_flow_mg_s': 2.65,
    'voltage_V': 305,
    'current_A': 3.48,
    'total_power_W': 1061,
    'propellant': 'N2',
    'stability': '10 hours continuous',
    'performance_vs_Xe': 'Degraded due to low ionization efficiency',
}

HET_N2O2_SUMMARY = {
    'thrust_mN': 24,
    'mass_flow_mg_s': 2.75,
    'propellant': 'N2/O2 (1.27:1)',
    'stability': '10 hours continuous',
    'comparison': 'Higher thrust than pure N2 at similar conditions',
}


# ============================================================================
# RIT-10 EBBM Experimental Data
# ============================================================================

# Pure Nitrogen Tests (from Figure 14 and Figure 15)
RIT_N2_OPERATING_POINTS = [
    # Beam current levels tested: 75, 100, 150, 200, 234 mA
    # Grid voltages constant: Screen 1500V, Accel -600V

    # 450W operating point (ARTEMIS mission power level)
    ExperimentalDataPoint(
        voltage_V=1500,  # Screen grid
        current_A=0.150,  # 150 mA beam current (representative)
        thrust_mN=5.25,
        mass_flow_mg_s=8.514 / 60.0,  # Convert sccm to mg/s (approximate)
        power_W=450,
        propellant='N2',
        notes='Optimal N2 gas flow at ARTEMIS power level'
    ),

    # Long-firing test point
    ExperimentalDataPoint(
        voltage_V=1500,
        current_A=0.150,  # 150 mA beam current
        thrust_mN=5.0,  # Approximate
        mass_flow_mg_s=9.89 / 60.0,  # Convert sccm to mg/s
        power_W=450,  # Approximate
        propellant='N2',
        notes='10hr long-firing test, no grid erosion observed'
    ),
]

# Pure Oxygen Tests (from Figure 17 and Figure 18)
RIT_O2_OPERATING_POINTS = [
    # 450W operating point
    ExperimentalDataPoint(
        voltage_V=1500,
        current_A=0.150,  # 150 mA beam current (representative)
        thrust_mN=6.0,
        mass_flow_mg_s=8.0 / 60.0,  # Approximate optimal flow
        power_W=450,
        propellant='O2',
        notes='Higher thrust than N2 at same power, dissociation effects observed'
    ),

    # Long-firing test point
    ExperimentalDataPoint(
        voltage_V=1500,
        current_A=0.150,
        thrust_mN=6.0,
        mass_flow_mg_s=9.89 / 60.0,
        power_W=450,
        propellant='O2',
        notes='10hr test, higher grid erosion than N2 (chemical + physical)'
    ),
]

# Summary statistics for RIT-10
RIT_N2_SUMMARY = {
    'thrust_at_450W_mN': 5.25,
    'optimal_flow_sccm': 8.514,
    'beam_current_levels_mA': [75, 100, 150, 200, 234],
    'screen_voltage_V': 1500,
    'accel_voltage_V': -600,
    'propellant': 'N2',
    'performance_vs_Xe': 'Lower thrust, needs higher gas flow and power',
    'grid_erosion': 'No measurable erosion after 10hr test',
}

RIT_O2_SUMMARY = {
    'thrust_at_450W_mN': 6.0,
    'propellant': 'O2',
    'dissociation': 'O2 -> O affects thrust/beam ratio',
    'grid_erosion': 'Higher than N2 (chemical + physical sputtering)',
    'thrust_comparison': 'Higher than N2 due to mass ratio',
}


# ============================================================================
# Comparison to IntakeSIM
# ============================================================================

def compare_to_intakesim(intakesim_thrust_uN: float, intakesim_power_W: float = 20.0):
    """
    Compare IntakeSIM predictions to Cifali experimental data.

    Args:
        intakesim_thrust_uN: IntakeSIM predicted thrust [μN]
        intakesim_power_W: IntakeSIM power level [W]

    Returns:
        Dictionary with comparison analysis
    """
    intakesim_thrust_mN = intakesim_thrust_uN / 1000.0

    # HET comparison
    het_thrust_mN = HET_N2_SUMMARY['thrust_range_mN'][0]  # Use lower bound
    het_power_W = HET_N2_SUMMARY['total_power_W']
    het_specific_thrust = het_thrust_mN / het_power_W  # mN/W

    # RIT comparison
    rit_thrust_mN = RIT_N2_SUMMARY['thrust_at_450W_mN']
    rit_power_W = 450.0
    rit_specific_thrust = rit_thrust_mN / rit_power_W  # mN/W

    # IntakeSIM specific thrust
    intakesim_specific_thrust = intakesim_thrust_mN / intakesim_power_W

    # Scaling analysis
    het_scaled_to_intakesim = het_specific_thrust * intakesim_power_W
    rit_scaled_to_intakesim = rit_specific_thrust * intakesim_power_W

    return {
        'intakesim_thrust_mN': intakesim_thrust_mN,
        'intakesim_power_W': intakesim_power_W,
        'intakesim_specific_thrust_mN_per_W': intakesim_specific_thrust,

        'het_thrust_mN': het_thrust_mN,
        'het_power_W': het_power_W,
        'het_specific_thrust_mN_per_W': het_specific_thrust,
        'het_scaled_to_intakesim_power_mN': het_scaled_to_intakesim,

        'rit_thrust_mN': rit_thrust_mN,
        'rit_power_W': rit_power_W,
        'rit_specific_thrust_mN_per_W': rit_specific_thrust,
        'rit_scaled_to_intakesim_power_mN': rit_scaled_to_intakesim,

        'ratio_to_het_scaled': intakesim_thrust_mN / het_scaled_to_intakesim if het_scaled_to_intakesim > 0 else 0,
        'ratio_to_rit_scaled': intakesim_thrust_mN / rit_scaled_to_intakesim if rit_scaled_to_intakesim > 0 else 0,

        'analysis': f"""
        IntakeSIM predicts {intakesim_thrust_mN:.3f} mN at {intakesim_power_W}W.

        Cifali HET: {het_thrust_mN} mN at {het_power_W}W ({het_specific_thrust:.5f} mN/W)
        Scaled to {intakesim_power_W}W: {het_scaled_to_intakesim:.3f} mN

        Cifali RIT: {rit_thrust_mN} mN at {rit_power_W}W ({rit_specific_thrust:.5f} mN/W)
        Scaled to {intakesim_power_W}W: {rit_scaled_to_intakesim:.3f} mN

        NOTE: Different scale (CubeSat vs ground thruster), different geometry.
        This comparison is for physical reasonableness check only, not direct validation.
        """
    }


def print_summary():
    """Print summary of Cifali experimental data."""
    print("\n" + "="*70)
    print("Cifali et al. (2011) Experimental Data Summary")
    print("="*70)

    print("\n[HET - PPS1350-TSD with N2]")
    for key, value in HET_N2_SUMMARY.items():
        print(f"  {key}: {value}")

    print("\n[HET - PPS1350-TSD with N2/O2 mixture]")
    for key, value in HET_N2O2_SUMMARY.items():
        print(f"  {key}: {value}")

    print("\n[RIT-10 EBBM with N2]")
    for key, value in RIT_N2_SUMMARY.items():
        print(f"  {key}: {value}")

    print("\n[RIT-10 EBBM with O2]")
    for key, value in RIT_O2_SUMMARY.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)


if __name__ == "__main__":
    print_summary()

    # Example comparison
    print("\n\nExample: Compare IntakeSIM 480 uN prediction at 20W:")
    comparison = compare_to_intakesim(480.0, 20.0)
    print(comparison['analysis'])
