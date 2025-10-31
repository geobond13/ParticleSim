"""
Diagnostic utilities for DSMC and PIC simulations.

This module provides comprehensive diagnostic capabilities for analyzing
particle-based simulations:
- Density profiles
- Velocity distributions
- Temperature calculations
- Compression ratio tracking
- Conservation law validation
- Data export (CSV, HDF5)
- Visualization utilities

Week 5 Deliverable: Diagnostic suite for intake validation.
"""

import numpy as np
from numba import njit
import csv
from typing import Dict, List, Tuple, Optional

from .constants import kB, SPECIES


@njit
def compute_density_profile(x, active, weight, n_particles, z_bins):
    """
    Compute number density profile along z-axis.

    Args:
        x: Particle positions (n_particles, 3) [m]
        active: Active flags (n_particles,)
        weight: Particle weights (n_particles,)
        n_particles: Number of particles
        z_bins: Bin edges along z-axis (n_bins+1,) [m]

    Returns:
        density: Number density in each bin (n_bins,) [m^-3]
    """
    n_bins = len(z_bins) - 1
    density = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_particles):
        if not active[i]:
            continue

        z = x[i, 0]

        # Find bin index
        for j in range(n_bins):
            if z_bins[j] <= z < z_bins[j + 1]:
                density[j] += weight[i]
                counts[j] += 1
                break

    # Normalize by bin volume (assuming unit cross-section for now)
    for j in range(n_bins):
        dz = z_bins[j + 1] - z_bins[j]
        if counts[j] > 0:
            density[j] = density[j] / dz  # m^-3 (per unit area)

    return density


@njit
def compute_velocity_distribution(v, active, n_particles, v_bins):
    """
    Compute velocity distribution histogram.

    Args:
        v: Particle velocities (n_particles, 3) [m/s]
        active: Active flags (n_particles,)
        n_particles: Number of particles
        v_bins: Bin edges for velocity magnitude (n_bins+1,) [m/s]

    Returns:
        histogram: Count in each velocity bin (n_bins,)
    """
    n_bins = len(v_bins) - 1
    histogram = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_particles):
        if not active[i]:
            continue

        v_mag = np.sqrt(v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2)

        # Find bin index
        for j in range(n_bins):
            if v_bins[j] <= v_mag < v_bins[j + 1]:
                histogram[j] += 1
                break

    return histogram


@njit
def compute_temperature_profile(v, mass, active, n_particles, z_bins, x):
    """
    Compute temperature profile along z-axis.

    Temperature from kinetic theory: T = m * <v^2> / (3 * kB)

    Args:
        v: Particle velocities (n_particles, 3) [m/s]
        mass: Particle mass [kg]
        active: Active flags (n_particles,)
        n_particles: Number of particles
        z_bins: Bin edges along z-axis (n_bins+1,) [m]
        x: Particle positions (n_particles, 3) [m]

    Returns:
        temperature: Temperature in each bin (n_bins,) [K]
    """
    n_bins = len(z_bins) - 1
    temperature = np.zeros(n_bins, dtype=np.float64)
    v_squared_sum = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_particles):
        if not active[i]:
            continue

        z = x[i, 0]
        v_sq = v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2

        # Find bin index
        for j in range(n_bins):
            if z_bins[j] <= z < z_bins[j + 1]:
                v_squared_sum[j] += v_sq
                counts[j] += 1
                break

    # Compute temperature: T = m * <v^2> / (3*kB)
    for j in range(n_bins):
        if counts[j] > 0:
            mean_v_sq = v_squared_sum[j] / counts[j]
            temperature[j] = mass * mean_v_sq / (3.0 * kB)

    return temperature


def compute_compression_ratio(
    x, active, weight, n_particles, z_inlet, z_outlet, dz_sample=0.005
):
    """
    Compute compression ratio between inlet and outlet regions.

    CR = (density at outlet) / (density at inlet)

    Args:
        x: Particle positions (n_particles, 3) [m]
        active: Active flags (n_particles,)
        weight: Particle weights (n_particles,)
        n_particles: Number of particles
        z_inlet: Z-coordinate of inlet sampling region [m]
        z_outlet: Z-coordinate of outlet sampling region [m]
        dz_sample: Width of sampling regions [m]

    Returns:
        CR: Compression ratio
        n_inlet: Number density at inlet [m^-3]
        n_outlet: Number density at outlet [m^-3]
    """
    # Count particles in inlet and outlet regions
    n_inlet_weighted = 0.0
    n_outlet_weighted = 0.0

    for i in range(n_particles):
        if not active[i]:
            continue

        z = x[i, 0]

        # Inlet region
        if z_inlet <= z < z_inlet + dz_sample:
            n_inlet_weighted += weight[i]

        # Outlet region
        if z_outlet <= z < z_outlet + dz_sample:
            n_outlet_weighted += weight[i]

    # Normalize by sample volume (assuming unit cross-section)
    n_inlet = n_inlet_weighted / dz_sample if n_inlet_weighted > 0 else 0.0
    n_outlet = n_outlet_weighted / dz_sample if n_outlet_weighted > 0 else 0.0

    # Compression ratio
    CR = n_outlet / n_inlet if n_inlet > 0 else 0.0

    return CR, n_inlet, n_outlet


def check_mass_conservation(
    n_particles_initial, n_particles_final, n_injected, n_removed
):
    """
    Check mass conservation throughout simulation.

    Args:
        n_particles_initial: Initial particle count
        n_particles_final: Final particle count
        n_injected: Total particles injected
        n_removed: Total particles removed (boundaries)

    Returns:
        error: Fractional error in mass conservation
        is_conserved: True if error < 1%
    """
    expected_final = n_particles_initial + n_injected - n_removed
    error = abs(n_particles_final - expected_final) / expected_final if expected_final > 0 else 0.0
    is_conserved = error < 0.01  # Less than 1% error

    return error, is_conserved


@njit
def check_energy_conservation(v, mass, active, n_particles, E_initial):
    """
    Check kinetic energy conservation.

    Args:
        v: Particle velocities (n_particles, 3) [m/s]
        mass: Particle mass [kg]
        active: Active flags (n_particles,)
        n_particles: Number of particles
        E_initial: Initial total kinetic energy [J]

    Returns:
        E_final: Final total kinetic energy [J]
        error: Fractional error
    """
    E_final = 0.0

    for i in range(n_particles):
        if not active[i]:
            continue

        v_sq = v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2
        E_final += 0.5 * mass * v_sq

    error = abs(E_final - E_initial) / E_initial if E_initial > 0 else 0.0

    return E_final, error


@njit
def check_momentum_conservation(v, mass, active, weight, n_particles, p_initial):
    """
    Check momentum conservation.

    Args:
        v: Particle velocities (n_particles, 3) [m/s]
        mass: Particle mass [kg]
        active: Active flags (n_particles,)
        weight: Particle weights (n_particles,)
        n_particles: Number of particles
        p_initial: Initial total momentum (3,) [kg*m/s]

    Returns:
        p_final: Final total momentum (3,) [kg*m/s]
        error: Fractional error magnitude
    """
    p_final = np.zeros(3, dtype=np.float64)

    for i in range(n_particles):
        if not active[i]:
            continue

        for dim in range(3):
            p_final[dim] += mass * v[i, dim] * weight[i]

    # Compute error magnitude
    p_initial_mag = np.sqrt(p_initial[0]**2 + p_initial[1]**2 + p_initial[2]**2)
    p_final_mag = np.sqrt(p_final[0]**2 + p_final[1]**2 + p_final[2]**2)

    error = abs(p_final_mag - p_initial_mag) / p_initial_mag if p_initial_mag > 0 else 0.0

    return p_final, error


class DiagnosticTracker:
    """
    Tracks simulation diagnostics over time.

    Usage:
        tracker = DiagnosticTracker(n_steps=1000, output_interval=10)
        for step in range(n_steps):
            # ... simulation step ...
            if step % output_interval == 0:
                tracker.record(step, time, particles, ...)
        tracker.save_csv('diagnostics.csv')
        tracker.plot()
    """

    def __init__(self, n_steps: int, output_interval: int):
        """
        Initialize diagnostic tracker.

        Args:
            n_steps: Total number of simulation steps
            output_interval: Record diagnostics every N steps
        """
        self.n_outputs = n_steps // output_interval + 1
        self.output_idx = 0

        # Time series data
        self.time = np.zeros(self.n_outputs)
        self.step = np.zeros(self.n_outputs, dtype=np.int32)
        self.n_particles = np.zeros(self.n_outputs, dtype=np.int32)
        self.compression_ratio = np.zeros(self.n_outputs)
        self.density_inlet = np.zeros(self.n_outputs)
        self.density_outlet = np.zeros(self.n_outputs)
        self.mean_velocity = np.zeros(self.n_outputs)
        self.mean_temperature = np.zeros(self.n_outputs)

        # Conservation tracking
        self.mass_conservation_error = np.zeros(self.n_outputs)
        self.energy_conservation_error = np.zeros(self.n_outputs)
        self.momentum_conservation_error = np.zeros(self.n_outputs)

    def record(
        self,
        step: int,
        time: float,
        x: np.ndarray,
        v: np.ndarray,
        active: np.ndarray,
        weight: np.ndarray,
        n_particles: int,
        mass: float,
        z_inlet: float,
        z_outlet: float,
        E_initial: Optional[float] = None,
        p_initial: Optional[np.ndarray] = None,
    ):
        """
        Record diagnostics at current timestep.

        Args:
            step: Current simulation step
            time: Current simulation time [s]
            x: Particle positions (n_particles, 3) [m]
            v: Particle velocities (n_particles, 3) [m/s]
            active: Active flags (n_particles,)
            weight: Particle weights (n_particles,)
            n_particles: Number of particles
            mass: Particle mass [kg]
            z_inlet: Inlet position [m]
            z_outlet: Outlet position [m]
            E_initial: Initial energy for conservation check [J]
            p_initial: Initial momentum for conservation check [kg*m/s]
        """
        if self.output_idx >= self.n_outputs:
            return

        idx = self.output_idx

        # Basic metrics
        self.time[idx] = time
        self.step[idx] = step
        self.n_particles[idx] = np.sum(active[:n_particles])

        # Compression ratio
        CR, n_in, n_out = compute_compression_ratio(
            x, active, weight, n_particles, z_inlet, z_outlet
        )
        self.compression_ratio[idx] = CR
        self.density_inlet[idx] = n_in
        self.density_outlet[idx] = n_out

        # Mean velocity and temperature
        v_mean = 0.0
        v_sq_mean = 0.0
        n_active = 0

        for i in range(n_particles):
            if not active[i]:
                continue

            v_mag = np.sqrt(v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2)
            v_mean += v_mag
            v_sq_mean += v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2
            n_active += 1

        if n_active > 0:
            self.mean_velocity[idx] = v_mean / n_active
            self.mean_temperature[idx] = mass * (v_sq_mean / n_active) / (3.0 * kB)

        # Conservation checks
        if E_initial is not None:
            _, e_error = check_energy_conservation(v, mass, active, n_particles, E_initial)
            self.energy_conservation_error[idx] = e_error

        if p_initial is not None:
            _, p_error = check_momentum_conservation(
                v, mass, active, weight, n_particles, p_initial
            )
            self.momentum_conservation_error[idx] = p_error

        self.output_idx += 1

    def save_csv(self, filename: str):
        """
        Save diagnostic data to CSV file.

        Args:
            filename: Output CSV filename
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'step', 'time_us', 'n_particles', 'compression_ratio',
                'density_inlet', 'density_outlet', 'mean_velocity_m/s',
                'mean_temperature_K', 'mass_error', 'energy_error', 'momentum_error'
            ])

            # Data rows
            for i in range(self.output_idx):
                writer.writerow([
                    self.step[i],
                    self.time[i] * 1e6,  # Convert to microseconds
                    self.n_particles[i],
                    self.compression_ratio[i],
                    self.density_inlet[i],
                    self.density_outlet[i],
                    self.mean_velocity[i],
                    self.mean_temperature[i],
                    self.mass_conservation_error[i],
                    self.energy_conservation_error[i],
                    self.momentum_conservation_error[i],
                ])

        print(f"Diagnostics saved to {filename}")

    def plot(self, show=True, save_filename=None):
        """
        Create diagnostic plots.

        Args:
            show: Display plots interactively
            save_filename: Save figure to file (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        time_us = self.time[:self.output_idx] * 1e6

        # Plot 1: Particle count
        ax = axes[0, 0]
        ax.plot(time_us, self.n_particles[:self.output_idx], 'b-', linewidth=2)
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Active Particles', fontsize=12)
        ax.set_title('Particle Population', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 2: Compression ratio
        ax = axes[0, 1]
        ax.plot(time_us, self.compression_ratio[:self.output_idx], 'r-', linewidth=2)
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Compression Ratio', fontsize=12)
        ax.set_title('Compression Ratio vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Plot 3: Density inlet vs outlet
        ax = axes[0, 2]
        ax.plot(time_us, self.density_inlet[:self.output_idx] / 1e20, 'b-',
                linewidth=2, label='Inlet')
        ax.plot(time_us, self.density_outlet[:self.output_idx] / 1e20, 'r-',
                linewidth=2, label='Outlet')
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Density (10²⁰ m⁻³)', fontsize=12)
        ax.set_title('Density at Inlet/Outlet', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 4: Mean velocity
        ax = axes[1, 0]
        ax.plot(time_us, self.mean_velocity[:self.output_idx] / 1000, 'g-', linewidth=2)
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Velocity (km/s)', fontsize=12)
        ax.set_title('Mean Velocity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 5: Mean temperature
        ax = axes[1, 1]
        ax.plot(time_us, self.mean_temperature[:self.output_idx], 'm-', linewidth=2)
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Temperature (K)', fontsize=12)
        ax.set_title('Mean Temperature', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot 6: Conservation errors
        ax = axes[1, 2]
        if np.any(self.energy_conservation_error[:self.output_idx] > 0):
            ax.semilogy(time_us, self.energy_conservation_error[:self.output_idx],
                        'r-', linewidth=2, label='Energy')
        if np.any(self.momentum_conservation_error[:self.output_idx] > 0):
            ax.semilogy(time_us, self.momentum_conservation_error[:self.output_idx],
                        'b-', linewidth=2, label='Momentum')
        ax.axhline(y=0.01, color='k', linestyle='--', linewidth=1, label='1% threshold')
        ax.set_xlabel('Time (μs)', fontsize=12)
        ax.set_ylabel('Fractional Error', fontsize=12)
        ax.set_title('Conservation Errors', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([1e-4, 1e0])

        plt.tight_layout()

        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_filename}")

        if show:
            plt.show()

    def summary(self):
        """
        Print summary statistics.
        """
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)

        idx = self.output_idx - 1 if self.output_idx > 0 else 0

        print(f"\nFinal State:")
        print(f"  Active particles: {self.n_particles[idx]:,}")
        print(f"  Compression ratio: {self.compression_ratio[idx]:.2f}")
        print(f"  Inlet density: {self.density_inlet[idx]:.2e} m^-3")
        print(f"  Outlet density: {self.density_outlet[idx]:.2e} m^-3")
        print(f"  Mean velocity: {self.mean_velocity[idx]:.1f} m/s")
        print(f"  Mean temperature: {self.mean_temperature[idx]:.1f} K")

        print(f"\nConservation Errors (final):")
        if np.any(self.energy_conservation_error[:idx] > 0):
            print(f"  Energy: {self.energy_conservation_error[idx]:.2%}")
        if np.any(self.momentum_conservation_error[:idx] > 0):
            print(f"  Momentum: {self.momentum_conservation_error[idx]:.2%}")

        print(f"\nTime-Averaged Metrics:")
        mean_CR = np.mean(self.compression_ratio[10:idx])  # Skip first 10 for steady-state
        print(f"  Mean compression ratio: {mean_CR:.2f}")
        print(f"  Mean temperature: {np.mean(self.mean_temperature[10:idx]):.1f} K")

        print("="*70 + "\n")
