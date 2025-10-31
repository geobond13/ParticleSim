"""
Validation Framework for IntakeSIM

Base classes and utilities for validating IntakeSIM against literature
and experimental data.

Created for SBIR Phase I deliverable - Week 6 validation study.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv


@dataclass
class ValidationMetric:
    """Single validation metric with reference and simulated values."""
    name: str
    reference_value: float
    simulated_value: float
    tolerance_percent: float
    units: str

    @property
    def error_percent(self) -> float:
        """Calculate percentage error."""
        if self.reference_value == 0:
            return 0.0
        return 100.0 * (self.simulated_value - self.reference_value) / self.reference_value

    @property
    def absolute_error(self) -> float:
        """Calculate absolute error."""
        return abs(self.simulated_value - self.reference_value)

    @property
    def passes(self) -> bool:
        """Check if metric is within tolerance."""
        return abs(self.error_percent) <= self.tolerance_percent

    @property
    def status(self) -> str:
        """Return status emoji."""
        if self.passes:
            return "[PASS]"
        else:
            return "[FAIL]"

    def __str__(self) -> str:
        """String representation."""
        return (f"{self.status} {self.name}: "
                f"{self.simulated_value:.3g} {self.units} "
                f"(ref: {self.reference_value:.3g}, "
                f"error: {self.error_percent:+.1f}%)")


class ValidationCase(ABC):
    """
    Base class for validation cases.

    Provides common infrastructure for:
    - Loading reference data
    - Running simulations
    - Comparing results
    - Generating reports
    """

    def __init__(self, name: str, description: str):
        """
        Initialize validation case.

        Args:
            name: Short name (e.g., "Parodi_Intake")
            description: Longer description
        """
        self.name = name
        self.description = description
        self.metrics: List[ValidationMetric] = []
        self.reference_data: Dict = {}
        self.simulation_data: Dict = {}

    @abstractmethod
    def load_reference_data(self) -> Dict:
        """
        Load reference data from literature.

        Returns:
            Dictionary with reference values
        """
        pass

    @abstractmethod
    def run_simulation(self) -> Dict:
        """
        Run IntakeSIM simulation for this case.

        Returns:
            Dictionary with simulation results
        """
        pass

    def compare_results(self) -> Dict:
        """
        Compare simulation to reference data.

        Returns:
            Dictionary with comparison statistics
        """
        # This will be overridden by subclasses, but provides default behavior
        results = {
            'metrics': self.metrics,
            'n_pass': sum(1 for m in self.metrics if m.passes),
            'n_total': len(self.metrics),
            'pass_rate': 0.0
        }

        if len(self.metrics) > 0:
            results['pass_rate'] = results['n_pass'] / results['n_total']

        return results

    def plot_comparison(self, save_filename: Optional[str] = None, show: bool = True):
        """
        Create comparison plots.

        Args:
            save_filename: If provided, save plot to this file
            show: If True, display plot
        """
        # Default implementation - subclasses should override
        print(f"No default plot implementation for {self.name}")

    def generate_report_section(self) -> str:
        """
        Generate markdown report section.

        Returns:
            Markdown-formatted text for validation report
        """
        lines = []
        lines.append(f"## {self.name}\n")
        lines.append(f"{self.description}\n")
        lines.append("")

        # Summary
        results = self.compare_results()
        pass_rate = results['pass_rate'] * 100
        lines.append(f"**Overall:** {results['n_pass']}/{results['n_total']} metrics passing ({pass_rate:.0f}%)\n")
        lines.append("")

        # Metrics table
        lines.append("| Metric | Reference | Simulated | Error | Status |")
        lines.append("|--------|-----------|-----------|-------|--------|")

        for metric in self.metrics:
            lines.append(
                f"| {metric.name} | "
                f"{metric.reference_value:.3g} {metric.units} | "
                f"{metric.simulated_value:.3g} {metric.units} | "
                f"{metric.error_percent:+.1f}% | "
                f"{metric.status} |"
            )

        lines.append("")
        return "\n".join(lines)

    def save_results_csv(self, filename: str):
        """
        Save validation results to CSV.

        Args:
            filename: Output CSV file path
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Metric', 'Reference_Value', 'Simulated_Value',
                'Units', 'Error_Percent', 'Tolerance_Percent', 'Pass'
            ])

            # Data
            for metric in self.metrics:
                writer.writerow([
                    metric.name,
                    metric.reference_value,
                    metric.simulated_value,
                    metric.units,
                    metric.error_percent,
                    metric.tolerance_percent,
                    'PASS' if metric.passes else 'FAIL'
                ])

        print(f"Results saved to '{filename}'")

    def print_summary(self):
        """Print validation summary to console."""
        print("\n" + "="*70)
        print(f"{self.name}")
        print("="*70)
        print(f"{self.description}\n")

        results = self.compare_results()
        print(f"Overall: {results['n_pass']}/{results['n_total']} metrics passing "
              f"({results['pass_rate']*100:.0f}%)\n")

        for metric in self.metrics:
            print(metric)

        print("")


def uncertainty_analysis(
    values: np.ndarray,
    statistical_fraction: float = 0.05
) -> Tuple[float, float, float]:
    """
    Compute uncertainty budget.

    Args:
        values: Array of sampled values (e.g., from Monte Carlo)
        statistical_fraction: Systematic uncertainty as fraction of mean

    Returns:
        (mean, statistical_uncertainty, total_uncertainty)
    """
    mean = np.mean(values)
    std = np.std(values)

    # Statistical uncertainty (standard error of mean)
    statistical_unc = std / np.sqrt(len(values))

    # Systematic uncertainty (assumed from geometry, surface model, etc.)
    systematic_unc = statistical_fraction * mean

    # Total uncertainty (quadrature sum)
    total_unc = np.sqrt(statistical_unc**2 + systematic_unc**2)

    return mean, statistical_unc, total_unc


def create_comparison_plot(
    reference_data: Dict[str, np.ndarray],
    simulation_data: Dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    save_filename: Optional[str] = None,
    show: bool = True
):
    """
    Create standardized comparison plot for validation.

    Args:
        reference_data: Dictionary with 'x' and 'y' arrays for reference
        simulation_data: Dictionary with 'x' and 'y' arrays for simulation
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        save_filename: If provided, save plot to file
        show: If True, display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Reference data
    ax.plot(
        reference_data['x'], reference_data['y'],
        'o-', linewidth=2, markersize=8, color='blue',
        label='Reference (Literature)', alpha=0.7
    )

    # Simulation data
    ax.plot(
        simulation_data['x'], simulation_data['y'],
        's--', linewidth=2, markersize=8, color='red',
        label='IntakeSIM Simulation', alpha=0.7
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{save_filename}'")

    if show:
        plt.show()
    else:
        plt.close()
