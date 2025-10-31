# IntakeSIM Visualization Guide

**Purpose:** Comprehensive guide to visualization capabilities for DSMC and PIC simulations in IntakeSIM

**Last Updated:** January 2025

---

## 📊 Overview

IntakeSIM provides visualization capabilities across three levels:
1. **Real-time diagnostics** - Monitor simulation progress
2. **Validation plots** - Compare to literature benchmarks
3. **Publication-quality figures** - SBIR deliverables and papers

---

## 1. DSMC Module Visualizations

### 1.1 Time-Series Diagnostics (`DiagnosticTracker.plot()`)

**Implemented in:** `src/intakesim/diagnostics.py`

Creates 2×3 panel figure with:
```python
from intakesim.diagnostics import DiagnosticTracker

tracker = DiagnosticTracker(n_steps=1000, output_interval=10)
# ... run simulation ...
tracker.plot(show=True, save_filename='diagnostics.png')
```

**Six panels:**

1. **Particle Population**
   - X-axis: Time (μs)
   - Y-axis: Active particle count
   - Purpose: Check injection/removal balance, steady-state convergence

2. **Compression Ratio vs Time**
   - X-axis: Time (μs)
   - Y-axis: CR = n_outlet / n_inlet
   - Purpose: Monitor intake performance, detect steady-state

3. **Density at Inlet/Outlet**
   - X-axis: Time (μs)
   - Y-axis: Number density (10²⁰ m⁻³)
   - Two lines: Blue (inlet), Red (outlet)
   - Purpose: Understand compression mechanism

4. **Mean Velocity**
   - X-axis: Time (μs)
   - Y-axis: Velocity (km/s)
   - Purpose: Detect velocity changes (should drop from 7.8→~5 km/s in intake)

5. **Mean Temperature**
   - X-axis: Time (μs)
   - Y-axis: Temperature (K)
   - Purpose: Thermal equilibration monitoring

6. **Conservation Errors**
   - X-axis: Time (μs)
   - Y-axis: Fractional error (log scale)
   - Lines: Energy (red), Momentum (blue)
   - Threshold: 1% dashed line
   - Purpose: Validate simulation accuracy

**When to use:** Every DSMC simulation run for quality control

---

### 1.2 Thermal Equilibration (`examples/02_thermal_equilibration.py`)

**Two plots generated:**

#### Plot A: Temperature Evolution
```
Hot population (red) + Cold population (blue) → Equilibrium
```
- X-axis: Time (μs)
- Y-axis: Temperature (K)
- Shows: T_hot(t), T_cold(t), T_avg(t), Expected equilibrium (dashed)
- Purpose: **Validation of VHS collision model** (Week 2 checkpoint)
- Expected: Both populations converge to (T_hot + T_cold)/2

#### Plot B: Collision Rate vs Time
- X-axis: Time (μs)
- Y-axis: Collisions per timestep
- Purpose: Monitor collision frequency (should be ~constant in equilibrium)

**When to use:** VHS collision model validation, before intake simulations

---

### 1.3 Collision Rate Scaling (`examples/02_thermal_equilibration.py`)

**Log-log plot:**
```python
plt.loglog(densities, collision_rates, 'o-')
plt.loglog(densities, linear_reference, '--')
```
- X-axis: Number density (m⁻³)
- Y-axis: Collisions per timestep
- Reference line: Linear scaling (ν ∝ n)
- Purpose: **Verify collision algorithm scales correctly with density**
- Expected: Perfect overlap with linear reference

**When to use:** Unit testing collision module, debugging performance

---

### 1.4 Parameter Study Plots (`examples/06_parameter_study.py`)

**Four-panel figure:**

#### Panel 1: Compression Ratio vs L/D
- X-axis: L/D ratio (10, 15, 20, 30, 50)
- Y-axis (left): Compression ratio
- Y-axis (right, red): Clausing factor K(L/D)
- Shows: Trade-off between channel length and transmission
- Purpose: **Optimize intake geometry**
- Key insight: Shorter channels (L/D < 20) give higher transmission but lower geometric CR

#### Panel 2: Compression Ratio vs Channel Diameter
- X-axis: Channel diameter (mm)
- Y-axis: Compression ratio
- Shows: Effect of channel size on performance
- Purpose: **Honeycomb geometry optimization**
- Trade-off: Smaller d → higher per-channel CR, but fewer channels overall

#### Panel 3: Compression Ratio vs Altitude
- X-axis: Altitude (km) [200, 212, 225, 237, 250]
- Y-axis: Compression ratio
- Shows: VLEO performance envelope
- Purpose: **Mission design** - identify altitude limits
- Expected: CR decreases with altitude (lower ambient density)

#### Panel 4: Transmission Efficiency by Study Type
- Bar chart: Mean efficiency for L/D, diameter, altitude studies
- Values labeled on bars
- Purpose: **Compare sensitivity** of different design variables

**When to use:** Design optimization, SBIR proposal figures

---

### 1.5 Velocity Distribution Functions (`diagnostics.py`)

**Histogram plot (to be implemented in example):**
```python
v_bins = np.linspace(0, 10000, 50)  # m/s
histogram = compute_velocity_distribution(v, active, n_particles, v_bins)

plt.bar(v_bins[:-1], histogram, width=np.diff(v_bins))
plt.xlabel('Velocity (m/s)')
plt.ylabel('Count')
```

**Two use cases:**

#### A. Maxwellian Distribution Check
- Plot: Histogram + analytical Maxwell-Boltzmann overlay
```
f(v) = 4π n (m/2πkT)^(3/2) v² exp(-mv²/2kT)
```
- Purpose: **Validate thermal equilibrium** (Week 3 checkpoint)
- Expected: Histogram matches analytical curve with R² > 0.95

#### B. Intake Inlet/Outlet Comparison
- Two histograms: Inlet (blue) vs Outlet (red)
- Purpose: **Understand velocity thermalization in intake**
- Expected: Outlet peak shifted to lower velocity (CLL reflection effect)

**When to use:**
- Validation: Check Maxwell-Boltzmann distribution
- Physics insight: Visualize non-equilibrium effects

---

### 1.6 Spatial Density Profiles (`diagnostics.py`)

**1D density profile along intake:**
```python
z_bins = np.linspace(0, domain_length, 50)
density = compute_density_profile(x, active, weight, n_particles, z_bins)

plt.plot(z_bins[:-1], density / 1e20, linewidth=2)
plt.xlabel('Axial Position (m)')
plt.ylabel('Density (10²⁰ m⁻³)')
plt.axvline(z_inlet, color='green', linestyle='--', label='Inlet')
plt.axvline(z_outlet, color='red', linestyle='--', label='Outlet')
```

**Purpose:**
- Visualize **compression gradient** through intake
- Identify regions of particle buildup or depletion
- Debug boundary condition issues

**Expected profile:**
- Flat at inlet (freestream)
- Gradual increase through tapered section
- Plateau at outlet (compressed chamber)

**When to use:**
- Geometry validation
- Understanding flow physics
- Debugging unexpected CR values

---

### 1.7 Temperature Profiles (`diagnostics.py`)

**1D temperature profile:**
```python
T_profile = compute_temperature_profile(v, mass, active, n_particles, z_bins, x)

plt.plot(z_bins[:-1], T_profile, linewidth=2)
plt.xlabel('Axial Position (m)')
plt.ylabel('Temperature (K)')
plt.axhline(T_wall, color='gray', linestyle='--', label='Wall temperature')
```

**Purpose:**
- Visualize **thermalization due to CLL reflection**
- Verify energy accommodation at walls
- Expected: T drops from ~1000 K (freestream) → ~500-700 K (after multiple wall collisions)

**When to use:**
- CLL model validation (Week 3)
- Understanding energy dissipation mechanism

---

### 1.8 Validation Comparison Plots (`validation/`)

**Implemented in:** `validation_framework.py`

#### A. Romano Intake Benchmark
```python
from validation.romano_validation import RomanoIntakeValidation

validation = RomanoIntakeValidation()
comparison = validation.compare_results()
validation.plot_comparison()
```

**Creates:**
1. **Bar chart:** IntakeSIM vs Romano η_c with error bars
2. **Altitude sweep:** η_c vs altitude (150-250 km) - two lines
3. **Error plot:** Percentage difference vs altitude

**Purpose:** Primary DSMC validation for SBIR Phase I

#### B. Parodi Intake Comparison
```python
from validation.parodi_validation import ParodiIntakeValidation

validation = ParodiIntakeValidation()
comparison = validation.compare_results()
```

**Creates:**
1. **Species-specific CR:** Bar chart for N₂, O₂, O
2. **System-level CR:** IntakeSIM local CR vs Parodi system CR
3. **Error analysis:** Identifies geometry approximation gap

**Purpose:** Advanced validation target for coupled system (Week 12-13)

---

## 2. PIC Module Visualizations (Week 7-13)

### 2.1 Plasma Density Evolution

**Time-series plot:**
```python
plt.plot(time_us, n_plasma / 1e17, linewidth=2)
plt.xlabel('Time (μs)')
plt.ylabel('Plasma Density (10¹⁷ m⁻³)')
plt.axhline(1.65, color='red', linestyle='--', label='Parodi target')
```

**Purpose:** Monitor ionization buildup to steady-state
**Expected:** Exponential rise to plateau at ~1.65×10¹⁷ m⁻³ (Parodi validation)

---

### 2.2 Electron Energy Distribution Function (EEDF)

**Critical PIC diagnostic:**
```python
E_bins = np.logspace(-1, 2, 100)  # 0.1 to 100 eV
eedf = compute_eedf(v_electrons, weights)

plt.semilogx(E_bins, eedf, linewidth=2)
plt.xlabel('Electron Energy (eV)')
plt.ylabel('EEDF (eV⁻³/² m⁻³)')
```

**Overlays:**
- Analytical Maxwellian: f(E) ∝ √E exp(-E/T_e)
- Expected T_e = 7.8 eV (from Parodi)

**Purpose:**
- **Validate MCC module** (Week 8)
- **RF heating model** (Week 10)
- Verify non-Maxwellian tail at high energy (ionization region)

**When to use:** Every PIC simulation - THE key plasma diagnostic

---

### 2.3 Spatial Plasma Profiles

**1D profiles along thruster axis (r or z):**
```python
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Panel 1: Density
axes[0].plot(z_bins, n_e / 1e17, label='Electrons')
axes[0].plot(z_bins, n_ions / 1e17, label='Ions')
axes[0].set_ylabel('Density (10¹⁷ m⁻³)')

# Panel 2: Potential
axes[1].plot(z_bins, phi_profile)
axes[1].set_ylabel('Potential (V)')

# Panel 3: Electric Field
axes[2].plot(z_bins, E_field / 1e3)
axes[2].set_xlabel('Axial Position (m)')
axes[2].set_ylabel('Electric Field (kV/m)')
```

**Purpose:**
- Visualize **sheath structure** near walls
- Verify **quasineutrality** in bulk plasma (n_e ≈ n_ions)
- Validate **Poisson solver** (Week 7)

**Expected features:**
- Flat n_e, n_ions in bulk
- Sharp drop at sheath edge (s ~ 5λ_D)
- Potential drop across sheath (φ_sheath ~ T_e/e)

---

### 2.4 Power Balance Waterfall Chart

**CRITICAL VALIDATION PLOT:**
```python
powers = {
    'RF Input': P_rf,
    'Ionization': P_ionization,
    'Excitation': P_excitation,
    'Ion Loss': P_ion_loss,
    'Electron Loss': P_electron_loss,
    'Net': P_in - P_out
}

fig, ax = plt.subplots(figsize=(8, 6))
y = np.arange(len(powers))
ax.barh(y, list(powers.values()), color=['green', 'red', 'orange', 'blue', 'purple', 'black'])
ax.set_yticks(y)
ax.set_yticklabels(list(powers.keys()))
ax.axvline(0, color='k', linewidth=1)
ax.set_xlabel('Power (W)')
ax.set_title('PIC Power Balance (Must be <10% error)')
```

**Requirement:** |P_in - P_out| / P_in < 10%

**Purpose:** **Mandatory validation** for all PIC simulations (Week 9, 11)

**When to use:** EVERY PIC run - if this fails, physics is wrong!

---

### 2.5 Sheath Potential Profile

**Validation against Child-Langmuir law:**
```python
plt.plot(x_sheath / lambda_D, phi_sheath / Te_eV, linewidth=2, label='PIC')

# Analytical Child-Langmuir
x_theory = np.linspace(0, 5, 100)
phi_theory = (x_theory / 5)**1.33  # Approximation
plt.plot(x_theory, phi_theory, '--', color='red', label='Child-Langmuir')

plt.xlabel('Distance from Wall (λ_D)')
plt.ylabel('Potential (T_e/e)')
plt.legend()
```

**Purpose:** Validate sheath physics (Week 9)
**Validation target:** Sheath thickness s = 5λ_D within 20%

---

### 2.6 Turner CCP Benchmark Comparison

**Multi-panel validation plot:**
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Density vs Position
axes[0,0].plot(z_intakesim, n_e_intakesim, 'o', label='IntakeSIM')
axes[0,0].plot(z_turner, n_e_turner, '-', label='Turner 2013')

# Panel 2: Potential vs Position
axes[0,1].plot(z_intakesim, phi_intakesim, 'o')
axes[0,1].plot(z_turner, phi_turner, '-')

# Panel 3: EEDF Comparison
axes[1,0].loglog(E_intakesim, eedf_intakesim, 'o')
axes[1,0].loglog(E_turner, eedf_turner, '-')

# Panel 4: Ionization Rate
axes[1,1].plot(z_intakesim, ionization_rate_intakesim, 'o')
axes[1,1].plot(z_turner, ionization_rate_turner, '-')
```

**Purpose:** **Primary PIC validation** (Week 9 checkpoint)
**Requirement:** All four panels within 20% of Turner benchmark

---

### 2.7 Ion Energy Distribution at Extraction

**Histogram at thruster exit plane:**
```python
E_ions_eV = 0.5 * m_ion * v_ions_axial**2 / e

plt.hist(E_ions_eV, bins=50, alpha=0.7, density=True)
plt.xlabel('Ion Energy (eV)')
plt.ylabel('Probability Density')
plt.axvline(np.mean(E_ions_eV), color='red', linestyle='--', label=f'Mean: {np.mean(E_ions_eV):.1f} eV')
```

**Purpose:**
- **Thrust calculation:** F = ṁ * v_exhaust
- Validate ion acceleration through grids
- Compare to Parodi thrust target (480 μN)

**Expected:** Bi-modal distribution
- Low-energy peak: ~5-10 eV (CEX ions)
- High-energy peak: ~50-100 eV (accelerated beam ions)

---

## 3. Coupled System Visualizations (Week 11-13)

### 3.1 Mass Flow Coupling Diagram

**Sankey-style flow diagram:**
```
Freestream → Intake → Compression → Ionization → Extraction → Thrust
   [100%]     [90%]      [475×]       [5%]         [95%]      [4.5%]
```

**Purpose:** Visualize **mass utilization efficiency** through entire ABEP system
**Identifies:** Bottlenecks (e.g., low ionization fraction)

---

### 3.2 Neutral Depletion Convergence

**Iteration history plot:**
```python
plt.plot(iteration, n_neutral / n_neutral_initial, 'o-', linewidth=2)
plt.axhline(1.0, color='gray', linestyle='--')
plt.xlabel('Coupling Iteration')
plt.ylabel('Neutral Density (relative)')
plt.title('DSMC-PIC Coupling Convergence')
```

**Convergence criterion:** Δn / n < 5% between iterations
**Expected:** 10-20% depletion after 5-10 iterations

**Purpose:** Validate iterative DSMC→PIC coupling (Week 13)

---

### 3.3 Parodi System-Level Validation

**Comprehensive comparison figure (4 panels):**

#### Panel 1: Compression Ratios
```
Bar chart: N₂, O₂, O
Blue: Parodi (475, 90, N/A)
Red: IntakeSIM (measured)
Error bars: ±30%
```

#### Panel 2: Plasma Parameters
```
Bar chart: n_plasma, T_e, T_ion
Target lines from Parodi
```

#### Panel 3: Thrust vs Power
```
Scatter plot:
X-axis: RF Power (W)
Y-axis: Thrust (μN)
Point: IntakeSIM (20W, 480μN target)
Comparison: Cifali experimental envelope
```

#### Panel 4: Energy Flow Diagram
```
Input: Kinetic (spacecraft motion) + RF (20W)
Losses: Wall heating, ionization, plume divergence
Output: Thrust (480 μN)
```

**Purpose:** **Final SBIR Phase I validation** - demonstrates system-level accuracy

---

## 4. Publication-Quality Figure Specifications

### 4.1 SBIR Proposal Figures

**Requirements:**
- **Resolution:** 300 DPI minimum
- **Format:** PNG or PDF (vector)
- **Size:** Full page (7" wide) or half-page (3.5" wide)
- **Fonts:** Arial or Helvetica, 10-12 pt labels, 14 pt titles
- **Colors:** Colorblind-safe palette (use `seaborn` color_blind)
- **Grid:** Light gray (alpha=0.3), behind data

**Example code:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("colorblind")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'

fig, ax = plt.subplots(figsize=(7, 5))
# ... plot code ...
plt.savefig('figure1_intake_compression.png', dpi=300, bbox_inches='tight')
```

---

### 4.2 Journal Paper Figures

**Additional requirements:**
- **Captions:** Self-contained (5-10 sentences)
- **Panel labels:** (a), (b), (c) in bold 14 pt
- **Error bars:** Always include for experimental/stochastic data
- **Legends:** Inside plot area when possible, 10 pt
- **Units:** Always in square brackets [m/s], not parentheses

**Recommended figures for IntakeSIM paper:**
1. **Fig 1:** Schematic of ABEP system (intake → ionization → thruster)
2. **Fig 2:** Validation - Romano benchmark (η_c vs altitude)
3. **Fig 3:** Parameter study results (4-panel from Section 1.4)
4. **Fig 4:** Velocity distributions (inlet vs outlet, Maxwell-Boltzmann fit)
5. **Fig 5:** PIC validation - Turner benchmark (4-panel from Section 2.6)
6. **Fig 6:** Coupled system - Parodi comparison (4-panel from Section 3.3)

---

## 5. Interactive Visualizations (Future: Phase II)

### 5.1 3D Particle Trajectories (for SPARTA/PICLas)

**Using Mayavi or ParaView:**
```python
from mayavi import mlab

# Particle positions colored by velocity magnitude
mlab.points3d(x, y, z, v_mag, scale_mode='none', scale_factor=0.001)
mlab.axes()
mlab.colorbar(title='Velocity (m/s)')
mlab.show()
```

**Purpose:**
- Visualize **3D intake geometry** (multi-channel honeycomb)
- Understand **particle trajectories** through complex geometry
- Publication figure for SBIR Phase II

**When to use:** Only after transitioning to 3D simulation (Option 3)

---

### 5.2 Animated Plasma Evolution

**MP4 video of PIC discharge:**
```python
import matplotlib.animation as animation

fig, ax = plt.subplots()

def animate(frame):
    ax.clear()
    ax.scatter(x_ions[frame], y_ions[frame], s=1, c='red', alpha=0.5)
    ax.scatter(x_electrons[frame], y_electrons[frame], s=0.5, c='blue', alpha=0.5)
    ax.set_title(f'Time: {frame*dt*1e6:.1f} μs')

anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50)
anim.save('plasma_evolution.mp4', fps=30)
```

**Purpose:**
- Conference presentations
- Investor demonstrations
- Understanding transient behavior

**When to use:** Phase II, after steady-state validation complete

---

## 6. Quick Reference: Which Plot When?

### DSMC Development

| Task | Plot Type | When | Section |
|------|-----------|------|---------|
| Verify VHS collisions | Thermal equilibration | Week 2 | 1.2 |
| Check conservation laws | Time-series diagnostics | Every run | 1.1 |
| Validate CLL reflection | Temperature profile | Week 3 | 1.7 |
| Optimize L/D ratio | Parameter study | Week 5 | 1.4 |
| Romano validation | η_c comparison | Week 6 | 1.8A |
| Debug compression | Density profile | As needed | 1.6 |

### PIC Development

| Task | Plot Type | When | Section |
|------|-----------|------|---------|
| Validate Poisson solver | Spatial profiles (φ, E) | Week 7 | 2.3 |
| Check MCC chemistry | EEDF | Week 8 | 2.2 |
| Turner benchmark | 4-panel comparison | Week 9 | 2.6 |
| RF heating validation | Power balance | Week 10 | 2.4 |
| Parodi plasma validation | Density evolution | Week 11 | 2.1 |
| Thrust calculation | Ion energy distribution | Week 12 | 2.7 |

### Validation & Reporting

| Deliverable | Plot Type | When | Section |
|-------------|-----------|------|---------|
| Week 6 checkpoint | Romano η_c | DSMC complete | 1.8A |
| Week 9 checkpoint | Turner benchmark | PIC complete | 2.6 |
| Week 13 checkpoint | Parodi system | Coupled complete | 3.3 |
| SBIR Phase I report | Figs 1-6 | Month 4 | 4.1 |
| Journal paper | All publication figures | Post Phase I | 4.2 |

---

## 7. Recommended Python Packages

```bash
# Core visualization
pip install matplotlib>=3.7.0      # 2D plotting
pip install seaborn>=0.12.0        # Statistical plots, color palettes

# Advanced visualization (Phase II)
pip install mayavi                 # 3D particle visualization
pip install plotly                 # Interactive web-based plots
pip install h5py                   # HDF5 file I/O for large datasets

# Publication tools
pip install scipy                  # Curve fitting, statistics
pip install uncertainties          # Error propagation
```

**Color palette recommendation:**
```python
import seaborn as sns
sns.set_palette("colorblind")  # Accessible to all readers
```

---

## 8. Data Export for External Visualization

### 8.1 ParaView (for 3D SPARTA/PICLas)

**VTK format export:**
```python
from pyevtk.hl import pointsToVTK

# Export particle data
pointsToVTK(
    './output',
    x, y, z,
    data={'velocity': v_mag, 'species': species_id}
)
```

**When to use:** Phase II with 3D simulations

---

### 8.2 CSV Export (for custom analysis)

**Already implemented in `DiagnosticTracker.save_csv()`:**
```python
tracker.save_csv('diagnostics.csv')
```

**Columns:**
- step, time_us, n_particles
- compression_ratio, density_inlet, density_outlet
- mean_velocity_m/s, mean_temperature_K
- mass_error, energy_error, momentum_error

**When to use:**
- External analysis in MATLAB, Origin, Excel
- Long-term archival of simulation data

---

## 9. Examples Gallery

### Minimal Working Example: Create Diagnostic Plot

```python
import numpy as np
from intakesim.diagnostics import DiagnosticTracker
from intakesim.particles import ParticleArrayNumba
# ... imports ...

# Setup
particles = ParticleArrayNumba(max_particles=10000)
tracker = DiagnosticTracker(n_steps=1000, output_interval=10)

# Simulation loop
for step in range(1000):
    # ... simulation code ...

    if step % 10 == 0:
        tracker.record(
            step=step,
            time=step*dt,
            x=particles.x,
            v=particles.v,
            active=particles.active,
            weight=particles.weight,
            n_particles=particles.n_particles,
            mass=SPECIES['N2'].mass,
            z_inlet=0.01,
            z_outlet=0.03
        )

# Generate plots
tracker.plot(show=True, save_filename='my_diagnostics.png')
tracker.save_csv('my_diagnostics.csv')
tracker.summary()  # Print statistics
```

**Output:** 6-panel diagnostic figure + CSV file + terminal summary

---

### Example: Custom Validation Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Your simulation results
intakesim_cr = 7.4
intakesim_cr_std = 1.2

# Literature target
romano_cr = 10.0
romano_tolerance = 0.30  # ±30%

# Create comparison plot
fig, ax = plt.subplots(figsize=(6, 5))

x_pos = [0, 1]
heights = [romano_cr, intakesim_cr]
errors = [romano_cr * romano_tolerance, intakesim_cr_std]
colors = ['blue', 'red']
labels = ['Romano (2021)', 'IntakeSIM']

bars = ax.bar(x_pos, heights, yerr=errors, color=colors, alpha=0.7,
              capsize=10, label=labels)

ax.set_ylabel('Compression Ratio', fontsize=14)
ax.set_title('DSMC Intake Validation', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val, err in zip(bars, heights, errors):
    ax.text(bar.get_x() + bar.get_width()/2, val + err + 1,
            f'{val:.1f} ± {err:.1f}',
            ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('validation_romano.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 10. Troubleshooting Common Issues

### Issue 1: Plots are empty / No data

**Diagnosis:** `output_idx = 0` in DiagnosticTracker
**Fix:** Check that `tracker.record()` is being called in simulation loop

### Issue 2: Conservation errors >10%

**Diagnosis:** Numerical instability, timestep too large, or particle leakage
**Fix:**
1. Reduce dt by factor of 2
2. Check boundary conditions (are particles escaping?)
3. Verify Numba compilation succeeded

### Issue 3: Compression ratio = 0 or infinity

**Diagnosis:** No particles in inlet or outlet sampling region
**Fix:**
1. Check `z_inlet` and `z_outlet` match geometry
2. Increase `dz_sample` for sampling width
3. Run longer to reach steady-state

### Issue 4: Matplotlib import error

**Diagnosis:** Matplotlib not installed or wrong environment
**Fix:**
```bash
conda activate intakesim
pip install matplotlib
```

### Issue 5: Figure resolution too low for publication

**Fix:**
```python
plt.savefig('figure.png', dpi=300, bbox_inches='tight')  # Not dpi=150!
```

---

## 11. Contact & Contributions

**Questions about visualizations?**
- Check examples in `examples/` directory
- Review `diagnostics.py` docstrings
- Open GitHub issue for bugs or feature requests

**Want to add new visualizations?**
1. Implement in `src/intakesim/diagnostics.py`
2. Add example in `examples/`
3. Document in this guide
4. Submit pull request

---

## 12. References

**Visualization best practices:**
- Tufte, E. "The Visual Display of Quantitative Information" (2001)
- Rougier, N. et al. "Ten Simple Rules for Better Figures" (2014) PLoS Comp Bio

**Domain-specific visualization:**
- Birdsall & Langdon (1991) - PIC diagnostic plots (Chapter 15)
- Bird (1994) - DSMC visualization techniques (Chapter 13)

**Color palettes:**
- ColorBrewer (colorbrewer2.org) - Colorblind-safe palettes
- Seaborn colorblind palette - Default recommendation

---

**Document Version:** 1.0 (January 2025)
**Maintainer:** IntakeSIM Development Team
**Last Updated:** Generated via Claude Code research session

---

## Appendix: Complete Plotting Code Templates

### Template 1: Multi-Panel Time-Series
```python
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
# ... subplot code ...
plt.tight_layout()
plt.savefig('time_series.png', dpi=300, bbox_inches='tight')
```

### Template 2: Validation Comparison
```python
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([...], color=['blue', 'red'], alpha=0.7)
ax.axhline(target, color='gray', linestyle='--')
# ... labels ...
plt.savefig('validation.png', dpi=300, bbox_inches='tight')
```

### Template 3: Phase Space Plot
```python
fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(x, v_x, c=energy, cmap='viridis', s=1, alpha=0.5)
cbar = plt.colorbar(scatter, label='Energy (eV)')
plt.savefig('phase_space.png', dpi=300, bbox_inches='tight')
```
