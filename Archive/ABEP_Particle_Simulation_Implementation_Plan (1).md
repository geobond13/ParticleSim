# AeriSat ABEP Particle-Based Simulation Implementation Plan

**Document Version:** 1.0  
**Date:** October 29, 2025  
**Author:** AeriSat Systems CTO Office  
**Classification:** Technical Implementation Roadmap

---

## Executive Summary

This document provides a comprehensive implementation plan for developing particle-based simulation capabilities for AeriSat's Air-Breathing Electric Propulsion (ABEP) system, building upon the methodology demonstrated by Parodi et al. (2025) in their paper "Particle-based Simulation of an Air-Breathing Electric Propulsion System."

**Primary Objective (Option 2):** Develop simplified 1D/2D particle methods for intake compression validation and plasma density evolution modeling, providing higher-fidelity physics than analytical models while remaining computationally tractable.

**Stretch Objective (Option 3):** Establish architecture and initial implementation for full 3D particle simulation capability using DSMC and PIC-MCC methods with unstructured mesh support.

**Timeline:** 8-12 weeks for Option 2 core implementation, 6-12 months for Option 3 complete system.

**Value Proposition:**
- Validate and refine AeriSat's existing analytical models against particle-based physics
- Provide high-fidelity predictions for SBIR proposals and investor presentations
- Enable optimization of intake geometry and thruster operating conditions
- Build toward flight-qualified simulation tools for mission planning

---

## 1. Technical Background

### 1.1 Motivation for Particle Methods

**Limitations of Current Analytical Models:**
- Assume Maxwell-Boltzmann equilibrium distributions
- Use lumped-parameter approximations for compression and ionization
- Cannot capture kinetic effects critical at VLEO pressures
- Limited accuracy for non-equilibrium flow physics

**Advantages of Particle Methods:**
- Naturally handle non-equilibrium velocity distributions
- Accurately model rarefied gas dynamics (Kn > 0.1)
- Capture particle-surface interactions at molecular level
- Enable prediction of ion energy distributions and plume divergence

**Key Physics Regimes in ABEP:**

| Region | Knudsen Number | Method | Primary Physics |
|--------|----------------|--------|-----------------|
| Freestream (200 km) | Kn >> 1 | Free molecular | Ballistic trajectories |
| Intake | 0.1 < Kn < 10 | DSMC | Transitional flow |
| Ionization chamber | Kn ~ 0.01 | PIC-MCC | Collisional plasma |
| Plume | Kn > 1 | PIC or hybrid | Rarefied plasma |

### 1.2 Method Overview

#### Direct Simulation Monte Carlo (DSMC)
**Purpose:** Simulate neutral particle flow through intake and compression

**Key Features:**
- Decouples particle motion from collisions
- Particles move ballistically, then collide stochastically
- Recovers continuum transport properties (viscosity, thermal conductivity)
- Handles surface interactions (reflection, accommodation, recombination)

**Governing Principle:**
```
For each timestep:
  1. Move particles ballistically
  2. Handle wall collisions
  3. Index particles to cells
  4. Perform binary collisions within cells
  5. Sample macroscopic properties
```

#### Particle-in-Cell (PIC) with Monte Carlo Collisions (MCC)
**Purpose:** Simulate plasma generation, confinement, and acceleration

**Key Features:**
- Charged particles coupled through electromagnetic fields
- Fields solved on grid (Finite Element or Finite Difference)
- Collisions with neutrals via Monte Carlo method
- Can use implicit time integration for disparate timescales

**Governing Equations:**
```
Particles:  dx/dt = v,  dv/dt = (q/m)(E + v × B)
Fields:     ∇·E = ρ/ε₀,  ∇×B = μ₀J + μ₀ε₀∂E/∂t
Collisions: Stochastic sampling based on cross-sections
```

### 1.3 Parodi et al. Validation Data

**From their 3U CubeSat simulation (200 km altitude):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Freestream density | 4.2×10¹⁷ m⁻³ | N₂+O mixture |
| Intake CR (N₂) | 475 | vs. 100-300 typical |
| Intake CR (O₂) | 90 | Lower due to recombination |
| Thruster density | 1.0×10²⁰ m⁻³ | Nitrogen only |
| Plasma density | 1.65×10¹⁷ m⁻³ | Peak value |
| Electron temp | 7.8 eV | Average, 9.4 eV effective |
| RF power absorbed | 20 W | At 13.56 MHz |
| Thrust achieved | 480 μN | From ion momentum flux |
| Ion energy (peak 1) | 350 eV | Low-frequency regime |
| Ion energy (peak 2) | 1250 eV | High-energy population |

**Key Findings:**
1. Compression ratio strongly depends on gas-surface accommodation
2. RF discharge creates oscillating azimuthal drift with phase-lagged heating
3. Ion energy distribution shows multiple peaks due to RF phase-dependent acceleration
4. Global model predictions match particle simulation averages well

---

## 2. Option 2: Simplified Particle Methods - Core Implementation

### 2.1 Scope and Objectives

**Primary Goals:**
1. Implement 1D DSMC for intake axial compression validation
2. Develop 0D/1D PIC-MCC for plasma chamber analysis
3. Validate against Parodi et al. results and AeriSat analytical models
4. Provide compression ratio and ionization efficiency predictions
5. Enable rapid parametric studies for design optimization

**Non-Goals (deferred to Option 3):**
- Full 3D geometry representation
- Unstructured mesh generation
- Coupled DSMC-PIC simulation
- Plume expansion modeling
- Self-consistent electromagnetic field calculation

**Success Criteria:**
- [ ] DSMC compression ratio within 10% of analytical model
- [ ] Plasma density predictions within 20% of global model
- [ ] Run time < 1 hour on laptop for parametric sweep
- [ ] Validated against at least 3 test cases from literature
- [ ] Documented, version-controlled, and CI-tested code

### 2.2 Architecture Overview

```
aerisat-particle-sim/
├── README.md
├── requirements.txt
├── setup.py
├── docs/
│   ├── theory.md              # Mathematical foundations
│   ├── validation.md          # Test cases and results
│   └── user_guide.md          # How to run simulations
├── src/
│   ├── aerisat_psim/
│   │   ├── __init__.py
│   │   ├── constants.py       # Physical constants
│   │   ├── particles.py       # Particle class and containers
│   │   ├── mesh.py            # 1D mesh and spatial indexing
│   │   ├── dsmc/
│   │   │   ├── __init__.py
│   │   │   ├── mover.py       # Ballistic motion
│   │   │   ├── collisions.py  # Binary collision models
│   │   │   ├── surfaces.py    # Wall interaction models
│   │   │   └── sampling.py    # Property computation
│   │   ├── pic/
│   │   │   ├── __init__.py
│   │   │   ├── mover.py       # Lorentz force integration
│   │   │   ├── field_solver.py # Poisson/field equations
│   │   │   ├── mcc.py         # Monte Carlo collisions
│   │   │   └── sources.py     # Ionization and heating
│   │   ├── cross_sections/
│   │   │   ├── __init__.py
│   │   │   ├── elastic.py     # Elastic scattering
│   │   │   ├── ionization.py  # Ionization reactions
│   │   │   └── data/          # LXCat data files
│   │   ├── diagnostics.py     # Output and visualization
│   │   └── utils.py           # Helper functions
├── tests/
│   ├── test_dsmc.py
│   ├── test_pic.py
│   └── test_integration.py
├── examples/
│   ├── 01_dsmc_intake.py      # Intake compression
│   ├── 02_pic_discharge.py    # RF plasma chamber
│   └── 03_coupled_system.py   # One-way coupling
└── validation/
    ├── parodi_comparison.py   # Reproduce Parodi results
    └── analytical_comparison.py # Compare to aerisat-abep-model.py
```

### 2.3 Implementation Phases

#### Phase 1: Core DSMC Implementation (2-3 weeks)

**Week 1: Particle Motion and Mesh**
- [ ] Implement Particle class with position, velocity, species
- [ ] Create 1D uniform mesh with cell indexing
- [ ] Implement ballistic motion with periodic/outflow boundaries
- [ ] Add basic diagnostics (density, velocity, temperature profiles)

**Week 2: Collision Models**
- [ ] Variable Hard Sphere (VHS) collision model
- [ ] Binary collision selection algorithm (Majorant Collision Frequency)
- [ ] Post-collision velocity assignment (isotropic scattering)
- [ ] Implement N₂-N₂, O-O, N₂-O collision pairs

**Week 3: Surface Interactions**
- [ ] Cercignani-Lampis-Lord (CLL) reflection model
- [ ] Surface recombination (O → O₂)
- [ ] Thermal accommodation with wall temperature
- [ ] Validate against free molecular flow limit

**Deliverable:** Working 1D DSMC code with validation against Chapman-Enskog transport

**Test Case 1: Couette Flow**
```python
# Validate viscosity recovery
L = 0.01  # Gap width [m]
T = 300   # Temperature [K]
n = 1e20  # Number density [m^-3]
v_wall = 100  # Wall velocity [m/s]

sim = DSMC1D(length=L, n_cells=50)
sim.initialize_particles(n, T, species='N2')
sim.walls[1].velocity = v_wall
sim.run(t_final=1e-3)

# Check: tau_xy = mu * dv/dx
mu_kinetic = 0.5 * m * n * lambda_mfp * v_thermal
mu_measured = sim.compute_shear_stress()
assert abs(mu_measured - mu_kinetic) / mu_kinetic < 0.05
```

**Test Case 2: Normal Shock**
```python
# Rankine-Hugoniot relations
M1 = 2.0  # Upstream Mach number
rho1, u1, T1 = 1.0, M1*a1, 300

sim = DSMC1D(length=0.1, n_cells=100)
sim.initialize_shock(M1, rho1, T1)
sim.run(t_final=1e-3)

# Validate density, velocity, temperature jumps
assert sim.compression_ratio() == pytest.approx(expected_CR, rel=0.10)
```

#### Phase 2: DSMC Intake Application (1-2 weeks)

**Week 4: Intake Geometry**
- [ ] Implement axisymmetric 1D approximation for tapered duct
- [ ] Add inlet boundary with orbital velocity injection
- [ ] Model area variation along axis
- [ ] Parametric geometry definition

**Week 5: Parametric Studies**
- [ ] Vary accommodation coefficients (0.8 ≤ σ_t ≤ 1.0)
- [ ] Scan altitude (180-250 km)
- [ ] Test different intake lengths and taper angles
- [ ] Generate compression ratio maps

**Deliverable:** DSMC intake analysis tool with parametric study capability

**Validation Target:**
```python
# Reproduce Parodi et al. intake performance
intake = DSMCIntake(
    length=0.180,      # 180 mm
    d_inlet=0.060,     # 60 mm diameter
    d_outlet=0.030,    # 30 mm diameter
    wall_temp=700,     # K
    sigma_n=1.0,       # Normal accommodation
    sigma_t=0.9        # Tangential accommodation
)

atm = Atmosphere(altitude=200e3, species=['N2', 'O'])
intake.inject_freestream(atm, velocity=7800)
intake.run(t_final=10e-3)

CR_N2 = intake.compression_ratio('N2')
print(f"N2 Compression Ratio: {CR_N2:.1f}")
# Target: 400-500 (Parodi reported 475)
```

#### Phase 3: Core PIC Implementation (2-3 weeks)

**Week 6: Field Solver and Particle Motion**
- [ ] Implement 1D Poisson solver (Finite Difference)
- [ ] Charge deposition (linear weighting)
- [ ] Electric field interpolation to particle positions
- [ ] Leap-frog particle pusher (Boris algorithm)

**Week 7: Monte Carlo Collisions**
- [ ] Load cross-section data from LXCat (Biagi database)
- [ ] Null-collision method (Vahedi algorithm)
- [ ] Elastic electron-neutral scattering
- [ ] Electron-impact ionization

**Week 8: RF Discharge Model**
- [ ] Time-varying azimuthal electric field
- [ ] Power absorption calculation
- [ ] Simple feedback controller for target power

**Deliverable:** 1D PIC-MCC plasma discharge simulator

**Test Case 3: Plasma Sheath**
```python
# Child-Langmuir sheath validation
L = 0.01  # Domain length [m]
n0 = 1e16 # Plasma density [m^-3]
Te = 3    # Electron temperature [eV]

sim = PIC1D(length=L, n_cells=100, dt=1e-11)
sim.initialize_plasma(n0, Te)
sim.boundaries[0].type = 'dielectric'  # Floating wall
sim.boundaries[1].type = 'grounded'
sim.run(t_final=1e-6)

# Check Debye sheath thickness
lambda_D = sqrt(eps0 * Te / (n0 * e))
sheath_thickness = sim.measure_sheath_thickness()
assert sheath_thickness == pytest.approx(5*lambda_D, rel=0.20)
```

**Test Case 4: Capacitive Discharge**
```python
# Compare to analytical CCP model
L = 0.05       # Gap [m]
f = 13.56e6    # RF frequency [Hz]
V_rf = 500     # RF voltage [V]
p = 10         # Pressure [Pa]
n_neutral = p / (kB * 300)

sim = PIC1D(length=L, n_cells=200, dt=1/(100*f))
sim.add_rf_drive(V_rf, f)
sim.add_neutral_background('N2', n_neutral, 300)
sim.run(t_final=100/f)  # 100 RF cycles

# Validate plasma density and electron temperature
n_plasma = sim.average_density()
Te = sim.average_electron_temperature()
# Compare to global model predictions
```

#### Phase 4: PIC Thruster Application (1-2 weeks)

**Week 9-10: RF Ionization Chamber**
- [ ] Implement 0D global model for comparison
- [ ] 1D axial PIC with fixed neutral background
- [ ] RF power deposition and control
- [ ] Ion extraction and current measurement

**Deliverable:** PIC analysis of AeriSat ionization chamber

**Validation Target:**
```python
# Compare to Parodi et al. thruster results
chamber = PICThruster(
    length=0.06,           # 60 mm chamber
    diameter=0.03,         # 30 mm
    n_neutral=1e20,        # m^-3 from DSMC
    T_neutral=700,         # K
    P_rf_target=20,        # W
    f_rf=13.56e6,          # Hz
    n_cells=100
)

chamber.run_to_steady_state(t_max=4e-6)

n_plasma = chamber.measure_plasma_density()
Te = chamber.measure_electron_temperature()
P_absorbed = chamber.measure_absorbed_power()

print(f"Plasma density: {n_plasma:.2e} m^-3")  # Target: 1.65e17
print(f"Electron temp: {Te:.1f} eV")            # Target: 7.8 eV
print(f"Power absorbed: {P_absorbed:.1f} W")    # Target: 20 W
```

#### Phase 5: One-Way Coupling and Validation (1-2 weeks)

**Week 11: System Integration**
- [ ] DSMC intake → neutral density field
- [ ] Import neutral field into PIC as background
- [ ] Automated workflow for coupled simulation
- [ ] Parametric study scripts

**Week 12: Validation and Documentation**
- [ ] Reproduce all Parodi et al. key metrics
- [ ] Compare to AeriSat analytical model
- [ ] Generate validation report
- [ ] Write user documentation

**Deliverable:** Validated particle simulation capability with documentation

**Integration Example:**
```python
# Full AeriSat 3U system simulation
system = ABEPParticleSim(
    altitude=200e3,
    cubesat_size='3U',
    intake_geometry='parodi',
    thruster_power=20  # W
)

# Step 1: DSMC intake
intake_results = system.run_dsmc_intake(
    t_final=10e-3,
    output_dir='results/intake/'
)

# Step 2: Extract neutral conditions
n_neutral = intake_results.density_at_thruster()
T_neutral = intake_results.temperature_at_thruster()

# Step 3: PIC thruster
thruster_results = system.run_pic_thruster(
    n_neutral=n_neutral,
    T_neutral=T_neutral,
    t_final=4e-6,
    output_dir='results/thruster/'
)

# Step 4: Analysis
report = system.generate_report()
report.save('results/aerisat_3u_analysis.pdf')
```

### 2.4 Key Algorithms

#### 2.4.1 DSMC Binary Collision Algorithm

**Majorant Collision Frequency Method:**

```python
def perform_collisions(self, cell, dt):
    """
    Perform binary collisions in a cell using MCF method.
    
    Reference: Bird (1994), Section 2.6
    """
    particles = cell.particles
    V_cell = cell.volume
    N_particles = len(particles)
    
    # Majorant collision frequency (maximum possible)
    n_gas = N_particles / V_cell  # Number density
    sigma_T_max = self.get_max_cross_section(particles)
    g_max = self.get_max_relative_velocity(particles)
    nu_max = n_gas * sigma_T_max * g_max
    
    # Expected number of collisions
    N_coll_expected = 0.5 * N_particles * nu_max * dt
    N_coll_attempt = int(N_coll_expected + np.random.random())
    
    for _ in range(N_coll_attempt):
        # Randomly select collision partners
        i, j = np.random.choice(N_particles, 2, replace=False)
        p1, p2 = particles[i], particles[j]
        
        # Relative velocity and cross section
        g = np.linalg.norm(p1.v - p2.v)
        sigma_T = self.collision_cross_section(p1, p2, g)
        
        # Accept collision with probability
        P_accept = sigma_T * g / (sigma_T_max * g_max)
        
        if np.random.random() < P_accept:
            self.execute_collision(p1, p2)

def execute_collision(self, p1, p2):
    """
    Variable Hard Sphere post-collision velocities.
    
    Reference: Bird (1994), Section 2.4
    """
    # Center of mass velocity
    m1, m2 = p1.mass, p2.mass
    v_cm = (m1*p1.v + m2*p2.v) / (m1 + m2)
    
    # Relative velocity magnitude (conserved for elastic)
    g_pre = np.linalg.norm(p1.v - p2.v)
    
    # Isotropic scattering in COM frame
    theta = np.arccos(2*np.random.random() - 1)
    phi = 2 * np.pi * np.random.random()
    
    g_post = g_pre * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Transform back to lab frame
    mu = m1*m2 / (m1 + m2)  # Reduced mass
    p1.v = v_cm + (m2/(m1+m2)) * g_post
    p2.v = v_cm - (m1/(m1+m2)) * g_post
```

#### 2.4.2 PIC Particle Pusher (Boris Algorithm)

**Implicit Lorentz Force Integration:**

```python
def push_particles(self, dt):
    """
    Boris algorithm for particle motion with E and B fields.
    
    Reference: Birdsall & Langdon (2004), Section 4-3
    """
    for p in self.particles:
        # Half acceleration from electric field
        E = self.interpolate_field(p.x)
        v_minus = p.v + 0.5 * (p.q/p.m) * E * dt
        
        # Rotation from magnetic field
        B = self.get_magnetic_field(p.x)
        t = 0.5 * (p.q/p.m) * B * dt
        s = 2*t / (1 + np.dot(t, t))
        
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        
        # Half acceleration from electric field
        p.v = v_plus + 0.5 * (p.q/p.m) * E * dt
        
        # Update position
        p.x += p.v * dt
        
        # Handle boundaries
        self.apply_boundary_conditions(p)
```

#### 2.4.3 Monte Carlo Collision (Vahedi Method)

**Null-Collision Algorithm:**

```python
def monte_carlo_collisions(self, particles, neutrals, dt):
    """
    Null-collision method for electron-neutral collisions.
    
    Reference: Vahedi & Surendra, Comp. Phys. Comm. 87, 179 (1995)
    """
    # Maximum collision frequency across all energies
    nu_max = self.compute_nu_max(neutrals.density)
    
    for p in particles:
        # Probability of attempting collision
        P_coll = 1 - np.exp(-nu_max * dt)
        
        if np.random.random() < P_coll:
            # Compute collision energy
            E_coll = 0.5 * p.m * np.linalg.norm(p.v)**2 / e
            
            # Get cross sections at this energy
            sigma_elastic = self.xs_elastic(E_coll)
            sigma_ionization = self.xs_ionization(E_coll)
            sigma_total = sigma_elastic + sigma_ionization
            
            # Select process type
            R = np.random.random()
            
            if R < sigma_elastic / (sigma_total + 1e-30):
                # Elastic collision
                self.elastic_scatter(p, neutrals)
            
            elif R < (sigma_elastic + sigma_ionization) / (sigma_total + 1e-30):
                # Ionization
                if E_coll > self.E_ionization:
                    self.ionize(p, neutrals)
            
            # else: null collision (do nothing)

def ionize(self, electron, neutrals):
    """
    Electron impact ionization: e + N2 → 2e + N2+
    """
    E_coll = 0.5 * electron.m * np.linalg.norm(electron.v)**2 / e
    E_available = E_coll - self.E_ionization  # Energy after ionization
    
    # Create new ion
    ion = Particle(
        x=electron.x.copy(),
        v=neutrals.sample_thermal_velocity(),  # Born at rest (thermal)
        m=neutrals.mass,
        q=e,
        species='N2+'
    )
    self.particles.append(ion)
    
    # Energy partitioning between electrons (simplified)
    E_secondary = E_available * np.random.random()
    E_primary = E_available - E_secondary
    
    # Update primary electron
    v_mag_new = np.sqrt(2 * E_primary * e / electron.m)
    electron.v = v_mag_new * (electron.v / np.linalg.norm(electron.v))
    
    # Create secondary electron
    theta = np.arccos(2*np.random.random() - 1)
    phi = 2*np.pi*np.random.random()
    v_secondary = np.sqrt(2 * E_secondary * e / electron.m) * np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])
    
    secondary = Particle(
        x=electron.x.copy(),
        v=v_secondary,
        m=electron.m,
        q=-e,
        species='e'
    )
    self.particles.append(secondary)
```

### 2.5 Performance Optimization

**Target Performance:**
- DSMC: 10⁶ particles, 10 ms simulation time → ~30 min wall clock
- PIC: 10⁵ particles, 4 μs simulation time → ~1 hour wall clock
- Combined system: ~2 hours for full characterization

**Optimization Strategies:**

1. **Spatial Indexing:**
```python
# Use scipy.spatial.cKDTree for fast neighbor searches
from scipy.spatial import cKDTree

class CellIndexing:
    def __init__(self, domain, n_cells):
        self.tree = None
        self.cell_width = domain.length / n_cells
    
    def update(self, particles):
        positions = np.array([p.x for p in particles])
        self.tree = cKDTree(positions)
    
    def get_cell_particles(self, cell_index):
        x_min = cell_index * self.cell_width
        x_max = (cell_index + 1) * self.cell_width
        indices = self.tree.query_ball_point(
            [x_min + 0.5*self.cell_width],
            r=0.5*self.cell_width
        )
        return indices
```

2. **Vectorized Operations:**
```python
# Use NumPy broadcasting for particle operations
class ParticleArray:
    """Structure-of-arrays for vectorized operations."""
    def __init__(self, n_particles):
        self.x = np.zeros((n_particles, 3))
        self.v = np.zeros((n_particles, 3))
        self.m = np.zeros(n_particles)
        self.q = np.zeros(n_particles)
        self.active = np.ones(n_particles, dtype=bool)
    
    def push_all(self, E_field, dt):
        """Vectorized particle push."""
        accel = self.q[:, np.newaxis] * E_field / self.m[:, np.newaxis]
        self.v += accel * dt
        self.x += self.v * dt
```

3. **Adaptive Time Stepping:**
```python
def compute_timestep(self):
    """Compute stable timestep based on CFL condition."""
    # DSMC: dt < mean collision time
    dt_dsmc = 0.1 / self.collision_frequency_max()
    
    # PIC: dt < plasma period / 20
    omega_p = np.sqrt(self.n_max * e**2 / (m_e * eps0))
    dt_pic = 0.05 * 2*np.pi / omega_p
    
    return min(dt_dsmc, dt_pic)
```

4. **Parallel Processing:**
```python
# Use multiprocessing for parametric studies
from multiprocessing import Pool

def run_single_case(params):
    altitude, power = params
    sim = ABEPParticleSim(altitude=altitude, power=power)
    return sim.run()

if __name__ == '__main__':
    param_list = [(alt, pwr) for alt in altitudes for pwr in powers]
    
    with Pool(processes=8) as pool:
        results = pool.map(run_single_case, param_list)
```

### 2.6 Testing and Validation Strategy

**Unit Tests:**
```python
# tests/test_dsmc.py
import pytest
from aerisat_psim.dsmc import DSMC1D

def test_free_molecular_flow():
    """Validate ballistic motion with no collisions."""
    sim = DSMC1D(length=1.0, n_cells=10)
    sim.collision_rate = 0  # No collisions
    
    # Inject particles with v_x = 100 m/s
    sim.inject_particles(n=1000, v_x=100, v_y=0, v_z=0)
    
    # Run for 0.01 s (particles should travel 1 m)
    sim.run(t_final=0.01)
    
    x_final = [p.x[0] for p in sim.particles]
    assert np.mean(x_final) == pytest.approx(1.0, rel=0.01)

def test_thermal_equilibrium():
    """Verify approach to Maxwell-Boltzmann distribution."""
    sim = DSMC1D(length=0.1, n_cells=20)
    sim.initialize_particles(n=1e19, T=300, species='N2')
    
    # Run until equilibrium
    sim.run(t_final=1e-3)
    
    v_dist = sim.compute_velocity_distribution()
    T_measured = sim.compute_temperature()
    
    assert T_measured == pytest.approx(300, rel=0.05)

def test_compression_ratio():
    """Validate intake compression against analytical limit."""
    intake = DSMCIntake(
        length=0.2,
        d_inlet=0.06,
        d_outlet=0.03,
        sigma_t=1.0  # Diffuse reflection
    )
    
    atm = Atmosphere(altitude=200e3)
    intake.inject_freestream(atm, velocity=7800)
    intake.run(t_final=10e-3)
    
    CR = intake.compression_ratio()
    CR_analytical = intake.compute_analytical_CR()
    
    assert CR == pytest.approx(CR_analytical, rel=0.20)
```

**Integration Tests:**
```python
# tests/test_integration.py
def test_dsmc_pic_coupling():
    """Verify one-way coupling preserves mass flux."""
    # DSMC stage
    intake = DSMCIntake(...)
    intake.run()
    
    n_thruster = intake.get_outlet_density()
    mdot_dsmc = intake.get_mass_flux()
    
    # PIC stage
    thruster = PICThruster(n_neutral=n_thruster)
    thruster.run()
    
    mdot_pic = thruster.get_ion_flux() * thruster.mass_utilization
    
    # Mass conservation check
    assert mdot_pic / mdot_dsmc == pytest.approx(
        thruster.mass_utilization,
        rel=0.10
    )
```

**Validation Against Literature:**
```python
# validation/parodi_comparison.py
def test_parodi_intake_CR():
    """Reproduce Parodi et al. compression ratio."""
    sim = setup_parodi_intake()
    sim.run()
    
    CR_N2 = sim.compression_ratio('N2')
    
    # Parodi reported CR = 475 for N2
    assert 400 <= CR_N2 <= 550  # Allow for stochastic variation

def test_parodi_plasma_density():
    """Reproduce Parodi et al. plasma density."""
    sim = setup_parodi_thruster()
    sim.run()
    
    n_plasma = sim.peak_plasma_density()
    
    # Parodi reported n = 1.65e17 m^-3
    assert n_plasma == pytest.approx(1.65e17, rel=0.20)
```

### 2.7 Deliverables and Documentation

**Code Deliverables:**
1. `aerisat_psim` Python package with DSMC and PIC modules
2. Example scripts for intake and thruster simulations
3. Validation suite against Parodi et al. and analytical models
4. Parametric study tools for design optimization

**Documentation Deliverables:**
1. **Theory Manual** (`docs/theory.md`):
   - Mathematical foundations of DSMC and PIC
   - Collision models and cross sections
   - Boundary condition implementations
   
2. **User Guide** (`docs/user_guide.md`):
   - Installation instructions
   - Tutorial examples
   - API reference
   
3. **Validation Report** (`docs/validation.md`):
   - Comparison to analytical models
   - Parodi et al. reproduction results
   - Uncertainty quantification

4. **Design Study Reports**:
   - Intake geometry optimization
   - Thruster power scaling analysis
   - Altitude performance envelope

---

## 3. Option 3: Full 3D Particle Simulation - Stretch Goal

### 3.1 Motivation and Scope

**Why Full 3D?**
- Capture true geometry effects (elliptical intake apertures, grid holes)
- Model electromagnetic field asymmetries in thruster
- Predict plume divergence and spacecraft interactions
- Enable CFD-level visualization for presentations and papers

**Additional Capabilities vs. Option 2:**
- Unstructured tetrahedral mesh generation
- 3D Poisson solver with complex boundary conditions
- Fully-coupled DSMC-PIC with bidirectional mass transfer
- Far-field plume expansion
- Parallelization with MPI for HPC clusters

**Computational Requirements:**
- 10⁷-10⁸ particles for adequate statistics
- 10⁵-10⁶ mesh elements
- 100-1000 CPU cores for reasonable turnaround
- ~1-10 TB storage for time-series data

**Timeline:** 6-12 months with dedicated team or HPC access

### 3.2 Architecture Extensions

**Additional Components:**
```
aerisat-particle-sim/
├── src/
│   ├── aerisat_psim/
│   │   ├── mesh3d/
│   │   │   ├── __init__.py
│   │   │   ├── tetrahedral.py     # Unstructured tet mesh
│   │   │   ├── generator.py       # Interface to Gmsh/TetGen
│   │   │   ├── adaptation.py      # AMR for high gradients
│   │   │   └── partition.py       # Mesh partitioning for MPI
│   │   ├── dsmc3d/
│   │   │   ├── __init__.py
│   │   │   ├── mover.py           # 3D particle trajectories
│   │   │   ├── collisions.py      # Cell-based collision detection
│   │   │   ├── surfaces.py        # Triangle-particle intersections
│   │   │   └── sampling.py        # Volume-weighted sampling
│   │   ├── pic3d/
│   │   │   ├── __init__.py
│   │   │   ├── field_solver_fem.py # 3D Poisson with FEM
│   │   │   ├── field_solver_fft.py # FFT solver for Cartesian
│   │   │   ├── mover_3d.py        # 3D Boris pusher
│   │   │   ├── weight_3d.py       # Trilinear/TSC interpolation
│   │   │   └── current_deposit.py # J-field for EM PIC
│   │   ├── coupled/
│   │   │   ├── __init__.py
│   │   │   ├── interface.py       # DSMC-PIC boundary
│   │   │   ├── load_balance.py    # Dynamic particle redistribution
│   │   │   └── checkpoint.py      # Save/restart capability
│   │   ├── parallel/
│   │   │   ├── __init__.py
│   │   │   ├── mpi_wrapper.py     # MPI communication
│   │   │   ├── domain_decomp.py   # Spatial decomposition
│   │   │   └── particle_exchange.py # Inter-rank particle transfer
│   │   └── visualization/
│   │       ├── __init__.py
│   │       ├── vtk_export.py      # VTK/VTU file output
│   │       ├── visit_interface.py # VisIt/ParaView integration
│   │       └── matplotlib_3d.py   # Lightweight plotting
```

### 3.3 Key Technical Challenges

#### Challenge 1: Unstructured Mesh Generation

**Solution: Interface with Gmsh**
```python
import gmsh

def generate_abep_mesh(intake_length=0.18, chamber_diameter=0.03):
    """
    Generate unstructured tet mesh for full ABEP geometry.
    """
    gmsh.initialize()
    gmsh.model.add("abep_system")
    
    # Define geometry using constructive solid geometry
    # (Intake duct, ionization chamber, grids, plume region)
    
    # Intake duct (cone)
    intake_cone = gmsh.model.occ.addCone(
        x=0, y=0, z=0,
        dx=0, dy=0, dz=intake_length,
        r1=0.03, r2=0.015,  # Taper from 60mm to 30mm
        tag=1
    )
    
    # Ionization chamber (cylinder)
    chamber = gmsh.model.occ.addCylinder(
        x=0, y=0, z=intake_length,
        dx=0, dy=0, dz=0.06,
        r=chamber_diameter/2,
        tag=2
    )
    
    # Grids (thin annular disks with holes)
    screen_grid = create_grid_with_holes(
        position=intake_length + 0.05,
        thickness=0.0005,
        hole_diameter=0.0019,
        n_holes=61
    )
    
    # Boolean union
    full_geometry = gmsh.model.occ.fuse(
        [(3, intake_cone)],
        [(3, chamber), (3, screen_grid), ...]
    )
    
    gmsh.model.occ.synchronize()
    
    # Mesh size field (fine near grids, coarse in plume)
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", 0.0005)  # 0.5 mm near grids
    gmsh.model.mesh.field.setNumber(1, "VOut", 0.01)   # 10 mm far field
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    gmsh.write("abep_mesh.msh")
    gmsh.finalize()
    
    return load_gmsh_mesh("abep_mesh.msh")
```

#### Challenge 2: 3D Poisson Solver

**Option A: Finite Element Method**
```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class PoissonSolver3D_FEM:
    """
    3D Poisson solver using linear tetrahedral elements.
    
    ∇²φ = -ρ/ε₀
    
    Weak form: ∫ ∇φ·∇ψ dV = ∫ (ρ/ε₀) ψ dV
    """
    def __init__(self, mesh):
        self.mesh = mesh
        self.assemble_stiffness_matrix()
    
    def assemble_stiffness_matrix(self):
        """
        Build global stiffness matrix from element contributions.
        """
        n_nodes = len(self.mesh.nodes)
        K = np.zeros((n_nodes, n_nodes))
        
        for elem in self.mesh.elements:
            # Compute element stiffness matrix
            K_e = self.element_stiffness(elem)
            
            # Assemble into global matrix
            for i, node_i in enumerate(elem.nodes):
                for j, node_j in enumerate(elem.nodes):
                    K[node_i, node_j] += K_e[i, j]
        
        # Apply boundary conditions (Dirichlet)
        for node in self.mesh.boundary_nodes:
            K[node, :] = 0
            K[node, node] = 1
        
        self.K = csr_matrix(K)  # Sparse format
    
    def element_stiffness(self, elem):
        """
        Local stiffness matrix for tetrahedral element.
        
        K_ij = ∫ ∇N_i · ∇N_j dV
        """
        # Compute gradients of shape functions
        grad_N = self.shape_function_gradients(elem)
        V = elem.volume
        
        K_e = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                K_e[i,j] = V * np.dot(grad_N[i], grad_N[j])
        
        return K_e
    
    def solve(self, charge_density):
        """
        Solve ∇²φ = -ρ/ε₀ for potential φ.
        """
        # Project charge density to nodes
        rhs = self.project_to_nodes(charge_density) / eps0
        
        # Apply boundary conditions to RHS
        for node in self.mesh.boundary_nodes:
            rhs[node] = self.boundary_values[node]
        
        # Solve linear system
        phi = spsolve(self.K, -rhs)
        
        # Compute electric field E = -∇φ
        E = self.compute_gradient(phi)
        
        return phi, E
```

**Option B: Spectral Methods (Cartesian meshes only)**
```python
import numpy.fft as fft

class PoissonSolver3D_FFT:
    """
    Fast Poisson solver using FFT for Cartesian grids.
    
    Much faster than FEM, but requires regular grid.
    """
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        
        # Wavenumbers
        kx = 2*np.pi*fft.fftfreq(Nx, Lx/Nx)
        ky = 2*np.pi*fft.fftfreq(Ny, Ly/Ny)
        kz = 2*np.pi*fft.fftfreq(Nz, Lz/Nz)
        
        self.kx, self.ky, self.kz = np.meshgrid(kx, ky, kz, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0,0,0] = 1  # Avoid division by zero
    
    def solve(self, rho):
        """
        Solve ∇²φ = -ρ/ε₀ using FFT.
        
        Time complexity: O(N log N) vs. O(N^1.5) for sparse direct solve
        """
        # Transform to Fourier space
        rho_k = fft.fftn(rho)
        
        # Solve: -k² φ_k = -ρ_k/ε₀
        phi_k = rho_k / (eps0 * self.k2)
        phi_k[0,0,0] = 0  # Set DC component
        
        # Transform back to real space
        phi = np.real(fft.ifftn(phi_k))
        
        # Compute electric field
        Ex = np.real(fft.ifftn(-1j * self.kx * phi_k))
        Ey = np.real(fft.ifftn(-1j * self.ky * phi_k))
        Ez = np.real(fft.ifftn(-1j * self.kz * phi_k))
        
        return phi, (Ex, Ey, Ez)
```

#### Challenge 3: MPI Parallelization

**Domain Decomposition Strategy:**
```python
from mpi4py import MPI

class MPISimulation:
    """
    MPI-parallel particle simulation with domain decomposition.
    """
    def __init__(self, mesh):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Partition mesh across ranks
        self.local_mesh = self.partition_mesh(mesh)
        
        # Identify ghost cells (overlap regions)
        self.ghost_cells = self.identify_ghost_cells()
    
    def partition_mesh(self, mesh):
        """
        Distribute mesh elements across MPI ranks using METIS.
        """
        from pymetis import part_graph
        
        # Build adjacency graph of mesh elements
        adjacency = mesh.build_dual_graph()
        
        # Partition for load balancing
        n_parts = self.size
        membership = part_graph(n_parts, adjacency=adjacency)
        
        # Extract local partition
        local_elements = [e for e, p in zip(mesh.elements, membership) 
                         if p == self.rank]
        
        return Mesh(local_elements)
    
    def exchange_particles(self):
        """
        Transfer particles that crossed domain boundaries.
        """
        particles_to_send = {rank: [] for rank in range(self.size)}
        
        # Identify particles that left local domain
        for p in self.particles:
            if not self.local_mesh.contains(p.x):
                target_rank = self.find_owner_rank(p.x)
                if target_rank != self.rank:
                    particles_to_send[target_rank].append(p)
        
        # All-to-all exchange using MPI
        particles_received = self.comm.alltoall(particles_to_send)
        
        # Remove departed particles, add arrived particles
        self.particles = [p for p in self.particles 
                         if self.local_mesh.contains(p.x)]
        for rank_particles in particles_received:
            self.particles.extend(rank_particles)
    
    def exchange_field_data(self, field):
        """
        Exchange field values at ghost cell boundaries.
        """
        for neighbor_rank in self.get_neighbor_ranks():
            # Send ghost cell data
            ghost_data = self.extract_ghost_data(field, neighbor_rank)
            self.comm.send(ghost_data, dest=neighbor_rank, tag=0)
            
            # Receive boundary data
            boundary_data = self.comm.recv(source=neighbor_rank, tag=0)
            self.inject_boundary_data(field, boundary_data, neighbor_rank)
```

#### Challenge 4: Visualization and Post-Processing

**VTK Export for ParaView:**
```python
import meshio

def export_vtk(filename, mesh, particles, fields):
    """
    Export particle and field data in VTK format.
    """
    # Cell data (field quantities)
    cell_data = {
        'density': fields['density'],
        'velocity': fields['velocity'],
        'temperature': fields['temperature']
    }
    
    # Point data (mesh nodes)
    point_data = {
        'potential': fields['potential'],
        'electric_field': fields['electric_field']
    }
    
    # Write mesh with fields
    meshio.write_points_cells(
        filename + '_mesh.vtu',
        mesh.nodes,
        {'tetra': mesh.elements},
        point_data=point_data,
        cell_data=cell_data
    )
    
    # Particle data as point cloud
    particle_positions = np.array([p.x for p in particles])
    particle_velocities = np.array([p.v for p in particles])
    particle_species = np.array([p.species_id for p in particles])
    
    meshio.write_points_cells(
        filename + '_particles.vtu',
        particle_positions,
        {},  # No cells, just points
        point_data={
            'velocity': particle_velocities,
            'species': particle_species
        }
    )

# Usage in simulation loop
for step in range(n_steps):
    sim.advance(dt)
    
    if step % output_interval == 0:
        export_vtk(f'output/step_{step:06d}', 
                   sim.mesh, sim.particles, sim.fields)
```

**Interactive Visualization with VisIt:**
```python
# Create .visit file for time series
def create_visit_file(output_dir, n_steps):
    with open(f'{output_dir}/timeseries.visit', 'w') as f:
        f.write('!NBLOCKS 1\n')
        for step in range(0, n_steps, output_interval):
            f.write(f'step_{step:06d}_mesh.vtu\n')
```

### 3.4 Implementation Roadmap

**Phase 1: Mesh Infrastructure (1-2 months)**
- [ ] Implement unstructured tetrahedral mesh data structure
- [ ] Interface with Gmsh for geometry import
- [ ] Develop spatial search structures (octree, kd-tree)
- [ ] Validate with simple test geometries

**Phase 2: 3D DSMC (2-3 months)**
- [ ] Extend DSMC to 3D unstructured mesh
- [ ] Implement triangle-particle collision detection
- [ ] Add 3D gas-surface interaction models
- [ ] Validate against analytical 3D flows

**Phase 3: 3D PIC (2-3 months)**
- [ ] Implement 3D FEM Poisson solver
- [ ] Develop 3D particle mover with complex boundaries
- [ ] Add 3D weighting and deposition algorithms
- [ ] Validate against 3D plasma benchmarks

**Phase 4: MPI Parallelization (1-2 months)**
- [ ] Implement domain decomposition
- [ ] Add particle and field exchange
- [ ] Optimize load balancing
- [ ] Scale testing up to 1000 cores

**Phase 5: Full System Coupling (2-3 months)**
- [ ] One-way DSMC→PIC coupling
- [ ] Two-way coupling with plasma-neutral interactions
- [ ] Grid geometry with ion extraction
- [ ] Far-field plume expansion

**Phase 6: Validation and Applications (1-2 months)**
- [ ] Reproduce Parodi et al. 3D results
- [ ] Compare to AeriSat analytical and 1D models
- [ ] Run parametric studies on HPC cluster
- [ ] Generate publication-quality visualizations

### 3.5 Computational Resources

**Hardware Requirements:**

| Simulation Type | CPU Cores | Memory | Storage | Runtime |
|----------------|-----------|--------|---------|---------|
| 1D DSMC | 1-4 | 4 GB | 10 GB | 30 min |
| 1D PIC | 1-4 | 8 GB | 20 GB | 1 hour |
| 3D DSMC | 100-500 | 200 GB | 1 TB | 12 hours |
| 3D PIC | 100-500 | 400 GB | 2 TB | 24 hours |
| 3D Coupled | 500-2000 | 1 TB | 10 TB | 2-7 days |

**Recommended HPC Access:**
- **XSEDE/ACCESS:** NSF-funded supercomputing allocations
- **NASA HECC:** High-End Computing Capability (if SBIR Phase II)
- **DOE INCITE:** Leadership computing (very competitive)
- **AWS/Azure HPC:** Commercial cloud (expensive but flexible)

**Software Stack:**
```bash
# Core dependencies
gcc/11.0               # C++ compiler with OpenMP
python/3.11            # Python interpreter
openmpi/4.1.4          # MPI library
petsc/3.18             # Sparse linear algebra
gmsh/4.11              # Mesh generation
hdf5/1.14              # Parallel I/O

# Python packages
numpy, scipy, numba    # Numerical computing
mpi4py                 # Python MPI bindings
meshio, h5py           # Mesh and data I/O
matplotlib, mayavi     # Visualization
pytest, pytest-mpi     # Testing
```

### 3.6 Alternative Approach: Leverage Existing Codes

**Rather than implementing from scratch, consider:**

#### Option 3A: SPARTA + Custom PIC Module
**SPARTA (Stochastic PArallel Rarefied-gas Time-accurate Analyzer)**
- Open-source DSMC code from Sandia National Labs
- Proven 3D unstructured mesh capability
- MPI-parallel with excellent scaling
- Extensive validation and documentation

```bash
# Download and compile SPARTA
git clone https://github.com/sparta/sparta
cd sparta/src
make yes-kokkos  # Optional GPU acceleration
make mpi

# Custom PIC module integrated as SPARTA "fix"
# Requires C++ development but leverages SPARTA infrastructure
```

**Integration Strategy:**
1. Use SPARTA for DSMC intake simulation
2. Extract neutral density field at thruster inlet
3. Feed into custom PIC code (can reuse Option 2 code extended to 3D)
4. Couple back ion currents if needed

**Advantages:**
- Production-quality DSMC immediately available
- Focus development effort on PIC module
- Proven parallel performance
- Good documentation and user community

#### Option 3B: Full Commercial Solution
**ESI Group SPIS-EP (Spacecraft Plasma Interaction System - Electric Propulsion)**
- Commercial PIC code for EP simulation
- Includes DSMC capabilities
- Validated against numerous flight missions
- ~$50k-100k/year license

**Use Case:** Design validation and investor demos, not R&D tool

### 3.7 Expected Outputs

**Publications:**
- Conference paper (IEPC, AIAA SciTech) on simulation methodology
- Journal paper (Journal of Electric Propulsion) with validation results
- Technical reports for SBIR deliverables

**Technical Artifacts:**
- Open-source Python/C++ codebase on GitHub
- Docker containers with full software stack
- Archived datasets for reproducibility
- Interactive web visualizations (using ParaView Glance)

**Business Value:**
- High-fidelity performance predictions for investor decks
- Design optimization database (500+ runs)
- Risk reduction for flight demonstration
- IP generation (novel algorithms, unique insights)

---

## 4. Integration with Existing AeriSat Models

### 4.1 Workflow Integration

**Current Analytical Model → Particle Simulations:**

```python
# Start with analytical model for initial sizing
analytical = ABEPPropulsionSystem(
    altitude=200e3,
    cubesat_size='3U',
    power_budget=60  # W
)
results_analytical = analytical.run_analysis()

# Extract parameters for particle sim
intake_params = {
    'length': 0.18,
    'diameter_inlet': 0.06,
    'diameter_outlet': 0.03,
    'wall_temp': 700,
    'accommodation': 0.9
}

thruster_params = {
    'chamber_length': 0.06,
    'chamber_diameter': 0.03,
    'n_neutral_target': results_analytical['mass_flow']['n_thruster'],
    'power_rf': 20,
    'frequency_rf': 13.56e6
}

# Run particle simulations for validation
particle_system = ABEPParticleSim()
particle_system.setup_intake(**intake_params)
particle_system.setup_thruster(**thruster_params)

results_particle = particle_system.run(
    dsmc_time=10e-3,
    pic_time=4e-6
)

# Compare results
comparison = compare_models(results_analytical, results_particle)
print(comparison.summary())

# Update analytical model calibration
if comparison.discrepancy('compression_ratio') > 0.1:
    analytical.nozzle.clausing_factor *= comparison.scaling_factor
    results_updated = analytical.run_analysis()
```

### 4.2 Calibration Database

**Use particle simulations to build surrogate models:**

```python
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

# Run parametric particle simulations
param_space = {
    'altitude': np.linspace(180e3, 250e3, 10),
    'intake_length': np.linspace(0.10, 0.25, 5),
    'accommodation': np.linspace(0.8, 1.0, 4),
    'rf_power': np.linspace(15, 30, 5)
}

# Generate training data (200 cases × 2 hours = ~400 hours = 17 days on 1 core)
# Parallelize across 20 cores → ~20 hours
results_database = []
for case in generate_cases(param_space):
    result = run_particle_sim(case)
    results_database.append(result)

df = pd.DataFrame(results_database)
df.to_csv('particle_sim_database.csv')

# Train Gaussian Process surrogate
features = ['altitude', 'intake_length', 'accommodation', 'rf_power']
targets = ['compression_ratio', 'plasma_density', 'thrust']

surrogates = {}
for target in targets:
    gp = GaussianProcessRegressor()
    gp.fit(df[features], df[target])
    surrogates[target] = gp

# Use surrogate in analytical model for fast prediction
def predict_fast(altitude, intake_length, accommodation, rf_power):
    X = [[altitude, intake_length, accommodation, rf_power]]
    predictions = {
        target: surrogates[target].predict(X)[0]
        for target in targets
    }
    return predictions

# Now analytical model runs use surrogate-corrected physics
analytical.correction_model = predict_fast
```

### 4.3 Uncertainty Quantification

**Propagate uncertainties from analytical model through particle sims:**

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define uncertainty distributions
problem = {
    'num_vars': 5,
    'names': ['n_atm', 'T_atm', 'accommodation', 'rf_power', 'neutral_density'],
    'bounds': [
        [3.5e17, 5.0e17],  # Atmospheric density ±20%
        [800, 1200],       # Temperature ±20%
        [0.85, 0.95],      # Accommodation coefficient
        [18, 22],          # RF power ±10%
        [0.9e20, 1.1e20]   # Neutral density in thruster ±10%
    ]
}

# Generate samples
param_samples = saltelli.sample(problem, 1024)

# Run particle sims (or use surrogates)
thrust_outputs = []
for params in param_samples:
    result = run_particle_sim_with_params(params)
    thrust_outputs.append(result['thrust'])

# Sensitivity analysis
Si = sobol.analyze(problem, np.array(thrust_outputs))
print("First-order sensitivity indices:")
print(Si['S1'])  # Shows which parameters matter most

# Result: Know which uncertainties to focus on reducing
```

---

## 5. Success Metrics and Risk Mitigation

### 5.1 Success Criteria (Option 2)

**Technical Metrics:**
- [ ] DSMC reproduces free molecular flow limit (Kn → ∞)
- [ ] DSMC matches continuum viscosity within 5% (Kn → 0)
- [ ] Intake compression ratio within 20% of Parodi (CR ~ 400-500)
- [ ] Plasma density within 30% of global model (n ~ 1.5e17 m⁻³)
- [ ] Electron temperature within 20% of Parodi (Te ~ 7-9 eV)
- [ ] Ion energy distribution shows bi-modal structure
- [ ] Code executes parametric sweep (10 cases) in < 24 hours

**Programmatic Metrics:**
- [ ] 80% test coverage with pytest
- [ ] Documentation complete (theory, user guide, validation)
- [ ] At least one peer-reviewed publication submitted
- [ ] Code publicly released on GitHub (if not ITAR-restricted)
- [ ] Integrated into AeriSat SBIR proposal workflow

### 5.2 Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| DSMC collision algorithm bugs | Medium | High | Extensive unit tests, analytical validation cases |
| PIC numerical instability | High | High | Implicit time integration, adaptive timestep |
| Poor agreement with Parodi | Medium | Medium | Start with exact reproduction of their case |
| Insufficient computational resources | Low | Medium | Optimize code, use cloud HPC if needed |
| Cross-section data inaccuracies | Medium | Medium | Use multiple databases, sensitivity studies |
| Team bandwidth limitations | High | High | Prioritize core capabilities, defer nice-to-haves |
| ITAR classification issues | Low | High | Consult legal early, sanitize published code |

### 5.3 Go/No-Go Decision Points

**Week 4 (End of Phase 1):**
- **Criteria:** DSMC passes free molecular and continuum tests
- **Decision:** Proceed to intake application
- **Fallback:** Debug collision algorithm, extend timeline by 1 week

**Week 8 (End of Phase 3):**
- **Criteria:** PIC plasma density matches global model within 50%
- **Decision:** Proceed to full validation
- **Fallback:** Simplify PIC model, focus on DSMC only

**Week 12 (End of Phase 5):**
- **Criteria:** Published validation report, code on GitHub
- **Decision:** Declare Option 2 success, consider Option 3
- **Fallback:** Conclude development, focus on applications

---

## 6. Resource Requirements

### 6.1 Personnel

**Option 2 (8-12 weeks):**
- **Primary Developer:** 0.5 FTE (George or contractor)
  - Responsible for algorithm implementation
  - Code reviews and testing
  - Documentation writing
  
- **Advisor/Consultant:** 0.1 FTE (academic expert)
  - Weekly technical review meetings
  - Validation strategy guidance
  - Publication co-authorship

**Option 3 (6-12 months):**
- **Lead Developer:** 1.0 FTE (senior engineer/postdoc)
- **Supporting Developer:** 0.5 FTE (graduate student)
- **HPC Specialist:** 0.2 FTE (for parallelization)
- **Principal Investigator:** 0.2 FTE (George - technical oversight)

### 6.2 Computing

**Option 2:**
- **Laptop/Workstation:** Sufficient
  - Modern CPU (8+ cores)
  - 16-32 GB RAM
  - 1 TB SSD storage
  - **Cost:** $2k-3k (likely already have)

**Option 3:**
- **HPC Allocation:** Required
  - XSEDE startup allocation (free, ~50k core-hours)
  - AWS HPC instances (~$2-5/core-hour)
  - **Estimated cost:** $0 (XSEDE) to $20k (AWS)

### 6.3 Software

**All Open-Source (Free):**
- Python, NumPy, SciPy, Matplotlib
- Gmsh (mesh generation)
- ParaView (visualization)
- Git/GitHub (version control)
- Pytest (testing)
- Sphinx (documentation)

**Optional Commercial:**
- MATLAB (~$500/year academic) - for rapid prototyping
- TechPlot 360 (~$2k) - high-quality visualization
- Intel VTune (~$700) - performance profiling

### 6.4 Budget Estimate

**Option 2 Minimal Budget:**
- Personnel: $20k-40k (contractor/intern for 3 months)
- Compute: $0 (use existing hardware)
- Software: $0 (all open-source)
- Conference travel: $2k (present results)
- **Total: $22k-42k**

**Option 2 with Support:**
- Personnel: $50k-80k (senior developer)
- Consultant: $10k (academic advisor)
- Cloud compute: $5k (for parametric studies)
- Publication fees: $3k (open access)
- **Total: $68k-98k**

**Option 3 Full Implementation:**
- Personnel: $150k-250k (team of 2-3 for 12 months)
- HPC allocation: $20k-50k (if commercial)
- Travel/conferences: $10k (multiple presentations)
- Publication fees: $5k
- **Total: $185k-315k**

### 6.5 Timeline Summary

```
Option 2 Implementation Timeline (12 weeks)
═══════════════════════════════════════════════

Weeks 1-3: DSMC Core
├─ Week 1: Particles & Mesh
├─ Week 2: Collisions
└─ Week 3: Surfaces & Validation

Weeks 4-5: DSMC Intake
├─ Week 4: Geometry & BC
└─ Week 5: Parametric Studies

Weeks 6-8: PIC Core
├─ Week 6: Fields & Motion
├─ Week 7: MCC
└─ Week 8: RF Discharge

Weeks 9-10: PIC Thruster
├─ Week 9-10: Chamber Modeling

Weeks 11-12: Integration
├─ Week 11: Coupling & Validation
└─ Week 12: Documentation & Release

Deliverable: Working particle simulation capability
═══════════════════════════════════════════════

Option 3 Extension Timeline (24 weeks additional)
═══════════════════════════════════════════════

Weeks 13-16: 3D Mesh Infrastructure
Weeks 17-22: 3D DSMC Implementation  
Weeks 23-28: 3D PIC Implementation
Weeks 29-32: MPI Parallelization
Weeks 33-36: Full System Coupling & Applications

Deliverable: Production 3D simulation code
═══════════════════════════════════════════════
```

---

## 7. Documentation and Training

### 7.1 Documentation Structure

**docs/theory.md** (50 pages)
1. Introduction to particle methods
2. DSMC mathematical foundations
3. PIC-MCC mathematical foundations
4. Collision models and cross sections
5. Boundary conditions
6. Numerical stability and accuracy

**docs/user_guide.md** (30 pages)
1. Installation and setup
2. Quick start tutorial
3. Example simulations
4. Input file format
5. Output file format
6. Visualization workflow
7. Troubleshooting

**docs/validation.md** (40 pages)
1. Test case descriptions
2. Results and comparisons
3. Uncertainty quantification
4. Lessons learned
5. Recommended practices

**docs/api_reference.md** (auto-generated)
- Complete API documentation from docstrings
- Generated with Sphinx

### 7.2 Example Scripts

**examples/01_dsmc_intake.py:**
```python
"""
DSMC simulation of AeriSat intake at 200 km altitude.

This example demonstrates:
- Freestream injection with orbital velocity
- Gas-surface interactions (CLL model)
- Compression ratio calculation
- Visualization of density profile

Runtime: ~30 minutes on 4 cores
"""

from aerisat_psim import DSMC1D, Atmosphere, Intake

# Setup atmosphere
atm = Atmosphere(altitude=200e3, species=['N2', 'O'])
print(f"Freestream density: {atm.density:.2e} kg/m³")

# Define intake geometry
intake = Intake(
    length=0.18,
    diameter_inlet=0.06,
    diameter_outlet=0.03,
    wall_temperature=700,
    accommodation_normal=1.0,
    accommodation_tangential=0.9
)

# Create DSMC simulation
sim = DSMC1D(
    geometry=intake,
    atmosphere=atm,
    n_particles=1_000_000,
    n_cells=200,
    timestep='auto'
)

# Run simulation
print("Running DSMC...")
sim.run(t_final=10e-3, output_interval=1e-4)

# Post-process
CR_N2 = sim.compression_ratio('N2')
CR_O = sim.compression_ratio('O')
print(f"\nResults:")
print(f"  N2 compression ratio: {CR_N2:.1f}")
print(f"  O compression ratio: {CR_O:.1f}")

# Visualize
sim.plot_density_profile(filename='intake_density.png')
sim.plot_temperature_profile(filename='intake_temperature.png')
sim.export_vtk('intake_result.vtu')

print("\nSimulation complete!")
```

**examples/02_pic_discharge.py:**
```python
"""
PIC simulation of RF discharge in ionization chamber.

This example demonstrates:
- RF power coupling
- Plasma generation and heating
- Electron temperature oscillation
- Ion extraction current

Runtime: ~1 hour on 4 cores
"""

from aerisat_psim import PIC1D, RFDischarge

# Setup chamber
chamber = RFDischarge(
    length=0.06,
    diameter=0.03,
    n_neutral=1e20,  # m^-3 (from DSMC)
    T_neutral=700,   # K
    P_rf_target=20,  # W
    f_rf=13.56e6,    # Hz
    n_cells=100
)

# Initial plasma seed
chamber.initialize_plasma(
    n_plasma=1e17,  # m^-3
    Te=5            # eV
)

# Run simulation
print("Running PIC...")
chamber.run(
    t_final=4e-6,
    output_interval=10  # RF cycles
)

# Analyze results
n_avg = chamber.average_density()
Te_avg = chamber.average_electron_temperature()
P_abs = chamber.absorbed_power()

print(f"\nResults:")
print(f"  Plasma density: {n_avg:.2e} m^-3")
print(f"  Electron temperature: {Te_avg:.2f} eV")
print(f"  Absorbed power: {P_abs:.1f} W")

# Visualize
chamber.plot_time_history(filename='discharge_history.png')
chamber.plot_eedf(filename='electron_distribution.png')
chamber.export_vtk('discharge_result.vtu')

print("\nSimulation complete!")
```

### 7.3 Training Materials

**Video Tutorials (15-20 min each):**
1. Installation and environment setup
2. Running your first DSMC simulation
3. Understanding DSMC collision statistics
4. PIC simulation walkthrough
5. Visualizing results in ParaView
6. Debugging common issues

**Jupyter Notebook Tutorials:**
- Interactive exploration of collision models
- Sensitivity studies with parameter sweeps
- Comparison to analytical models
- Uncertainty quantification workflow

---

## 8. Publication and Dissemination Strategy

### 8.1 Conference Presentations

**Target Conferences:**

1. **IEPC (International Electric Propulsion Conference)** - 2026
   - Abstract: "Particle-Based Simulation of Air-Breathing Electric Propulsion for CubeSats"
   - Audience: EP researchers and engineers
   - Timeline: Abstract deadline typically 6 months before conference

2. **AIAA SciTech Forum** - January 2026
   - Session: Electric Propulsion
   - Paper: "Validation of DSMC-PIC Coupling for ABEP System Analysis"
   - Audience: Broader aerospace community

3. **CubeSat Developers Workshop** - 2026
   - Talk: "Enabling Long-Duration VLEO Missions with ABEP"
   - Audience: Small satellite developers
   - Format: Invited presentation

### 8.2 Journal Publications

**Target Journals:**

1. **Journal of Electric Propulsion** (open access)
   - Paper: "Particle-Based Modeling and Validation of Air-Breathing Electric Propulsion Systems"
   - Content: Full methodology, validation against Parodi, parametric studies
   - Timeline: Submit Q2 2026, publish Q4 2026

2. **Computer Physics Communications** (optional)
   - Paper: "AeriSat-PSim: An Open-Source Particle Simulation Framework for ABEP"
   - Content: Software description, benchmarks, tutorial
   - Timeline: Submit Q3 2026 if code is publicly released

### 8.3 Technical Reports

**SBIR Deliverables:**
- Quarterly progress reports with simulation results
- Final report: "Validated Simulation Capability for AeriSat ABEP System"
- Case studies: Performance predictions for various altitudes and configurations

**White Papers:**
- "Particle Simulation Best Practices for Electric Propulsion Design"
- "Uncertainty Quantification in ABEP Performance Prediction"

### 8.4 Open-Source Release

**If ITAR-cleared:**
- GitHub repository: `AeriSat/aerisat-particle-sim`
- MIT or GPL-3.0 license
- Zenodo DOI for citation
- Announce on relevant mailing lists (IEPC, SMC)

**Documentation Website:**
- Hosted on GitHub Pages or Read the Docs
- Includes: theory, tutorials, API reference
- Examples gallery with visualizations

---

## 9. Conclusion and Next Steps

### 9.1 Recommendation

**Immediate Action (Weeks 1-4):**
1. **Start with Phase 1 (DSMC Core)** to build confidence and momentum
2. **Validate against analytical models** to ensure correctness
3. **Make go/no-go decision** at Week 4 checkpoint

**Medium-Term (Weeks 5-12):**
1. **Complete Option 2 core implementation**
2. **Generate validation report** comparing to Parodi and AeriSat models
3. **Integrate into SBIR proposal** with preliminary results

**Long-Term (6-12 months, if resourced):**
1. **Secure HPC allocation** (XSEDE startup or AWS)
2. **Begin Option 3 development** with dedicated team
3. **Target high-impact publication** in Journal of Electric Propulsion

### 9.2 Immediate Next Steps

**Week 1 Actions:**
```bash
# 1. Create GitHub repository
gh repo create AeriSat/aerisat-particle-sim --public --description "Particle-based simulation of ABEP systems"

# 2. Set up development environment
cd aerisat-particle-sim
python -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib pytest

# 3. Implement first test case
touch tests/test_free_molecular_flow.py
# (implement ballistic motion test)

# 4. Schedule kickoff meeting
# - Review Parodi paper together
# - Assign roles and responsibilities
# - Set up weekly check-ins
```

**Week 1 Deliverable:**
- Repository structure in place
- First passing test (ballistic motion)
- Development plan agreed upon

### 9.3 Decision Framework

**Should you pursue Option 2?**

✅ **YES, if:**
- You need validated performance predictions for SBIR
- You want to publish methodology/validation paper
- You have 3-4 months of development bandwidth
- Analytical model uncertainties are limiting confidence
- You want to build internal simulation capability

❌ **NO, if:**
- Analytical model is sufficient for current needs
- No bandwidth for development (focus on hardware)
- Immediate flight demonstration is the priority
- Commercial simulation codes can meet requirements

**Should you pursue Option 3?**

✅ **YES, if:**
- You've successfully completed Option 2
- You secured SBIR Phase II funding
- High-fidelity 3D predictions are needed for design freeze
- You're building a long-term simulation capability
- You have access to HPC resources

❌ **NO, if:**
- Option 2 results are sufficient
- Focus should remain on hardware development
- Can leverage commercial codes (SPIS-EP, SPARTA)
- Timeline to flight doesn't allow 12-month simulation development

### 9.4 Final Thoughts

Particle-based simulations represent the **gold standard** for ABEP system analysis, bridging the gap between analytical models and expensive hardware testing. By implementing Option 2, AeriSat would:

1. **De-risk design** with validated physics predictions
2. **Strengthen technical credibility** in proposals and publications
3. **Build internal expertise** in advanced simulation methods
4. **Generate IP** through novel algorithms and insights
5. **Enable rapid iteration** compared to hardware-only development

The investment of 8-12 weeks of focused development will pay dividends throughout the program, from seed funding pitches to final flight demonstration.

**The choice is yours, but the opportunity is clear.**

---

## 10. References and Resources

### 10.1 Key Papers

1. **Parodi et al. (2025)** - "Particle-based Simulation of an Air-Breathing Electric Propulsion System"
   - Primary methodology reference
   - Validation data source

2. **Bird (1994)** - "Molecular Gas Dynamics and the Direct Simulation of Gas Flows"
   - DSMC bible

3. **Birdsall & Langdon (2004)** - "Plasma Physics via Computer Simulation"
   - PIC bible

4. **Vahedi & Surendra (1995)** - "A Monte Carlo Collision Model for the Particle-in-Cell Method"
   - MCC algorithm reference

5. **Andreussi et al. (2022)** - "A review of air-breathing electric propulsion"
   - ABEP system overview

### 10.2 Software Resources

**Open-Source Codes:**
- **SPARTA:** https://sparta.github.io (DSMC)
- **XPDP1:** https://ptsg.egr.msu.edu (PIC, educational)
- **Smilei:** https://smileipic.github.io (PIC, advanced)
- **OpenFOAM-DSMC:** https://openfoam.org (DSMC module)

**Tutorials:**
- MIT OpenCourseWare: "Kinetic Modeling of Plasmas"
- Stanford: "Computational Plasma Physics"
- VKI Lecture Series on Electric Propulsion

**Cross-Section Databases:**
- LXCat: https://lxcat.net
- NIST Atomic Spectra Database: https://www.nist.gov/pml/atomic-spectra-database

### 10.3 Consultants / Collaborators

**Academic Groups:**
- **KU Leuven** (Lapenta group) - PIC methods, Parodi's institution
- **VKI** (Magin group) - ABEP modeling
- **MIT** (Peraire/Kamm) - DSMC methods
- **Michigan** (Boyd group) - Rarefied gas dynamics
- **Princeton** (PPPL) - Plasma simulation

**Industry:**
- **Busek** - ABEP thruster development
- **Techshot** - ABEP system integration
- **ArianeGroup** - Ion thruster expertise

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ABEP** | Air-Breathing Electric Propulsion - uses atmospheric particles as propellant |
| **DSMC** | Direct Simulation Monte Carlo - particle method for rarefied gas dynamics |
| **PIC** | Particle-in-Cell - particle method for plasma simulation |
| **MCC** | Monte Carlo Collisions - stochastic collision algorithm |
| **VLEO** | Very Low Earth Orbit - altitudes 180-450 km |
| **Kn** | Knudsen number - ratio of mean free path to characteristic length |
| **CLL** | Cercignani-Lampis-Lord - gas-surface interaction model |
| **VHS** | Variable Hard Sphere - collision model in DSMC |
| **FEM** | Finite Element Method - numerical technique for PDEs |
| **MPI** | Message Passing Interface - parallel computing standard |
| **HPC** | High-Performance Computing - supercomputers |

---

## Appendix B: Code Skeleton

**Complete file: `src/aerisat_psim/__init__.py`**
```python
"""
AeriSat Particle Simulation Package
====================================

Particle-based simulation of Air-Breathing Electric Propulsion systems.

Modules:
--------
- dsmc: Direct Simulation Monte Carlo for neutral gas flow
- pic: Particle-in-Cell with Monte Carlo Collisions for plasma
- mesh: Spatial discretization and particle indexing
- cross_sections: Collision cross-section database
- diagnostics: Output and visualization tools

Example:
--------
>>> from aerisat_psim import DSMC1D, Atmosphere
>>> atm = Atmosphere(altitude=200e3)
>>> sim = DSMC1D(length=0.2, n_cells=100)
>>> sim.inject_freestream(atm, velocity=7800)
>>> sim.run(t_final=10e-3)
>>> print(f"Compression ratio: {sim.compression_ratio():.1f}")

Authors: AeriSat Systems CTO Office
License: MIT (if not ITAR-restricted)
Version: 0.1.0
"""

from .constants import *
from .particles import Particle, ParticleArray
from .mesh import Mesh1D, Cell

from .dsmc import DSMC1D, Atmosphere, Intake
from .pic import PIC1D, RFDischarge

from .diagnostics import plot_density_profile, export_vtk

__version__ = "0.1.0"
__all__ = [
    'DSMC1D', 'PIC1D',
    'Atmosphere', 'Intake', 'RFDischarge',
    'Particle', 'ParticleArray',
    'Mesh1D', 'Cell'
]
```

---

**Document prepared for AeriSat Systems - October 2025**

**For questions or collaboration inquiries, contact: George, CTO**

---

*End of Implementation Plan*
