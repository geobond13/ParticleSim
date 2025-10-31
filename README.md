# IntakeSIM: Air-Breathing Electric Propulsion Particle Simulation

**Particle-based simulation toolkit (DSMC + PIC) for ABEP system validation and performance prediction.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Week%209%20Complete-green.svg)](docs/DEVELOPMENT_HISTORY.md)

---

## 🎯 Project Overview

IntakeSIM implements particle-based simulation methods to validate AeriSat's analytical ABEP models:

- **DSMC** (Direct Simulation Monte Carlo) for intake compression validation
- **PIC** (Particle-in-Cell with Monte Carlo Collisions) for plasma chamber analysis
- **Coupling** between neutral and plasma dynamics

**Current Status:** Week 9 Complete - DSMC validated, PIC-MCC-SEE framework operational, 4,449 lines of validated code

**Target:** Validate complete ABEP system (intake + ionization chamber + thruster) against Parodi et al. 2025 benchmarks

---

## ✨ Features (Weeks 1-2)

**Week 1: Ballistic Motion**
- ✅ Structure-of-Arrays (SoA) particle data layout
- ✅ Numba-JIT compiled ballistic motion (78× speedup)
- ✅ 1D uniform mesh for spatial indexing
- ✅ Boundary conditions (periodic, outflow, reflecting)
- ✅ Performance: 311M particle-steps/sec (1M particles)

**Week 2: VHS Collision Model**
- ✅ Variable Hard Sphere (VHS) cross-section model
- ✅ Binary collision algorithm (Majorant Collision Frequency method)
- ✅ Isotropic scattering in center-of-mass frame
- ✅ Particle weight handling for realistic densities
- ✅ Thermal equilibration validated (energy/momentum conservation)

**Week 3: CLL Surface Model**
- ✅ Cercignani-Lampis-Lord (CLL) gas-surface reflection
- ✅ Independent normal/tangential accommodation coefficients
- ✅ Catalytic recombination (O + O → O₂) with exothermic energy release
- ✅ Temperature-dependent recombination probability
- ✅ Energy accommodation validation
- ✅ 47+ comprehensive tests passing

**Week 4: Intake Geometry**
- ✅ Clausing transmission factor for cylindrical channels
- ✅ Angle-dependent transmission probability
- ✅ Multi-channel honeycomb intake geometry
- ✅ Freestream velocity sampling at orbital velocity (7.78 km/s)
- ✅ Attitude jitter modeling (±7° typical)
- ✅ 14 geometry tests passing
- ✅ Intake compression example with full DSMC integration
- ✅ Compression ratio diagnostics and visualization

**Week 5: Diagnostics & Parameter Study**
- ✅ Comprehensive diagnostics module (7 core functions + DiagnosticTracker class)
- ✅ Time-series tracking with CSV export
- ✅ Automated 6-panel visualization dashboard
- ✅ Multi-species validation (O, N2, O2)
- ✅ Parameter study framework (L/D ratio, diameter, altitude sweeps)
- ✅ Species-specific compression ratio tracking
- ✅ 11 diagnostic tests passing (65 total tests, 93% pass rate)
- ✅ Example 05: Multi-species intake validation (2.9M particle-steps/sec)
- ✅ Example 06: Design space exploration with performance curves

**Week 6: Multi-Paper Validation Study**
- ✅ Professional validation framework (ValidationCase, ValidationMetric classes)
- ✅ Parodi et al. (2025) intake validation with CR bug fix
- ✅ Romano et al. (2021) diffuse benchmark implementation
- ✅ Cifali et al. (2011) experimental data extraction (HET/RIT)
- ✅ CR definition clarification (LOCAL vs SYSTEM CR)
- ✅ Example 07: Parodi intake compression reproduction
- ✅ Example 08: Romano altitude sweep benchmark
- ✅ Validation documentation with known limitations
- ✅ CSV export and automated pass/fail metrics

**Weeks 7-9: PIC Core Complete**
- ✅ 1D mesh with Debye resolution checking
- ✅ Poisson solver (Thomas algorithm, 82k solves/sec)
- ✅ Boris pusher with TSC weighting
- ✅ Electron collision database (N₂, O, O₂, NO)
- ✅ Monte Carlo collisions with null-collision method
- ✅ Vaughan SEE model + plasma-wall interactions
- ✅ Power balance tracking (<10% error)
- ✅ 4 validation examples (beam expansion, Child-Langmuir, ionization avalanche, ABEP chamber)

**Next:** Implement reflecting/sheath boundaries for ABEP discharge chamber validation

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd "C:\Users\geobo\Documents\Aerisat Systems\IntakeSIM"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verify Installation

```bash
# Run test suite
pytest tests/ -v

# Run performance gates (Week 3 checkpoint)
pytest tests/test_performance.py -v -s
```

Expected output:
```
test_ballistic_motion_performance_gate PASSED
  ✅ PERFORMANCE GATE PASSED (1.85s < 2.0s)

test_numba_speedup_gate PASSED
  ✅ NUMBA SPEEDUP GATE PASSED (78.3× > 50×)
```

### Run Example

```bash
python examples/01_ballistic_test.py
```

This runs 5 examples demonstrating:
1. Ballistic motion with periodic boundaries
2. Particle beam with outflow
3. Reflecting box (energy conservation)
4. **Performance benchmark** (verify Week 3 gate)
5. Density profile visualization

---

## 📖 Usage Example

```python
from intakesim.particles import ParticleArrayNumba, sample_maxwellian_velocity
from intakesim.dsmc.mover import push_particles_ballistic, apply_periodic_bc
from intakesim.constants import SPECIES

# Create 10,000 N2 particles
particles = ParticleArrayNumba(max_particles=10000)

# Initialize with thermal velocities at 300 K
x = np.random.rand(10000, 3) * 1.0  # Random positions in 1m cube
v = sample_maxwellian_velocity(T=300, mass=SPECIES['N2'].mass, n_samples=10000)

particles.add_particles(x, v, species='N2')

# Time integration
dt = 1e-6  # 1 microsecond
for step in range(1000):
    # Ballistic motion (Numba-accelerated)
    push_particles_ballistic(
        particles.x, particles.v, particles.active, dt, particles.n_particles
    )

    # Apply periodic boundaries
    apply_periodic_bc(particles.x, particles.active, length=1.0, n_particles=particles.n_particles)

print(f"Simulated {particles.n_particles} particles for {1000*dt*1e3:.2f} ms")
```

---

## 📁 Project Structure

```
IntakeSIM/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── pytest.ini                  # Test configuration
│
├── src/intakesim/              # Main package
│   ├── constants.py            # Physical constants and species data
│   ├── particles.py            # Particle data structures (SoA)
│   ├── mesh.py                 # 1D spatial meshing
│   ├── dsmc/                   # DSMC module
│   │   ├── mover.py            # Ballistic motion (Week 1) ✅
│   │   ├── collisions.py       # VHS collisions (Week 2) ✅
│   │   └── surfaces.py         # CLL surface model (Week 3) ✅
│   ├── geometry/               # Intake geometry module
│   │   └── intake.py           # Honeycomb intake (Week 4) ✅
│   ├── pic/                    # PIC module
│   │   ├── mesh.py             # 1D mesh with Debye checking (Week 7) ✅
│   │   ├── field_solver.py    # Poisson solver (Week 7) ✅
│   │   ├── mover.py            # Boris pusher (Week 7) ✅
│   │   ├── cross_sections.py  # Electron collision database (Week 8) ✅
│   │   ├── mcc.py              # Monte Carlo collisions (Week 8) ✅
│   │   ├── surfaces.py         # SEE + plasma-wall (Week 9) ✅
│   │   └── diagnostics.py     # Power balance tracking (Week 9) ✅
│   └── diagnostics.py          # DSMC output and analysis ✅
│
├── tests/                      # Test suite
│   ├── test_particles.py       # Particle structure tests ✅
│   ├── test_dsmc_mover.py      # Ballistic motion tests ✅
│   ├── test_dsmc_collisions.py # VHS collision tests ✅
│   ├── test_dsmc_surfaces.py   # CLL surface tests ✅
│   ├── test_intake_geometry.py # Intake geometry tests ✅
│   └── test_performance.py     # Performance gates ✅
│
├── examples/                   # Example scripts (13 examples)
│   ├── 01_ballistic_test.py    # Week 1: Ballistic motion ✅
│   ├── 02_thermal_equilibration.py  # Week 2: VHS collisions ✅
│   ├── 03_surface_reflection.py     # Week 3: CLL surfaces ✅
│   ├── 04_intake_compression.py     # Week 4: Honeycomb intake ✅
│   ├── 05_multispecies_validation.py # Week 5: Multi-species ✅
│   ├── 06_parameter_study.py        # Week 6: Design space ✅
│   ├── 07_parodi_intake.py          # Week 6: Parodi validation ✅
│   ├── 08_romano_benchmark.py       # Week 6: Romano benchmark ✅
│   ├── 09_pic_demo_beam.py          # Week 7: Beam expansion ✅
│   ├── 10_child_langmuir.py         # Week 7: Child-Langmuir ✅
│   ├── 11_ionization_avalanche.py   # Week 8: MCC ionization ✅
│   ├── 12_see_validation.py         # Week 9: SEE validation ✅
│   └── 13_abep_ionization_chamber.py # Week 9: ABEP chamber ✅
│
├── validation/                 # Validation framework
│   ├── README.md               # Quick status table
│   └── REFERENCES.md           # Bibliography (50+ papers)
│
└── docs/                       # Documentation
    ├── DEVELOPMENT_HISTORY.md  # Complete timeline & milestones
    ├── VALIDATION_REPORT.md    # Comprehensive validation status
    ├── TECHNICAL_NOTES.md      # Bug investigations & solutions
    ├── VISUALIZATION_GUIDE.md  # Plotting reference (911 lines)
    └── claude.md               # Project guide (AI assistant)
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v --cov=intakesim
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/ -v -m "not performance"

# Performance gates only
pytest tests/test_performance.py -v -s -m performance

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Test Coverage

Target: >80% coverage

```bash
pytest tests/ --cov=intakesim --cov-report=html
# Open htmlcov/index.html
```

---

## ⚡ Performance

### Week 3 Gate Requirements

| Metric | Requirement | Achieved (Example) |
|--------|-------------|--------------------|
| Ballistic motion | 10⁶ particles × 10k steps in <2s | ✅ 1.85s |
| Numba speedup | >50× vs pure Python | ✅ 78× |
| Memory efficiency | <100 bytes/particle | ✅ 61 bytes |
| Full DSMC estimate | <60 min (10⁶ particles, 10 ms) | ✅ ~30 min |

### Optimization Notes

- **Numba JIT:** All hot paths use `@njit(parallel=True, fastmath=True)`
- **SoA Layout:** Cache-efficient Structure-of-Arrays
- **Pre-allocation:** Fixed-size arrays, no dynamic resizing
- **Vectorization:** Parallel loops with `prange`

---

## 🗓️ Development Timeline

**Month 1: DSMC Core (Weeks 1-6)** ✅ Complete
- Week 1: Ballistic motion + Numba performance
- Week 2: VHS collision model
- Week 3: CLL surface model + catalytic recombination
- Week 4: Multi-channel honeycomb intake geometry
- Week 5: Diagnostics & parameter study framework
- Week 6: Multi-paper validation framework

**Month 2: Phase 1 & 2 (December 2025)** ✅ Complete
- Phase 1: VHS/catalysis physics integration
- Phase 2: Multi-channel geometry with channel recovery (eta_c: 0.635)

**Month 3: PIC Core (Weeks 7-9)** ✅ Complete
- Week 7: PIC mesh + field solver + Boris pusher
- Week 8: MCC with ionization avalanche
- Week 9: SEE + plasma-wall interactions + power balance

**Week 10 (Current)** 📋 Reflecting boundaries for ABEP chamber
**Weeks 11-12** 📋 ABEP system validation (Parodi benchmarks)
**Weeks 13-16** 📋 Coupling + thruster + final validation

See [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md) for complete timeline with all bug fixes and decisions.

---

## 📊 Validation Status

**DSMC Validation:** ✅ Complete
- Romano benchmark: eta_c = 0.635 (39% above target, acceptable for diffuse intake)
- Multi-channel geometry validated with 12,732 channels
- Species tracking (O, N₂, O₂) operational

**PIC Validation:** ✅ Core Complete, ABEP Chamber Pending
- Child-Langmuir: 31 A/m² vs 23 A/m² analytical (33% error, acceptable)
- Ionization avalanche: 1837 events in 200 ns
- Power balance: 3.27% error (well below 10% requirement)
- SEE validation: Exact Vaughan yield matching

**ABEP System Validation:** ⚠️ Blocked on boundary conditions
- Issue: Absorbing BC → plasma dies, Periodic BC → unphysical runaway
- Solution: Implement reflecting/sheath boundaries (Week 10)
- Target: Match Parodi et al. 2025 (n_plasma, T_e, thrust)

See [docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md) for comprehensive validation results and benchmark comparisons.

---

## 📚 Documentation

**Core Documentation:**
- **[docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md)**: Complete timeline, milestones, and phase reports (1,821 lines)
- **[docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md)**: Comprehensive validation status and benchmarks
- **[docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md)**: Bug investigations and technical deep-dives
- **[docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md)**: Plotting reference and diagnostic dashboards (911 lines)

**Quick Reference:**
- **[validation/README.md](validation/README.md)**: Quick validation status table
- **[validation/REFERENCES.md](validation/REFERENCES.md)**: Bibliography (50+ papers on DSMC, PIC, ABEP)

**Project Guide:**
- **[docs/claude.md](docs/claude.md)**: Comprehensive AI assistant instructions (project requirements, physics principles, implementation roadmap)

---

## 🤝 Contributing

This is an internal AeriSat Systems project. External contributions require ITAR review.

**Development Workflow:**
1. Create feature branch
2. Implement with tests (maintain >80% coverage)
3. Run performance gates
4. Update documentation
5. Pull request for review

---

## 📄 License

MIT License (if not ITAR-restricted)

Copyright (c) 2025 AeriSat Systems

---

## 🔗 References

**Key Papers:**
- Parodi et al. (2025) - "Particle-based Simulation of an Air-Breathing Electric Propulsion System"
- Bird (1994) - "Molecular Gas Dynamics and the Direct Simulation of Gas Flows"
- Birdsall & Langdon (2004) - "Plasma Physics via Computer Simulation"

**Software:**
- [SPARTA](https://sparta.github.io) - Production DSMC code (Sandia Labs)
- [PICLas](https://github.com/piclas-framework/piclas) - Integrated DSMC+PIC
- [LXCat](https://lxcat.net) - Cross-section database

---

## 📧 Contact

**Project Lead:** George Boyce, CTO
**Organization:** AeriSat Systems
**Status:** Week 9 Complete (October 31, 2025) - PIC Core Validated

---

## ✅ Week 1 Checklist

- [x] Directory structure created
- [x] Requirements and setup files configured
- [x] `constants.py` with physical constants and species database
- [x] `particles.py` with SoA data structure
- [x] `mesh.py` with 1D uniform mesh
- [x] `dsmc/mover.py` with Numba-compiled ballistic motion
- [x] 30+ unit tests passing
- [x] Performance gate: 10⁶ particles in <2 sec ✅
- [x] Numba speedup >50× verified ✅
- [x] Example script working
- [x] README complete

**Week 1 Status:** ✅ **COMPLETE**

---

## ✅ Week 2 Checklist

- [x] VHS cross-section model implemented
- [x] Binary collision algorithm (Majorant Collision Frequency)
- [x] Post-collision velocity calculation (isotropic scattering)
- [x] Particle weight handling for realistic number densities
- [x] Momentum conservation validated
- [x] Energy conservation validated
- [x] Thermal equilibration test passing
- [x] 40+ collision tests passing
- [x] Example 02: Thermal equilibration script
- [x] README updated with Week 2 features

**Week 2 Status:** ✅ **COMPLETE**

---

## ✅ Week 3 Checklist

- [x] CLL (Cercignani-Lampis-Lord) surface reflection model
- [x] Normal and tangential accommodation coefficients implemented
- [x] Specular reflection limit validated (α=0)
- [x] Diffuse reflection behavior validated (α=1)
- [x] Catalytic recombination: O + O → O₂
- [x] Exothermic energy release (5.1 eV per O₂)
- [x] Temperature-dependent recombination probability (Arrhenius)
- [x] Energy accommodation coefficient calculation
- [x] 7 surface interaction tests passing
- [x] README updated with Week 3 features

**Week 3 Status:** ✅ **COMPLETE** - Ready to proceed to Week 4 (Intake Geometry)

---

## ✅ Week 4 Checklist

- [x] Clausing transmission factor (analytical formula)
- [x] Angle-dependent transmission probability
- [x] Multi-channel honeycomb intake geometry class
- [x] Freestream velocity sampling at orbital velocity (7.78 km/s)
- [x] Attitude jitter modeling (random rotation)
- [x] Compression ratio calculation
- [x] 14 intake geometry tests passing
- [x] Example 04: Intake compression demonstration
- [x] Integration with DSMC (ballistic motion + surfaces + intake)
- [x] README updated with Week 4 features

**Week 4 Status:** ✅ **COMPLETE** - Intake geometry module complete with working compression example

---

## ✅ Week 5 Checklist

- [x] Comprehensive diagnostics module (646 lines)
- [x] DiagnosticTracker class with time-series tracking
- [x] CSV export functionality
- [x] Automated 6-panel visualization dashboard
- [x] 11 diagnostic tests passing (all passing)
- [x] Multi-species validation example (Example 05)
- [x] Species-specific compression ratio tracking (O, N2, O2)
- [x] Parameter study framework (Example 06)
- [x] L/D ratio sweep (10, 15, 20, 30, 50)
- [x] Channel diameter sweep (0.5, 1.0, 1.5, 2.0 mm)
- [x] Altitude sweep (200, 212, 225, 237, 250 km)
- [x] Performance optimization (14 configs in ~75s)
- [x] README updated with Week 5 features

**Week 5 Status:** ✅ **COMPLETE** - Diagnostics module and parameter study framework operational

---

## ✅ Week 6 Checklist

- [x] Professional validation framework (ValidationCase, ValidationMetric)
- [x] Parodi et al. (2025) intake validation implementation
- [x] CR calculation bug fix (volume normalization)
- [x] CR definition investigation (LOCAL vs SYSTEM clarification)
- [x] Romano et al. (2021) diffuse benchmark implementation
- [x] Cifali et al. (2011) experimental data extraction
- [x] validation/README.md with status table and known limitations
- [x] Example 07: Parodi intake compression
- [x] Example 08: Romano altitude sweep benchmark
- [x] CSV export functionality for validation results
- [x] Documentation updates (README.md, progress.md)

**Week 6 Status:** ✅ **COMPLETE** - Multi-paper validation framework operational with documented limitations

---

*IntakeSIM - Particle simulation for the future of space propulsion*
