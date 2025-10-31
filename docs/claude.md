# IntakeSIM - ABEP Particle Simulation Project

## PROJECT STATUS: 📋 PLANNING PHASE (Not Yet Implemented)

**Mission**: Develop particle-based simulation capability to validate and refine AeriSat's analytical ABEP models through high-fidelity Direct Simulation Monte Carlo (DSMC) and Particle-in-Cell (PIC) methods.

**Date Created**: October 29, 2025
**Current Phase**: Architecture Selection & Tool Evaluation
**Target Completion**: Q2 2026 (Option 2) or Q4 2026 (Option 3)

**Key Deliverables**:
- ✅ Complete planning documentation (2,866 total lines across 3 documents)
- 📋 Tool selection decision pending (SPARTA vs PICLas vs Python)
- 🎯 Target: Validated simulation matching Parodi et al. within 30%

---

## Core Principles

### Rule #1: Physics First, Not Assumptions

**The Technical Addendum taught us:**
- Do NOT use simple tapered ducts → Use multi-channel honeycomb with Clausing transmission
- Do NOT ignore surface chemistry → Catalytic recombination changes O→O₂ compression by 5×
- Do NOT use 1D axial PIC for azimuthal RF → Use effective heating or 2D electrostatic model
- Do NOT skip SEE/charge exchange → These change T_e by 1-2 eV and affect current extraction
- Do NOT use one-way coupling → Neutral depletion creates 10-20% feedback effects

**Validation is mandatory:**
- Every physics model must pass analytical limits (Kn→0, Kn→∞, power balance)
- Compare to literature benchmarks (Turner CCP, Sharipov Poiseuille, Parodi ABEP)
- Document all assumptions and their validity ranges

### Rule #2: Performance Matters from Day 1

**Pure Python is 10-100× too slow for realistic particle counts.**

**If using Python (Option 2):**
- Numba @njit compilation is MANDATORY, not optional
- Structure-of-Arrays (SoA) data layout for cache efficiency
- Vectorized operations over particle arrays
- Performance gates at Week 3: 10⁶ particles in <60 min

**If using compiled codes (SPARTA/PICLas):**
- MPI parallelization for production runs
- Proper load balancing across domains
- Profile before optimizing

### Rule #3: Validate Constantly, Document Everything

**Test hierarchy:**
1. Unit tests: Individual algorithms (Boris pusher, VHS collisions)
2. Integration tests: Coupled systems (DSMC→PIC mass conservation)
3. Physics benchmarks: Analytical solutions (Child-Langmuir, thermal transpiration)
4. Validation cases: Literature data (Parodi et al., Turner benchmark)
5. Power balance: <10% error required for all PIC simulations

**Documentation requirements:**
- Theory manual: Mathematical foundations
- User guide: Installation, examples, API reference
- Validation report: All test cases with uncertainty quantification
- Publication strategy: Conference abstracts → journal papers

---

## The Physics Problem

### System Overview

**Air-Breathing Electric Propulsion (ABEP) for VLEO CubeSats:**

1. **Intake** (200 km altitude):
   - Freestream: n ~ 4.2×10¹⁷ m⁻³, v_orbital = 7.8 km/s, T ~ 1000 K
   - Species: O (83%), N₂ (14%), O₂ (2%), NO (<1%)
   - Goal: Compress neutrals by 400-500× through honeycomb channels

2. **Ionization Chamber**:
   - RF discharge at 13.56 MHz, 20 W absorbed power
   - Target: n_plasma ~ 1.65×10¹⁷ m⁻³, T_e ~ 7.8 eV
   - Ion production → extraction through grids

3. **Thruster**:
   - Ion acceleration through electrostatic grids
   - Target thrust: 480 μN with <1 kg mass flow per year
   - Specific impulse: ~3000-5000 s

### Why Particle Methods?

**Analytical models assume:**
- Maxwell-Boltzmann equilibrium (invalid at Kn > 0.1)
- Lumped parameters for compression and ionization
- No kinetic effects, no velocity distribution details

**Particle methods capture:**
- Non-equilibrium velocity distributions in rarefied intake
- Molecular-scale gas-surface interactions (sticking, recombination)
- Energy distributions → ion thruster performance
- Transitional flow regime (0.1 < Kn < 10) accurately

**Key Physics Regimes:**

| Region | Knudsen Number | Method | Time Scale | Spatial Scale |
|--------|----------------|--------|------------|---------------|
| Freestream (200 km) | Kn >> 1 | Ballistic | dt ~ 1 μs | λ_mfp ~ 1 m |
| Intake channels | 0.1 < Kn < 10 | DSMC | dt ~ 0.1 μs | dx ~ 1 mm |
| Ionization chamber | Kn ~ 0.01 | PIC-MCC | dt ~ 0.1 ns | dx ~ λ_D ~ 0.1 mm |
| Plume | Kn > 1 | PIC or hybrid | dt ~ 0.1 ns | dx ~ 1 cm |

---

## Implementation Options

### Option 2: Python Prototype (12-16 weeks) - **RECOMMENDED for Phase 1**

**Scope:**
- 1D/2D DSMC for intake compression validation
- 0D/1D PIC-MCC for plasma density evolution
- One-way coupling DSMC→PIC with iterative neutral depletion feedback
- Validation against Parodi et al. (2025) and analytical models

**Timeline:** 12-16 weeks (NOT 8-12 as originally estimated)

**Resource Requirements:**
- Personnel: 1 developer @ 0.5 FTE (3-4 months)
- Hardware: 8-core laptop/workstation, 16-32 GB RAM
- Budget: $22k-$98k depending on support level

**Deliverables:**
- Python package `aerisat_psim` with DSMC and PIC modules
- Validation report comparing to Parodi et al.
- Example scripts for intake and thruster parametric studies
- Documentation (theory, user guide, API reference)

**Pros:**
- ✅ Full control over physics implementation
- ✅ Rapid prototyping and iteration
- ✅ Easy integration with existing AeriSat Python tools
- ✅ Low-risk investment for proof-of-concept
- ✅ Learning by building → deep understanding

**Cons:**
- ❌ Limited to 1D/2D geometry (no full 3D visualization)
- ❌ Slower than compiled codes (even with Numba: 10⁶ particles ~ 60 min)
- ❌ Not production-quality for large parametric sweeps

**When to choose:** SBIR Phase I proof-of-concept, physics validation, investor demonstrations

---

### Option 3a: SPARTA + Custom PIC (6-9 months) - **Documents' Original Recommendation**

**Scope:**
- Use SPARTA (Sandia Labs) for production-quality 3D DSMC
- Develop custom 3D PIC module in Python or C++
- Couple via file I/O (DSMC output → PIC input)
- MPI parallelization for HPC clusters

**SPARTA Advantages:**
- ✅ **Production-quality DSMC** from Sandia National Labs
- ✅ **Proven 3D unstructured mesh** capability
- ✅ **MPI-parallel** with excellent scaling (tested to 100k cores)
- ✅ **Active development** and extensive documentation
- ✅ **Open-source** (GPL) with large user community

**Implementation Strategy:**
1. Learn SPARTA (tutorials, examples) - 2 weeks
2. Implement ABEP surface models (CLL + catalysis) as custom "fix" - 4 weeks
3. Validate 3D intake against Option 2 results - 2 weeks
4. Develop 3D PIC module separately - 12 weeks
5. Couple systems via VTK file I/O - 2 weeks
6. Applications and optimization - 4 weeks

**Timeline:** 6-9 months

**Resource Requirements:**
- Personnel: 2-3 FTE (lead developer + HPC specialist)
- Hardware: HPC allocation (XSEDE free or AWS ~$20k)
- Budget: $185k-$315k

**Pros:**
- ✅ Production-quality DSMC immediately available
- ✅ Focus development effort on novel PIC physics
- ✅ 3D geometry for publications and investor demos
- ✅ Proven parallel performance

**Cons:**
- ❌ C++ learning curve for SPARTA modifications
- ❌ PIC development still needed (6+ months)
- ❌ Coupling via file I/O (not as tight as native coupling)
- ❌ Two separate codebases to maintain

**When to choose:** SBIR Phase II with funding secured, need high-fidelity 3D predictions, have HPC access

---

### Option 3b: PICLas Integrated Framework (4-6 months) - **NEW RECOMMENDATION**

**What is PICLas?**
PICLas is an open-source particle simulation framework from University of Stuttgart that **natively combines DSMC and PIC in a single code**, designed specifically for coupled neutral-plasma flows.

**Key Features:**
- ✅ **Integrated DSMC + PIC** in single framework (no file I/O coupling)
- ✅ **3D unstructured mesh** with high-order DG methods
- ✅ **MPI-parallel** with dynamic load balancing
- ✅ **Proven applications**: Hypersonic reentry, ion thrusters, atmospheric plasmas
- ✅ **Active development**: University of Stuttgart, ESA collaborations
- ✅ **Open-source**: GPL-3.0

**Potential Advantages over SPARTA+Custom:**
- ✅ Native DSMC-PIC coupling (better than file I/O)
- ✅ Single codebase to learn and maintain
- ✅ Could save 3-6 months vs building custom PIC
- ✅ Existing examples for EP applications

**Potential Challenges:**
- ❓ **CRITICAL QUESTION**: Can PICLas handle ABEP-specific physics?
  - Effective RF heating models (not self-consistent ICP)
  - Catalytic surface recombination (O + O → O₂)
  - Multi-channel honeycomb intake geometry
  - Secondary electron emission (SEE) with Vaughan model
  - Charge exchange reactions (O⁺ + O, N₂⁺ + N₂)
- ❌ Fortran codebase (harder to extend than Python/C++)
- ❌ Steeper learning curve initially
- ❌ Smaller user community than SPARTA

**Evaluation Plan (Month 5 after Option 2):**
```
Week 1: Install PICLas, run tutorials
Week 2: Test honeycomb geometry import (Gmsh → PICLas)
Week 3: Implement test case: Can we model RF heating?
Week 4: Implement test case: Can we model catalytic surfaces?
Week 5: Compare performance: PICLas vs Option 2 Python prototype
Week 6: Decision: PICLas vs SPARTA+custom vs extend Python
```

**When to choose:** After Option 2 success, if evaluation shows PICLas can handle ABEP physics, before committing to SPARTA+custom

---

## Recommended Implementation Path

### Phase 1 (Months 1-4): Python Prototype - EXECUTE NOW

**Implement Option 2 with all Technical Addendum corrections:**

**Month 1: DSMC Core + Numba Optimization**
- Week 1: Particle arrays (SoA), Numba push, basic 1D mesh
- Week 2: VHS collisions with majorant frequency method
- Week 3: CLL + catalytic recombination, thermal transpiration validation
- **Checkpoint**: Performance gate (10⁶ particles <60 min), >50× speedup vs pure Python

**Month 2: DSMC Intake with Realistic Physics**
- Week 4: Multi-channel honeycomb with Clausing transmission
- Week 5: Angled freestream injection, AO surface aging model
- Week 6: Parametric studies (altitude, accommodation, geometry)
- **Checkpoint**: N₂ CR = 400-550, matches Parodi within 30%

**Month 3: PIC Core + Plasma-Surface Physics**
- Week 7: 1D Poisson + Boris pusher with TSC deposition
- Week 8: MCC with full chemistry {O, N₂, O₂, NO, ions, electrons, CEX}
- Week 9: SEE & ion-induced emission, CCP benchmark
- **Checkpoint**: Turner benchmark within 20%, power balance <10% error

**Month 4: PIC Thruster + Coupling**
- Week 10: Effective RF heating model, power balance validation
- Week 11: Thruster geometry with ion extraction, Parodi validation
- Week 12: One-way coupling + neutral depletion iteration
- Week 13-14: Convergence testing, uncertainty quantification
- **Checkpoint**: n_plasma within 30%, T_e within 20%, coupling converges <10 iterations

**Month 4 (End): Documentation & Delivery**
- Week 15: Validation report, publication drafts
- Week 16: GitHub release, SBIR integration, conference abstracts
- **Deliverable**: Working Python simulation + validation report

### Phase 2 (Month 5): Production Tool Evaluation - PAUSE & ASSESS

**Critical Decision Point: Do we need 3D production capability?**

**Evaluation Criteria:**
- [ ] Did Option 2 reveal limitations requiring 3D geometry?
- [ ] Is SBIR Phase II funding secured ($185k-$315k)?
- [ ] Do investors/reviewers require high-fidelity 3D visualizations?
- [ ] Are parametric sweeps limited by Python performance?

**If YES to ≥3 criteria → Evaluate production tools:**

**Week 1-2: PICLas Deep Dive**
- Install PICLas, run all tutorials
- Test ABEP-specific physics:
  - Can it model effective RF heating (not just self-consistent)?
  - Can we implement catalytic surfaces with custom wall functions?
  - Does chemistry system support our species set?
- Prototype 3D honeycomb intake geometry

**Week 3-4: SPARTA Evaluation (if PICLas insufficient)**
- Install SPARTA, run DSMC examples
- Assess custom "fix" development for ABEP surfaces
- Estimate PIC development effort (compare to extending Python PIC to 3D)

**Week 5-6: Decision & Architecture**
- **Path A**: PICLas works → Plan PICLas implementation (4-6 months)
- **Path B**: PICLas limited → SPARTA + extend Python PIC to 3D (6-9 months)
- **Path C**: Python sufficient → Optimize Python with Cython/C++ (2-3 months)

**If NO to evaluation criteria → Conclude with Option 2:**
- Invest saved resources in hardware development
- Use Python simulation for design optimization within 1D/2D validity
- Revisit 3D decision at SBIR Phase II proposal time

### Phase 3 (Months 6-12): Production Implementation (if pursuing)

**Depends on Phase 2 decision:**

**If PICLas selected:**
- Months 6-7: Implement ABEP physics models in PICLas
- Months 8-9: Validate against Python prototype
- Months 10-11: Production runs on HPC, parametric sweeps
- Month 12: Publications, SBIR deliverables

**If SPARTA + PIC selected:**
- Months 6-8: SPARTA intake simulations + validation
- Months 9-11: 3D PIC development (extend Python or C++)
- Month 12: Coupled system validation + publications

**If Python optimization selected:**
- Months 6-7: Cython/C++ hot-path optimization
- Month 8: Limited 2D extensions where critical
- Months 9-10: Large parametric studies
- Months 11-12: Publications + applications

---

## Critical Physics Requirements (All Options)

Regardless of framework choice, these physics models are **MANDATORY**:

### 1. Multi-Channel Honeycomb Intake

**❌ DO NOT:**
```python
# Simple 1D tapered duct
area_ratio = (d_inlet / d_outlet)**2  # WRONG
```

**✅ DO:**
```python
class HoneycombIntake:
    """
    n_channels circular tubes with Clausing transmission.

    K(θ) = K₀ * cos(θ) / (1 + A*sin²(θ))

    where K₀ = Clausing factor for L/D ratio
    """
    def transmission_probability(self, incident_angle):
        LD = self.channel_length / self.channel_diameter
        K_0 = self._clausing_factor(LD)  # ~0.6 for L/D=15
        cos_theta = np.cos(incident_angle)
        return K_0 * cos_theta / (1 + 0.5*LD*(1-cos_theta**2))
```

### 2. Complete VLEO Chemistry

**Species required:**
- Neutrals: O, N₂, O₂, NO
- Ions: O⁺, N₂⁺, O₂⁺, NO⁺
- Electrons: e⁻

**Reactions required:**
- Electron-impact ionization: e + N₂ → N₂⁺ + 2e (threshold 15.58 eV)
- Charge exchange: O⁺ + O → O + O⁺ (σ ~ 2×10⁻¹⁹ m², **LARGE!**)
- Catalytic recombination: O + O(surface) → O₂ (γ ~ 0.01-0.1)
- Ion-neutral: O⁺ + N₂ → NO⁺ + N

**Data sources:**
- LXCat database (Biagi, Phelps) for electron collisions
- Literature fits for ion-neutral and charge exchange

### 3. RF Heating: Effective Model or 2D Electrostatic

**❌ DO NOT claim self-consistent 1D ICP:**
Parodi's RF heating is azimuthal (E_θ). 1D axial PIC **cannot** represent this.

**✅ DO use effective heating:**
```python
class EffectiveRFHeating:
    """
    Stochastic heating calibrated to P_abs = 20 W.

    DISCLAIMER: Closure model, not self-consistent ICP.
    Each electron gains random energy with variance set by target power.
    """
    def apply_heating(self, electrons, dt):
        sigma_E_eV = np.sqrt(2 * self.P_target * dt / (len(electrons) * e))
        for e in electrons:
            dE = np.random.normal(0, sigma_E_eV)
            e.energy = max(0.5, e.energy + dE)  # Floor at 0.5 eV
```

**✅ OR use 2D (r,z) with prescribed E_θ:**
```python
def azimuthal_E_field(r, t):
    """
    E_θ(r,t) from Faraday's law with prescribed B(t).

    Still not fully self-consistent, but captures radial profile.
    """
    return -mu_0 * pi * f_rf * (N_turns/L_coil) * I_coil(t) * r
```

**Document in all outputs:**
> "RF heating modeled via effective collision frequency calibrated to P_abs = 20 W. This is not a self-consistent electromagnetic PIC solution."

### 4. Secondary Electron Emission (SEE)

**Mandatory for realistic wall physics:**

```python
def see_yield(E_impact_eV, material='molybdenum'):
    """
    Vaughan formula: δ(E) = δ_max * (E/E_max)^n * exp(n*(1-E/E_max))

    Typical parameters:
    - Molybdenum: δ_max = 1.25, E_max = 350 eV
    - Ceramic: δ_max = 2.5, E_max = 300 eV
    """
    n = 0.62
    E_ratio = E_impact_eV / E_max
    return delta_max * E_ratio**n * np.exp(n * (1 - E_ratio))
```

**Impact:** Lowers T_e by 1-2 eV, reduces sheath potential, critical for grid extraction

### 5. Charge Exchange Reactions

**O⁺ + O → O + O⁺ has σ ~ 2×10⁻¹⁹ m² (HUGE!)**

**Effect:**
- Creates slow ions in plume → divergence
- Reduces thrust efficiency by ~10-20%
- **Must include** for realistic performance predictions

### 6. Catalytic Surface Recombination

**O + O(adsorbed) → O₂ + 5.1 eV**

```python
def surface_recombination(particle, surface):
    """
    Temperature-dependent catalytic coefficient.

    γ(T) = γ₀ * exp(-E_a / kT)

    Typical: γ₀ ~ 0.01-0.15 for metals at 700 K
    """
    if particle.species == 'O':
        gamma = surface.recombination_coefficient(T_wall)
        if np.random.random() < gamma:
            # Recombine into O₂
            particle.species = 'O2'
            particle.mass = 32 * AMU
            # Energy release → thermal velocity at T_wall
            particle.velocity = sample_maxwellian(T_wall, particle.mass)
```

**Impact:** Reduces O compression by 5× (O₂ has lower CR), critical for accurate predictions

### 7. Neutral Depletion Feedback

**❌ DO NOT use one-way coupling:**
```python
# WRONG: PIC uses fixed neutral density
n_neutral = dsmc.outlet_density()  # Set once
pic.run(n_neutral=n_neutral)  # No feedback
```

**✅ DO iterate until convergence:**
```python
# CORRECT: Iterative coupling
for iteration in range(max_iterations):
    # PIC consumes neutrals
    ionization_rate = pic.volumetric_ionization()  # m^-3 s^-1

    # DSMC updates with sink
    dsmc.add_volumetric_sink(ionization_rate)
    n_neutral_new = dsmc.run()['outlet_density']

    # Check convergence
    if abs(n_neutral_new - n_neutral_old) / n_neutral_old < 0.05:
        break  # Converged!

    n_neutral_old = 0.5*n_neutral_new + 0.5*n_neutral_old  # Under-relax
```

**Impact:** 10-20% density change in thruster, affects n_plasma and thrust predictions

### 8. Power Balance Validation

**MANDATORY CHECK for all PIC simulations:**

```python
def validate_power_balance(pic_sim):
    """
    P_in ≈ P_out within 10%

    CRITICAL VALIDATION - if this fails, physics is wrong!
    """
    P_in = pic_sim.rf_absorbed_power()

    P_ion_loss = pic_sim.ion_flux_to_walls() * pic_sim.sheath_potential() * e
    P_electron_loss = pic_sim.electron_flux_to_walls() * pic_sim.electron_temp() * e
    P_ionization = pic_sim.ionization_power()
    P_excitation = pic_sim.excitation_power()

    P_out = P_ion_loss + P_electron_loss + P_ionization + P_excitation

    error = abs(P_in - P_out) / P_in

    assert error < 0.10, f"Power balance error {error:.1%} > 10%!"
```

---

## Performance Requirements

### Python (Option 2) with Numba

**Mandatory from Day 1:**
- All hot paths use `@njit` compilation
- Structure-of-Arrays (SoA) layout: `x[N,3]` not `particles[N].x`
- Pre-allocate arrays, no dynamic resizing in loops
- Parallel loops with `prange` where possible

**Performance Gates:**

| Test | Requirement | Rationale |
|------|-------------|-----------|
| DSMC particle push (10⁶ particles) | <2 seconds | 10,000 timesteps → 20,000 s total acceptable |
| DSMC collisions (10⁵ pairs) | <1 second | Majorant method with fast RNG |
| PIC particle push (10⁵ particles) | <0.5 seconds | Boris algorithm well-vectorized |
| PIC Poisson solve (200 cells) | <0.1 seconds | Tridiagonal matrix, direct solve |
| **Full DSMC run** (10⁶ part, 10 ms) | **<60 min** | **Week 3 checkpoint** |
| **Full PIC run** (10⁵ part, 4 μs) | **<120 min** | **Week 9 checkpoint** |

**If gates not met:** Add Cython, consider C++ hot-path rewrite

### SPARTA/PICLas (Option 3)

**MPI Scaling Requirements:**
- Weak scaling: >80% efficiency up to 100 cores
- Strong scaling: >60% efficiency up to 50 cores
- Load balance: <10% imbalance across ranks

**Expected Performance:**
- DSMC: 10⁸ particles on 100 cores → 12 hours
- PIC: 10⁷ particles on 100 cores → 24 hours
- Coupled: 2-7 days for production run

---

## Validation Hierarchy

### Level 1: Unit Tests (30+ tests required)

**DSMC:**
- [ ] Ballistic motion (no collisions): trajectory conservation
- [ ] Thermal equilibration: approach to Maxwell-Boltzmann
- [ ] Viscosity recovery: Couette flow matches Chapman-Enskog
- [ ] Diffusion coefficient: concentration decay matches Fick's law

**PIC:**
- [ ] Particle pusher: energy conservation in static E field
- [ ] Charge deposition: sum(deposited) = sum(particles) * q
- [ ] Poisson solver: analytical test case (Gaussian charge)
- [ ] MCC: ionization threshold enforced (no ionization below E_threshold)

### Level 2: Integration Tests (10+ tests)

**DSMC-PIC Coupling:**
- [ ] Mass conservation: ∫(mdot_DSMC) = ∫(mdot_PIC) * utilization
- [ ] Convergence: coupling iteration converges in <10 steps
- [ ] Power balance: PIC P_in = P_out within 10%

### Level 3: Physics Benchmarks (5+ cases)

**DSMC:**
- [ ] Thermal transpiration: Δp/p = √(T_hot/T_cold) - 1 within 10%
- [ ] Poiseuille flow (transitional): Sharipov solution within 15%
- [ ] Molecular beam: cos(θ) angular distribution, R² > 0.95

**PIC:**
- [ ] Child-Langmuir sheath: thickness = 5λ_D within 20%
- [ ] Turner CCP benchmark (2013): n_e, φ within 20%
- [ ] Power balance: <10% error for all discharge conditions

### Level 4: Validation Cases (Target: Parodi et al. 2025)

**Intake Performance:**
- [ ] N₂ compression ratio: 400-550 (Parodi: 475)
- [ ] O₂ compression ratio: 70-110 (Parodi: 90)
- [ ] Temperature rise: 700-800 K (Parodi: ~750 K)

**Thruster Performance:**
- [ ] Plasma density: 1.3-2.0×10¹⁷ m⁻³ (Parodi: 1.65×10¹⁷)
- [ ] Electron temperature: 6-10 eV (Parodi: 7.8 eV)
- [ ] RF power absorbed: 18-22 W (Parodi: 20 W)
- [ ] Thrust: 300-700 μN (Parodi: 480 μN)

**System-Level:**
- [ ] Coupling converges: <10 iterations
- [ ] Neutral depletion: 10-20% density drop in thruster
- [ ] Ion energy distribution: Bi-modal structure observed

---

## Success Criteria

### Technical Metrics (Revised from Original Plan)

| Metric | Original Plan | **Corrected** | Rationale |
|--------|---------------|---------------|-----------|
| DSMC compression ratio | Within 20% | **Within 30%** | Stochastic + surface uncertainty |
| Plasma density | Within 20% | **Within 30%** | Chemistry simplifications acceptable |
| Electron temperature | Within 20% | **Within 20%** | Less sensitive to chemistry |
| **Power balance** | Not specified | **<10% error** | **CRITICAL VALIDATION** |
| **Debye resolution** | Not specified | **Δx ≤ 0.5 λ_D** | **Numerical stability** |
| Performance (DSMC) | 30 min | **<60 min** | Realistic with Numba |
| Performance (PIC) | 1 hour | **<120 min** | Realistic with Numba |
| **Coupling convergence** | Not specified | **<10 iterations** | **Feasibility check** |

### Programmatic Deliverables

**Option 2 (Python Prototype):**
- [ ] GitHub repository (public or ITAR-documented decision)
- [ ] 80% test coverage with CI/CD
- [ ] Documentation: theory + user guide + validation report
- [ ] At least 1 conference abstract submitted (IEPC 2026 or AIAA SciTech)
- [ ] SBIR proposal integration with preliminary results

**Option 3 (Production Tool):**
- [ ] All Option 2 deliverables plus:
- [ ] MPI-parallel code scaling to 100+ cores
- [ ] 3D geometry visualizations in ParaView
- [ ] Parametric study database (100+ runs)
- [ ] Journal paper submitted (Journal of Electric Propulsion)

### Business Impact

- [ ] Investor deck includes high-fidelity particle sim results
- [ ] SBIR Phase II proposal strengthened with validated predictions
- [ ] At least 2-3 design insights from parametric studies
- [ ] Team has in-house particle simulation expertise

---

## Risk Management

### Technical Risks

| Risk | Prob | Impact | Mitigation |
|------|------|--------|------------|
| 1D geometry too limiting | High | High | Add 2D PIC option, multi-channel intake model |
| Python performance insufficient | Med | High | **Numba mandatory, performance gates at Week 3** |
| Missing plasma-surface physics | High | High | SEE, CEX, catalysis implemented from start |
| One-way coupling inaccurate | High | Med | Neutral depletion feedback by Week 13 |
| PICLas can't handle ABEP physics | Med | Med | Thorough evaluation before commitment |
| Cross-section data inaccuracies | Med | Med | Pin LXCat version, sensitivity studies |
| Timeline slip (12→16 weeks) | High | Med | Weekly check-ins, ruthless scope control |
| Poor agreement with Parodi | Med | High | Start with exact reproduction of their case |

### Programmatic Risks

| Risk | Prob | Impact | Mitigation |
|------|------|--------|------------|
| ITAR classification delays | Low | High | Legal pre-decision meeting Week 1 |
| Team bandwidth limitations | High | High | Prioritize core capabilities, defer nice-to-haves |
| Option 3 funding not secured | Med | Med | Option 2 designed to be valuable standalone |
| PICLas learning curve too steep | Med | Med | Allocate 4-week evaluation before commitment |

---

## Decision Framework

### When to Choose Option 2 (Python Prototype)

✅ **YES if:**
- Need validated performance predictions for SBIR Phase I
- Want to publish methodology/validation paper
- Have 3-4 months of development bandwidth
- Analytical model uncertainties limit confidence
- Want to build in-house simulation expertise
- Budget constrained (<$100k)

❌ **NO if:**
- Analytical model sufficient for current needs
- No bandwidth (focus on hardware)
- Immediate flight demonstration is priority
- Commercial codes can meet requirements

### When to Choose Option 3a (SPARTA + Custom PIC)

✅ **YES if:**
- Option 2 successfully completed
- SBIR Phase II funding secured ($185k-$315k)
- PICLas evaluation shows critical limitations
- Need production-quality DSMC immediately
- Have HPC cluster access
- Can dedicate 2-3 FTE for 6-9 months

❌ **NO if:**
- PICLas can handle ABEP physics (choose 3b instead)
- Python optimization sufficient (extend Option 2)
- Timeline to flight doesn't allow 9-month dev

### When to Choose Option 3b (PICLas Integrated)

✅ **YES if:**
- Option 2 successfully completed
- PICLas evaluation confirms ABEP physics support
- Need native DSMC-PIC coupling
- Team has Fortran experience or willingness to learn
- Can save 3-6 months vs SPARTA+custom

❌ **NO if:**
- PICLas can't model effective RF heating
- PICLas surface models insufficient for catalysis
- Community/support too limited for comfort

---

## Tool Selection Recommendation

### Executive Summary

**Recommended Path:**

1. **Phase 1 (Months 1-4): Execute Option 2 Python Prototype**
   - Low-risk investment ($22k-$98k)
   - Validates physics models against Parodi et al.
   - Deliverable for SBIR Phase I
   - Learning experience builds expertise

2. **Phase 2 (Month 5): Evaluate PICLas First**
   - 4-6 week deep dive into capabilities
   - Test ABEP-specific physics (RF, catalysis, honeycomb)
   - Prototype intake geometry
   - Compare performance to Python

3. **Phase 3 Decision (Month 5 end):**
   - **If PICLas works** → Use PICLas (saves 3-6 months vs SPARTA+custom)
   - **If PICLas limited** → SPARTA + extend Python PIC to 3D
   - **If Python sufficient** → Optimize Python with Cython

### Why This Sequence?

**Option 2 First:**
- De-risks the entire effort (proves concept before big investment)
- Generates immediate value (SBIR deliverable in 4 months)
- Builds understanding needed to evaluate PICLas vs SPARTA
- Low cost relative to Option 3 ($22k vs $185k)

**PICLas Evaluation Second:**
- Could save 3-6 months vs building custom PIC
- Native coupling better than file I/O
- Only requires 4-6 week evaluation (low risk)
- If it works, massive time/cost savings

**SPARTA+Custom as Fallback:**
- Only pursue if PICLas can't handle ABEP physics
- Proven DSMC quality
- Larger community and documentation
- Can reuse Option 2 Python PIC with 3D extensions

---

## Getting Started Checklist

### Week 1 Actions

**Day 1: Environment Setup**
```bash
# Create development environment
conda create -n intakesim python=3.11
conda activate intakesim
pip install numpy scipy numba matplotlib pytest

# Create directory structure
mkdir -p IntakeSIM/{src,tests,examples,validation,docs}
```

**Day 2: Read Key References**
- [ ] Parodi et al. (2025) - Full paper, all sections
- [ ] Lieberman & Lichtenberg Ch. 11 - RF discharge theory
- [ ] Birdsall & Langdon Ch. 4 - PIC algorithms
- [ ] Bird (1994) Ch. 2 - DSMC collision methods

**Day 3: Legal & Repository Setup**
- [ ] ITAR classification meeting scheduled
- [ ] Decision: Public GitHub or private repo?
- [ ] Create repository: `AeriSat/IntakeSIM` or `AeriSat/aerisat-particle-sim`
- [ ] Add `.gitignore`, `LICENSE`, `README.md`

**Day 4: First Test Implementation**
- [ ] Implement ballistic particle motion with Numba
- [ ] Verify >50× speedup vs pure Python
- [ ] Write unit test for trajectory conservation
- [ ] Commit to repository

**Day 5: Team Kickoff Meeting**
- [ ] Review this document (claude.md)
- [ ] Review Technical Addendum corrections
- [ ] Assign roles (who codes what)
- [ ] Set up weekly check-ins (1 hour, technical discussion)
- [ ] Agree on communication channels (Slack, email, etc.)

### Week 2-3: DSMC Core Development

See detailed timeline in Implementation Options > Option 2 section

### Continuous Activities

**Weekly:**
- [ ] Technical check-in meeting (review progress, debug issues)
- [ ] Update progress.md with completed tasks
- [ ] Run test suite, maintain coverage
- [ ] Profile performance on large test cases

**Bi-weekly:**
- [ ] Literature review (any new ABEP papers?)
- [ ] Compare to analytical model (divergence analysis)

**Monthly:**
- [ ] Advisor review meeting (if engaged)
- [ ] Update budget tracking
- [ ] Assess timeline vs plan

---

## Publication Strategy

### Target Venues

**Priority 1: Journal of Electric Propulsion** (open access)
- **Title**: "Coupled Particle Simulation of Air-Breathing Electric Propulsion: Physics Modeling and Validation"
- **Content**: Full methodology including SEE, CEX, catalysis, neutral depletion coupling
- **Timeline**: Submit Q2 2026 (after Option 2 complete)
- **Impact**: Establishes AeriSat as thought leader in ABEP simulation

**Priority 2: Conference Presentations**
- **IEPC 2026** (International Electric Propulsion Conference)
  - Abstract: "Particle-Based Performance Prediction for CubeSat ABEP Systems"
  - Deadline: Typically 6 months before conference

- **AIAA SciTech 2026** (January)
  - Session: Electric Propulsion
  - Paper: "Neutral Depletion Effects in Air-Breathing Ion Thrusters"

**Priority 3 (if Option 3): Computer Physics Communications**
- **Title**: "IntakeSIM: An Open-Source Framework for ABEP System Simulation"
- **Content**: Software description, benchmarks, tutorials
- **Timeline**: Submit Q4 2026 if code publicly released

### Disclosure Protocol

**Before First Public Presentation:**
1. Legal review of ITAR classification
2. Sanitize mission-specific parameters
3. Use generic labels: "representative 3U CubeSat ABEP system"
4. No warfighter/interceptor language

**Boilerplate for Papers:**
> "Simulations performed using open-source tools with publicly available data. Geometry and operating conditions represent a generic CubeSat ABEP system and do not constitute export-controlled technical data."

---

## Contact and Collaboration

**Internal Team:**
- **CTO Office**: George Boyce (project lead)
- **Developer**: TBD (to be hired/assigned)

**Potential External Collaborators:**
- **KU Leuven (Lapenta group)**: PIC expertise, Parodi's institution
- **VKI (Magin group)**: ABEP modeling
- **MIT (Peraire/Kamm)**: DSMC methods
- **University of Michigan (Boyd group)**: Rarefied gas dynamics
- **University of Stuttgart (PICLas team)**: If pursuing Option 3b

**Consultant/Advisor:**
- Budget allocated for 0.1 FTE academic expert
- Weekly technical review meetings
- Validation strategy guidance
- Publication co-authorship

---

## Appendix: Quick Reference

### Key Equations

**DSMC:**
- Collision probability: `P = 1 - exp(-ν_max * Δt)`
- VHS cross-section: `σ = σ_ref * (v_ref / v_rel)^(2ω-1)`
- CLL reflection: Complex (see Bird 1994, Section 12.4)

**PIC:**
- Boris pusher: `v_plus = v_minus + (v_minus × t) × s` where `s = 2t/(1+t²)`
- Poisson equation: `∇²φ = -ρ/ε₀`
- Debye length: `λ_D = √(ε₀ kT_e / (n_e e²))`

**RF Heating:**
- Effective collision frequency: `ν_eff = 2P_abs / (n_e V m_e <v²>)`
- Power balance: `P_in = P_ion + P_electron + P_ionization + P_excitation`

### Physical Constants

```python
# Fundamental constants
e = 1.602e-19      # Elementary charge [C]
m_e = 9.109e-31    # Electron mass [kg]
eps0 = 8.854e-12   # Permittivity [F/m]
kB = 1.381e-23     # Boltzmann constant [J/K]
AMU = 1.661e-27    # Atomic mass unit [kg]

# Atmospheric species masses
m_O = 16 * AMU     # Oxygen atom
m_N2 = 28 * AMU    # Nitrogen molecule
m_O2 = 32 * AMU    # Oxygen molecule
m_NO = 30 * AMU    # Nitric oxide

# Ionization thresholds
E_ionize_N2 = 15.58  # [eV]
E_ionize_O = 13.62   # [eV]
E_ionize_O2 = 12.07  # [eV]
```

### File Structure Reference

```
IntakeSIM/
├── README.md                  # Project overview
├── claude.md                  # This file (AI assistant guide)
├── progress.md                # Timeline and decision log
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── docs/
│   ├── Quick_Reference_Summary (1).md
│   ├── ABEP_Particle_Simulation_Implementation_Plan (1).md
│   ├── ABEP_Particle_Simulation_Technical_Addendum.md
│   ├── theory.md              # Mathematical foundations (to be written)
│   ├── user_guide.md          # Installation and tutorials (to be written)
│   └── validation.md          # Test cases and results (to be written)
├── src/intakesim/
│   ├── __init__.py
│   ├── constants.py
│   ├── dsmc/
│   │   ├── mover.py
│   │   ├── collisions.py
│   │   └── surfaces.py
│   ├── pic/
│   │   ├── mover.py
│   │   ├── field_solver.py
│   │   ├── mcc.py
│   │   └── sources.py
│   └── diagnostics.py
├── tests/
│   ├── test_dsmc.py
│   ├── test_pic.py
│   └── test_integration.py
├── examples/
│   ├── 01_dsmc_intake.py
│   ├── 02_pic_discharge.py
│   └── 03_coupled_system.py
└── validation/
    ├── parodi_comparison.py
    └── analytical_comparison.py
```

---

## Final Remarks

This project transforms AeriSat from **"we have analytical models"** to **"we have validated particle simulations"** - a significant technical differentiator in the ABEP space.

**The path is clear:**
1. Start with Python prototype (low-risk, 4 months)
2. Evaluate PICLas vs SPARTA (1 month)
3. Scale to production tool if needed (6 months)

**The physics is challenging but achievable:**
- Multi-channel intake physics
- Complete VLEO chemistry
- Plasma-surface interactions
- Coupled neutral-plasma dynamics

**The payoff is substantial:**
- SBIR proposals strengthened
- Investor confidence increased
- Design optimization enabled
- Publications establishing thought leadership

**Remember the corrected principles:**
- Physics first (SEE, CEX, catalysis, coupling are mandatory)
- Performance matters (Numba from day 1)
- Validate constantly (power balance, benchmarks, Parodi comparison)
- Be pragmatic (leverage existing tools when appropriate)

---

**Questions?** Start by reading the full implementation plan, then review the technical addendum for physics corrections. Schedule kickoff meeting when ready to begin.

**Ready to start?** See "Getting Started Checklist" above and create your first particle class with Numba acceleration!

---

*IntakeSIM Project Guide - Created October 29, 2025*
*For AeriSat Systems CTO Office - Air-Breathing Electric Propulsion Development*
