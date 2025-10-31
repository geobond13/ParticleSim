# ABEP Particle Simulation - Quick Reference Guide

**Full Documentation:** `ABEP_Particle_Simulation_Implementation_Plan.md`

---

## TL;DR - What Are We Building?

**Option 2 (Core Goal):** Simplified 1D/2D particle simulations for ABEP
- **What:** DSMC for intake + PIC for plasma chamber
- **Why:** Validate analytical models, predict performance with higher fidelity
- **Timeline:** 8-12 weeks
- **Resources:** 1 developer, laptop, open-source tools
- **Cost:** $20k-80k depending on support level

**Option 3 (Stretch):** Full 3D production simulation capability
- **What:** Everything above + 3D geometry + MPI parallelization + coupling
- **Why:** Production-quality predictions, publications, design optimization
- **Timeline:** 6-12 months additional
- **Resources:** 2-3 developers, HPC cluster, $150k-300k
- **Decision:** Only pursue if Option 2 succeeds and resources available

---

## Week-by-Week Roadmap (Option 2)

### Phase 1: DSMC Core (Weeks 1-3)
**Week 1:** Particle motion, mesh, basic diagnostics
- ✅ Deliverable: Particles move ballistically, plots work

**Week 2:** Collision models (Variable Hard Sphere)
- ✅ Deliverable: Binary collisions conserve momentum/energy

**Week 3:** Wall interactions (CLL model)
- ✅ Deliverable: Thermal accommodation validated

### Phase 2: DSMC Intake (Weeks 4-5)
**Week 4:** Intake geometry, freestream injection
- ✅ Deliverable: Tapered duct with orbital velocity BC

**Week 5:** Parametric studies
- ✅ Deliverable: Compression ratio vs. altitude/accommodation

### Phase 3: PIC Core (Weeks 6-8)
**Week 6:** Poisson solver, particle pusher
- ✅ Deliverable: Child-Langmuir sheath reproduced

**Week 7:** Monte Carlo Collisions
- ✅ Deliverable: Elastic + ionization working

**Week 8:** RF discharge model
- ✅ Deliverable: Power coupling with feedback control

### Phase 4: PIC Thruster (Weeks 9-10)
**Week 9-10:** Ionization chamber with RF heating
- ✅ Deliverable: Plasma density and Te match Parodi

### Phase 5: Integration (Weeks 11-12)
**Week 11:** DSMC→PIC one-way coupling
- ✅ Deliverable: Full system simulation

**Week 12:** Documentation and validation report
- ✅ Deliverable: Published GitHub repo, comparison to Parodi

---

## Key Validation Targets (from Parodi et al.)

| Metric | Parodi Value | Acceptable Range | Why It Matters |
|--------|--------------|------------------|----------------|
| N₂ Compression Ratio | 475 | 400-550 | Validates intake physics |
| Plasma Density | 1.65×10¹⁷ m⁻³ | 1.3-2.0×10¹⁷ | Confirms ionization efficiency |
| Electron Temp | 7.8 eV | 6-10 eV | Energy balance check |
| RF Power Absorbed | 20 W | 18-22 W | Power coupling validation |
| Thrust | 480 μN | 400-600 μN | System performance metric |

---

## Critical Code Components

### DSMC Binary Collision (50 lines)
```python
def perform_collisions(cell, dt):
    # 1. Compute majorant collision frequency
    nu_max = n_gas * sigma_max * v_rel_max
    
    # 2. Select pairs stochastically
    N_attempts = int(0.5 * N_particles * nu_max * dt)
    
    # 3. Accept/reject based on actual cross section
    for _ in range(N_attempts):
        p1, p2 = random_pair()
        if random() < sigma(p1,p2) * v_rel / (sigma_max * v_rel_max):
            scatter_isotropic(p1, p2)
```

### PIC Particle Pusher (30 lines)
```python
def push_particle(p, E, B, dt):
    # Boris algorithm for E×B drift
    v_minus = p.v + 0.5 * (q/m) * E * dt
    
    t = 0.5 * (q/m) * B * dt
    s = 2*t / (1 + |t|²)
    v_plus = v_minus + (v_minus × t) × s
    
    p.v = v_plus + 0.5 * (q/m) * E * dt
    p.x += p.v * dt
```

### MCC Ionization (40 lines)
```python
def monte_carlo_collision(electron, neutrals, dt):
    P_coll = 1 - exp(-nu_max * dt)
    
    if random() < P_coll:
        E = 0.5 * m * v² / e
        
        if random() < sigma_ionize(E) / sigma_total(E):
            if E > E_ionization:
                create_ion(electron.x)
                create_secondary_electron()
```

---

## File Structure

```
aerisat-particle-sim/
├── src/aerisat_psim/
│   ├── dsmc/          # 500 lines total
│   │   ├── mover.py          # 100 lines
│   │   ├── collisions.py     # 200 lines
│   │   └── surfaces.py       # 100 lines
│   ├── pic/           # 700 lines total
│   │   ├── mover.py          # 150 lines
│   │   ├── field_solver.py   # 200 lines
│   │   ├── mcc.py            # 250 lines
│   │   └── sources.py        # 100 lines
│   └── diagnostics.py # 300 lines
├── tests/             # 500 lines
├── examples/          # 300 lines
└── docs/              # 15,000 words

TOTAL CODE: ~2,500 lines (very manageable!)
```

---

## Testing Strategy

### Unit Tests (30+ tests)
- ✅ Ballistic motion (no collisions)
- ✅ Thermal equilibration (with collisions)
- ✅ Viscosity recovery (continuum limit)
- ✅ Plasma sheath structure
- ✅ RF power absorption

### Integration Tests (10+ tests)
- ✅ Intake compression ratio
- ✅ Thruster plasma density
- ✅ One-way coupling mass conservation

### Validation (5+ cases)
- ✅ Parodi et al. intake (CR ~ 475)
- ✅ Parodi et al. thruster (n ~ 1.65e17)
- ✅ AeriSat analytical model comparison
- ✅ Global model cross-check
- ✅ Literature shock tube data

---

## Resource Requirements

### Option 2 Minimal
- **Personnel:** 1 developer @ 0.5 FTE for 3 months
- **Hardware:** Existing laptop (8 cores, 16 GB RAM)
- **Software:** All open-source (Python, NumPy, SciPy)
- **Budget:** $20k-40k

### Option 2 with Support
- **Personnel:** 1 senior developer + 1 advisor
- **Hardware:** Workstation + cloud compute for sweeps
- **Budget:** $70k-100k

### Option 3 Full (only if pursuing)
- **Personnel:** 2-3 FTE for 12 months
- **Hardware:** HPC cluster (1000 cores)
- **Budget:** $180k-320k

---

## Decision Points

### Week 4: DSMC Checkpoint
**Question:** Does DSMC reproduce analytical limits?
- ✅ YES → Continue to intake application
- ❌ NO → Debug 1 week, reassess

### Week 8: PIC Checkpoint
**Question:** Does PIC match global model within 50%?
- ✅ YES → Continue to validation
- ❌ NO → Simplify model or extend timeline

### Week 12: Final Decision
**Question:** Is validation report complete?
- ✅ YES → Success! Consider Option 3
- ❌ NO → Iterate 2 more weeks, then conclude

### Month 6: Option 3 Decision
**Question:** Should we pursue full 3D?
- ✅ YES if: Secured funding, need high-fidelity 3D, have team
- ❌ NO if: Option 2 sufficient, focus on hardware

---

## Success Metrics

### Technical
- [ ] 80% test coverage
- [ ] All Parodi metrics within 30%
- [ ] Parametric sweep (10 cases) in <24 hours
- [ ] Code runs on laptop without crashes

### Programmatic
- [ ] Documentation complete (theory + user guide + API)
- [ ] GitHub repository public (or documented ITAR decision)
- [ ] Validation report published internally
- [ ] At least 1 conference abstract submitted

### Business Impact
- [ ] SBIR proposal includes particle sim results
- [ ] Investor deck shows high-fidelity predictions
- [ ] Team has validated modeling capability
- [ ] 2-3 design insights from parametric studies

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Algorithm bugs | Extensive unit tests, reproduce known results |
| Numerical instability | Implicit methods, adaptive timestep |
| Poor Parodi agreement | Start with exact reproduction of their case |
| Timeline slip | Weekly check-ins, early problem detection |
| Bandwidth | Prioritize ruthlessly, hire contractor if needed |

---

## Publication Strategy

### Immediate (2026)
1. **IEPC Conference** - "Particle Simulation of ABEP for CubeSats"
2. **AIAA SciTech** - "DSMC-PIC Validation for ABEP"
3. **Journal of Electric Propulsion** - Full methodology paper

### If Option 3 Pursued
4. **Computer Physics Communications** - Software description
5. **Journal of Spacecraft and Rockets** - Mission analysis with 3D sims

---

## Getting Started Checklist

### This Week
- [ ] Read Parodi paper thoroughly (all team members)
- [ ] Review existing `aerisat-abep-model.py` code
- [ ] Create GitHub repository
- [ ] Set up Python environment (venv, numpy, scipy, pytest)
- [ ] Implement first test case (ballistic motion)

### Next Week
- [ ] Weekly meeting scheduled (1 hour, technical discussion)
- [ ] Development plan finalized
- [ ] First pull request merged (particle class + unit tests)
- [ ] Advisor identified (academic or internal)

### Month 1 Goal
- [ ] DSMC core working
- [ ] At least 5 passing unit tests
- [ ] Preliminary validation against free molecular flow

---

## Key Contacts and Resources

### Software
- **SPARTA DSMC:** https://sparta.github.io
- **LXCat Database:** https://lxcat.net
- **ParaView:** https://www.paraview.org

### Papers (in `/mnt/project/`)
- Parodi et al. (2025) - This uploaded paper!
- Andreussi et al. (2022) - `Andreussiet_al_2022.pdf`
- Goebel EP Fundamentals - `Goebel__EP_fund.pdf`

### Potential Collaborators
- **KU Leuven (Lapenta)** - PIC expertise, Parodi's institution
- **VKI (Magin)** - ABEP modeling
- **MIT (Boyd)** - DSMC methods

---

## Frequently Asked Questions

**Q: Can I do this with Claude/AI assistance?**
A: Absolutely! The implementation plan includes detailed algorithms that Claude can help code. Claude Code (in terminal) would be excellent for iterative development.

**Q: Do I need to know plasma physics deeply?**
A: Helpful but not required. The algorithms are well-documented. Focus on implementing, testing, and validating rather than deriving from first principles.

**Q: What if I can't reproduce Parodi's results?**
A: Start by reproducing their exact setup (geometry, BC, parameters). Then systematically vary one thing at a time. Contact them directly—academics usually help!

**Q: Should I use C++ instead of Python?**
A: Python is fine for Option 2. Only consider C++ for Option 3 if performance becomes limiting. Profile first!

**Q: What about ITAR restrictions?**
A: Document your ITAR classification decision. If restricted, keep code private. If not, open-sourcing builds credibility and community.

**Q: Can I hire someone to do this?**
A: Yes! A plasma physics PhD student or postdoc could implement Option 2 in 3 months for $30k-50k. Be clear on deliverables and testing requirements.

**Q: Is Option 3 really necessary?**
A: Only if Option 2 shows you're limited by 1D assumptions. Most SBIR work can succeed with Option 2 + analytical models.

---

## Bottom Line

✅ **Option 2 is achievable in 3 months with focused effort**

✅ **It will significantly strengthen AeriSat's technical credibility**

✅ **Start small, validate constantly, and build incrementally**

✅ **Option 3 is for later if you secure Phase II funding**

✅ **Either way, you'll have particle simulation expertise in-house**

---

**Questions? Start by reading the full plan, then let's discuss implementation strategy!**

---

*Quick Reference Guide for AeriSat ABEP Particle Simulation Project*
*Full details in: ABEP_Particle_Simulation_Implementation_Plan.md*
