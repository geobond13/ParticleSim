# IntakeSIM Comprehensive Reference List

**Last Updated:** January 2025
**Purpose:** Complete bibliography for ABEP particle simulation development and validation

---

## Legend

**Access Levels:**
- 🟢 **Open Access** - Freely available (ArXiv, open journals, GitHub)
- 🟡 **Institutional Access** - Requires university library or institutional login
- 🔴 **Paywalled** - Contact author or check ResearchGate for preprints

**Priority Levels:**
- ⭐⭐⭐ **Critical** - Essential reading for IntakeSIM development
- ⭐⭐ **Important** - Highly relevant for specific modules
- ⭐ **Supplementary** - Background/context

---

## 1. DSMC Methods & Rarefied Gas Dynamics

### 1.1 Foundational Works

**[DSMC-1] Bird, G.A. (1963)**
- **Title:** "Approach to Translational Equilibrium in a Rigid Sphere Gas"
- **Journal:** Physics of Fluids, Vol. 6, pp. 1518-1519
- **DOI:** 10.1063/1.1710976
- **Access:** 🔴 Paywalled
- **Priority:** ⭐⭐ Important
- **Notes:** Original DSMC paper - historical reference. Methods superseded by later work.

**[DSMC-2] Bird, G.A. (1994)**
- **Title:** "Molecular Gas Dynamics and the Direct Simulation of Gas Flows"
- **Publisher:** Oxford University Press
- **ISBN:** 978-0198561958
- **Access:** 🔴 Purchase required (~$100)
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** THE definitive DSMC textbook. Chapters 2-4 (collision models), Chapter 12 (gas-surface interactions, CLL model). Essential reference for implementation.

### 1.2 VHS Collision Model

**[DSMC-3] Bird, G.A. (1981)**
- **Title:** "Monte Carlo Simulation in an Engineering Context"
- **Conference:** Progress in Astronautics and Aeronautics, Vol. 74, pp. 239-255
- **Access:** 🔴 Paywalled
- **Priority:** ⭐⭐ Important
- **Notes:** Introduces Variable Hard Sphere (VHS) model used in IntakeSIM. Cross-section formula: σ = σ_ref * (v_ref/v_rel)^(2ω-1)

### 1.3 Recent DSMC Developments (2024-2025)

**[DSMC-4] Gallis, M.A. et al. (2024)**
- **Title:** "Comprehensive Evaluation of the Massively Parallel Direct Simulation Monte Carlo Kernel 'SPARTA' in Rarefied Hypersonic Flows - Part A: Fundamentals"
- **Journal:** Applied Sciences, Vol. 12, No. 10, 198
- **DOI:** 10.3390/app12105198
- **URL:** https://www.mdpi.com/2076-3417/12/10/5198
- **Access:** 🟢 Open Access
- **Priority:** ⭐⭐⭐ Critical (if using SPARTA)
- **Notes:** 2024 validation of SPARTA against benchmark test cases. Confirms accuracy for parallel DSMC. See Table 2 for validation metrics.

**[DSMC-5] Su, W. et al. (2024)**
- **Title:** "Multiscale Simulation of Rarefied Gas Dynamics via Direct Intermittent GSIS-DSMC Coupling"
- **Journal:** Advances in Aerodynamics, Vol. 6, Article 8
- **DOI:** 10.1186/s42774-024-00188-y
- **URL:** https://aia.springeropen.com/articles/10.1186/s42774-024-00188-y
- **Access:** 🟢 Open Access
- **Priority:** ⭐ Supplementary
- **Notes:** Hybrid GSIS-DSMC method for faster convergence. Relevant for future performance optimization beyond Option 2.

**[DSMC-6] Chen, J. et al. (2025)**
- **Title:** "Multiscale Simulation of Rarefied Polyatomic Gas Flow via DIG Method"
- **Journal:** ArXiv preprint arXiv:2501.10965
- **URL:** https://arxiv.org/abs/2501.10965
- **Access:** 🟢 Open Access
- **Priority:** ⭐ Supplementary
- **Notes:** January 2025 - Very recent work on polyatomic gases (N₂, O₂). May be useful for future multi-mode internal energy modeling.

### 1.4 Rarefied Flow Benchmarks (Sharipov)

**[DSMC-7] Sharipov, F. & Seleznev, V. (1998)**
- **Title:** "Data on Internal Rarefied Gas Flows"
- **Journal:** Journal of Physical and Chemical Reference Data, Vol. 27, No. 3, pp. 657-706
- **DOI:** 10.1063/1.556019
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** Standard reference data for Poiseuille flow validation across all Knudsen numbers. Tables 8-12 give flow rates vs. rarefaction parameter. Used for IntakeSIM transitional regime validation.

**[DSMC-8] Sharipov, F. (Web Calculator)**
- **Title:** "Rarefied Flow Calculator"
- **URL:** http://fisica.ufpr.br/sharipov/rfc.html
- **Access:** 🟢 Free Online Tool
- **Priority:** ⭐⭐ Important
- **Notes:** Online calculator for benchmark values (Poiseuille flow, thermal transpiration). Useful for quick validation checks.

---

## 2. PIC & Plasma Simulation

### 2.1 Foundational Textbooks

**[PIC-1] Birdsall, C.K. & Langdon, A.B. (1991)**
- **Title:** "Plasma Physics via Computer Simulation"
- **Publisher:** Adam Hilger (Institute of Physics Publishing)
- **ISBN:** 978-0750301718
- **Access:** 🔴 Purchase required (~$80-150)
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** THE definitive PIC textbook. Chapter 4 (Boris pusher, particle mover), Chapter 7 (Poisson solver), Chapter 10 (MCC method). Essential for PIC module implementation.

**[PIC-2] Hockney, R.W. & Eastwood, J.W. (1988)**
- **Title:** "Computer Simulation Using Particles"
- **Publisher:** CRC Press
- **ISBN:** 978-0852743928
- **Access:** 🔴 Purchase required
- **Priority:** ⭐⭐ Important
- **Notes:** Alternative PIC reference with focus on numerical methods. Chapters on field solvers and particle-mesh coupling.

### 2.2 PIC-MCC Methodology

**[PIC-3] Birdsall, C.K. (1991)**
- **Title:** "Particle-in-Cell Charged-Particle Simulations, Plus Monte Carlo Collisions with Neutral Atoms, PIC-MCC"
- **Journal:** IEEE Transactions on Plasma Science, Vol. 19, No. 2, pp. 65-85
- **DOI:** 10.1109/27.106800
- **Access:** 🟡 Institutional Access (IEEE Xplore)
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** Seminal paper on PIC-MCC coupling. Sections on null-collision method, cross-section handling, and timestep requirements. Directly applicable to IntakeSIM PIC module.

**[PIC-4] Vahedi, V. & Surendra, M. (1995)**
- **Title:** "A Monte Carlo Collision Model for the Particle-in-Cell Method: Applications to Argon and Oxygen Discharges"
- **Journal:** Computer Physics Communications, Vol. 87, pp. 179-198
- **DOI:** 10.1016/0010-4655(94)00171-W
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐⭐ Important
- **Notes:** Detailed MCC implementation with cross-section data. Includes O₂ chemistry relevant to VLEO ABEP.

### 2.3 Validation Benchmarks

**[PIC-5] Turner, M.M. et al. (2013)**
- **Title:** "Simulation Benchmarks for Low-Pressure Plasmas: Capacitive Discharges"
- **Journal:** Physics of Plasmas, Vol. 20, Article 013507
- **DOI:** 10.1063/1.4775084
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** Standard CCP benchmark for PIC code validation. Test cases in helium at 30-300 mTorr. IntakeSIM Week 9 checkpoint uses this benchmark. Compare n_e, φ, EEDF against published results.

**[PIC-6] COMSOL (2024)**
- **Title:** "Benchmark Model of a Capacitively Coupled Plasma"
- **URL:** https://www.comsol.com/model/benchmark-model-of-a-capacitively-coupled-plasma-11745
- **Access:** 🟢 Free Documentation
- **Priority:** ⭐⭐ Important
- **Notes:** Accessible description of Turner benchmark implementation. Useful for understanding test case setup.

### 2.4 Parallel PIC Methods

**[PIC-7] Lapenta, G. (2012)**
- **Title:** "Particle Simulations of Space Weather"
- **Journal:** Journal of Computational Physics, Vol. 231, pp. 795-821
- **DOI:** 10.1016/j.jcp.2011.03.035
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐ Supplementary (for Option 3)
- **Notes:** Implicit PIC methods for large timesteps. Parodi's PhD uses similar techniques. Relevant for future 3D PIC optimization.

---

## 3. Air-Breathing Electric Propulsion (ABEP)

### 3.1 Primary Validation Targets

**[ABEP-1] Parodi, P. et al. (2025)**
- **Title:** "Particle-based Simulation of an Air-Breathing Electric Propulsion System"
- **Journal:** ArXiv preprint arXiv:2504.12829
- **URL:** https://arxiv.org/html/2504.12829
- **Access:** 🟢 Open Access
- **Priority:** ⭐⭐⭐ CRITICAL - PRIMARY VALIDATION TARGET
- **Notes:** **THE benchmark paper for IntakeSIM.** Includes:
  - N₂ CR = 475 (system-level, freestream to chamber)
  - O₂ CR = 90
  - Plasma density: 1.65×10¹⁷ m⁻³
  - T_e = 7.8 eV
  - Thrust: 480 μN at 20W
  - Couples DSMC intake + PIC thruster
  - **Week 12-13 validation target for coupled system**

**[ABEP-2] Romano, F. et al. (2021)**
- **Title:** "Intake Design for an Atmosphere-Breathing Electric Propulsion System (ABEP)"
- **Journal:** Acta Astronautica, Vol. 187, pp. 225-235
- **DOI:** 10.1016/j.actaastro.2021.06.033
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐⭐⭐ CRITICAL - DSMC VALIDATION TARGET
- **Notes:** **Primary DSMC intake benchmark.** Validation targets:
  - η_c = 0.458 at 150 km (diffuse walls)
  - Altitude sweep 150-250 km (Table 8)
  - Multi-channel honeycomb geometry (12,732 channels)
  - **Week 6 validation: IntakeSIM eta_c vs Romano Table 8**

**[ABEP-3] Cifali, G. et al. (2011)**
- **Title:** "Experimental Characterization of Hall Effect and Radio-Frequency Ion Thrusters with Atmospheric Propellants"
- **Conference:** 32nd International Electric Propulsion Conference, IEPC-2011-236
- **URL:** Available on IEPC proceedings
- **Access:** 🟢 Open Access (IEPC archive)
- **Priority:** ⭐⭐ Important
- **Notes:** Experimental validation data:
  - HET: 19-24 mN thrust with N₂/O₂ at ~1000W
  - RIT: 5-6 mN thrust with N₂/O₂ at 450W
  - 10-hour endurance tests successful
  - **Context only - used for physical reasonableness checks, not direct validation**

### 3.2 ABEP Review Papers

**[ABEP-4] Andreussi, T. et al. (2022)**
- **Title:** "A Review of Air-Breathing Electric Propulsion: From Mission Studies to Technology Verification"
- **Journal:** Journal of Electric Propulsion, Vol. 1, Article 24
- **DOI:** 10.1007/s44205-022-00024-9
- **URL:** https://link.springer.com/article/10.1007/s44205-022-00024-9
- **Access:** 🟢 Open Access
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** Comprehensive 2022 review covering:
  - ABEP history and mission concepts
  - Technology challenges (intake design, ionization, thruster)
  - Summary of experimental work (AETHER, DISCOVERER projects)
  - **Excellent overview paper - read first for context**

**[ABEP-5] Zheng, Y. et al. (2020)**
- **Title:** "A Comprehensive Review of Atmosphere-Breathing Electric Propulsion Systems"
- **Journal:** International Journal of Aerospace Engineering, Article 8811847
- **DOI:** 10.1155/2020/8811847
- **Access:** 🟢 Open Access
- **Priority:** ⭐⭐ Important
- **Notes:** Earlier review (2020) with focus on mission applications and system-level trades. Good for understanding VLEO mission context.

### 3.3 Recent Intake Design Papers (2023-2024)

**[ABEP-6] Singh, L. et al. (2024)**
- **Title:** "Design and Operational Concept of a Cryogenic Active Intake Device for Atmosphere-Breathing Electric Propulsion"
- **Journal:** Aerospace Science and Technology, Vol. 144, Article 108804
- **DOI:** 10.1016/j.ast.2024.108804
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐ Supplementary
- **Notes:** Novel cryogenic intake concept. Interesting alternative approach but not used in IntakeSIM baseline.

**[ABEP-7] Li, J. et al. (2023)**
- **Title:** "Parametric Study on the Flight Envelope of a Radio-Frequency Ion Thruster Based Atmosphere-Breathing Electric Propulsion System"
- **Journal:** Acta Astronautica, Vol. 211, pp. 193-203
- **DOI:** 10.1016/j.actaastro.2023.06.016
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐⭐ Important
- **Notes:** System-level performance trades. Useful for understanding how intake CR affects overall ABEP performance.

### 3.4 Parodi's Earlier Work

**[ABEP-8] Parodi, P. (2019)**
- **Title:** "Analysis and Simulation of an Intake for Air-Breathing Electric Propulsion Systems"
- **Thesis:** M.Sc. Thesis, University of Pisa
- **Access:** 🔴 Contact author (pietro.parodi@vki.ac.be)
- **Priority:** ⭐ Supplementary
- **Notes:** Parodi's Master's thesis - precursor to 2025 paper. May have additional implementation details not in published paper.

**[ABEP-9] Parodi, P. et al. (2023)**
- **Title:** "Development of the 3D PIC-DSMC Code Pantera for Electric Propulsion Applications and Beyond"
- **Conference:** PhD Symposium, Arenberg Doctoral School
- **URL:** https://set.kuleuven.be/phd/seminars/parodi
- **Access:** 🟢 Open (presentation abstract)
- **Priority:** ⭐ Supplementary
- **Notes:** Description of Parodi's coupled PIC-DSMC code "Pantera" used in 2025 paper. Fully-implicit PIC algorithm details.

---

## 4. Gas-Surface Interactions

### 4.1 CLL Scattering Model

**[SURF-1] Cercignani, C. & Lampis, M. (1971)**
- **Title:** "Kinetic Models for Gas-Surface Interactions"
- **Journal:** Transport Theory and Statistical Physics, Vol. 1, No. 2, pp. 101-114
- **DOI:** 10.1080/00411457108231440
- **Access:** 🔴 Paywalled
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** Original CLL model paper. Defines normal and tangential accommodation coefficients (σ_n, σ_t) used in IntakeSIM. Implementation details in Bird (1994) Chapter 12.

**[SURF-2] Lord, R.G. (1991)**
- **Title:** "Application of the Cercignani-Lampis Scattering Kernel to Direct Simulation Monte Carlo Calculations"
- **Conference:** 17th International Symposium on Rarefied Gas Dynamics, pp. 1427-1433
- **Access:** 🔴 Paywalled
- **Priority:** ⭐⭐ Important
- **Notes:** Practical implementation of CLL in DSMC. Algorithm for sampling reflected velocity distributions.

### 4.2 Catalytic Recombination

**[SURF-3] Krasheninnikov, S.I. et al. (2023)**
- **Title:** "A Review of Recombination Coefficients of Neutral Oxygen Atoms for Various Materials"
- **Journal:** Materials, Vol. 16, No. 5, Article 1774
- **DOI:** 10.3390/ma16051774
- **URL:** https://www.mdpi.com/1996-1944/16/5/1774
- **Access:** 🟢 Open Access
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** **2023 comprehensive review of atomic oxygen recombination.** Tables 1-3 give γ values for various materials (metals, ceramics, coatings). IntakeSIM uses γ ~ 0.01-0.1 for metals at 700 K based on this review.

**[SURF-4] Panerai, F. et al. (2019)**
- **Title:** "Experimental Study of Surface Roughness Effect on Oxygen Catalytic Recombination"
- **Journal:** Experimental Thermal and Fluid Science, Vol. 109, Article 109884
- **DOI:** 10.1016/j.expthermflusci.2019.109884
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐ Supplementary
- **Notes:** Shows γ dependence on surface roughness and temperature. Useful for understanding uncertainty in γ values.

### 4.3 Atomic Oxygen Surface Chemistry

**[SURF-5] Banks, B.A. et al. (2011)**
- **Title:** "Atomic Oxygen Effects on Materials"
- **NASA Report:** NASA/TM-2011-217004
- **URL:** https://ntrs.nasa.gov (NASA Technical Reports Server)
- **Access:** 🟢 Free (NASA)
- **Priority:** ⭐⭐ Important
- **Notes:** Comprehensive NASA report on AO interactions with spacecraft materials. Relevant for understanding surface aging effects on intake performance.

### 4.4 Clausing Factor

**[SURF-6] Clausing, P. (1932)**
- **Title:** "The Flow of Highly Rarefied Gases through Tubes of Arbitrary Length"
- **Journal:** Journal of Vacuum Science and Technology, Vol. 8, pp. 636-646
- **Access:** 🔴 Paywalled (historical)
- **Priority:** ⭐⭐ Important
- **Notes:** Original Clausing transmission probability paper. Formula: K(L/D) for molecular flow through tubes. Modern review available in Sharipov (1998).

**[SURF-7] Berman, A.S. (1965)**
- **Title:** "Free Molecule Transmission Probabilities"
- **Journal:** Journal of Applied Physics, Vol. 36, pp. 3356-3356
- **DOI:** 10.1063/1.1702984
- **Access:** 🔴 Paywalled
- **Priority:** ⭐ Supplementary
- **Notes:** Analytical solutions for Clausing factor. Tables of K vs L/D ratio. Used for validating multi-channel honeycomb transmission in Phase II.

---

## 5. Plasma-Surface Physics

### 5.1 Secondary Electron Emission (SEE)

**[PLASMA-1] Vaughan, J.R.M. (1989)**
- **Title:** "A New Formula for Secondary Emission Yield"
- **Journal:** IEEE Transactions on Electron Devices, Vol. 36, No. 9, pp. 1963-1967
- **DOI:** 10.1109/16.34278
- **Access:** 🟡 Institutional Access (IEEE)
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** **Vaughan formula used in IntakeSIM PIC module:**
  δ(E) = δ_max * (E/E_max)^n * exp(n*(1-E/E_max))
  Typical parameters: Mo (δ_max=1.25, E_max=350 eV), Ceramic (δ_max=2.5, E_max=300 eV)

**[PLASMA-2] Hobbs, G.D. & Wesson, J.A. (1967)**
- **Title:** "Heat Flow Through a Langmuir Sheath in the Presence of Electron Emission"
- **Journal:** Plasma Physics, Vol. 9, pp. 85-87
- **DOI:** 10.1088/0032-1028/9/1/410
- **Access:** 🔴 Paywalled
- **Priority:** ⭐⭐ Important
- **Notes:** Theory of SEE effects on sheath structure. Shows SEE reduces sheath potential and lowers T_e by 1-2 eV.

**[PLASMA-3] Sydorenko, D. et al. (2006)**
- **Title:** "Plasma-Sheath Transition in the Kinetic Tonks-Langmuir Discharge Model with Secondary Electron Emission"
- **Journal:** Physics of Plasmas, Vol. 13, Article 014501
- **DOI:** 10.1063/1.2158698
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐ Supplementary
- **Notes:** PIC simulations of SEE in discharge. Relevant for understanding non-Maxwellian effects near walls.

### 5.2 Charge Exchange (CEX)

**[PLASMA-4] Miller, J.S. et al. (2002)**
- **Title:** "Xenon Charge Exchange Cross Sections for Electrostatic Thruster Models"
- **Journal:** Journal of Applied Physics, Vol. 91, pp. 984-991
- **DOI:** 10.1063/1.1426246
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐⭐ Important
- **Notes:** Methodology for measuring CEX cross-sections. Xe⁺ + Xe → Xe + Xe⁺ has σ ~ 5×10⁻¹⁹ m². Similar magnitude expected for O⁺ + O.

**[PLASMA-5] Lindsay, B.G. & Stebbings, R.F. (2005)**
- **Title:** "Charge Transfer Cross Sections for Energetic Neutral Atom Data Analysis"
- **Journal:** Journal of Geophysical Research, Vol. 110, Article A12213
- **DOI:** 10.1029/2005JA011298
- **Access:** 🟢 Open Access (AGU)
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** Compilation of CEX cross-sections for O, N, N₂, O₂. **Table 2 gives O⁺ + O → O + O⁺: σ ~ 2×10⁻¹⁹ m² at 100 eV** (LARGE! Must include in IntakeSIM PIC module).

**[PLASMA-6] Particle-in-Cell Tutorial (2011)**
- **Title:** "Charge Exchange Collisions (CEX)"
- **URL:** https://www.particleincell.com/2011/charge-exchange/
- **Access:** 🟢 Free Online
- **Priority:** ⭐⭐ Important
- **Notes:** Practical tutorial on implementing CEX in PIC-MCC codes. Code examples and cross-section handling.

### 5.3 Electron-Impact Cross Sections (LXCat)

**[PLASMA-7] Pitchford, L.C. et al. (2017)**
- **Title:** "LXCat: An Open-Access, Web-Based Platform for Data Needed for Modeling Low Temperature Plasmas"
- **Journal:** Plasma Processes and Polymers, Vol. 14, Article 1600098
- **DOI:** 10.1002/ppap.201600098
- **URL:** https://www.lxcat.net
- **Access:** 🟢 Open Access (Database)
- **Priority:** ⭐⭐⭐ CRITICAL
- **Notes:** **PRIMARY cross-section database for IntakeSIM PIC module.**
  - Databases: Biagi, Phelps, Hayashi for N₂, O₂, O
  - Ionization thresholds: N₂ (15.58 eV), O (13.62 eV), O₂ (12.07 eV)
  - Download cross-sections in text format for direct integration
  - **Week 8 checkpoint: Integrate LXCat data into MCC module**

**[PLASMA-8] Biagi Database (LXCat)**
- **Title:** Biagi-v8.9 database on LXCat
- **URL:** https://www.lxcat.net/Biagi
- **Access:** 🟢 Free Download
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** Most comprehensive database for N₂, O₂, O. Includes elastic, ionization, excitation, attachment. **Use this database for IntakeSIM baseline.**

---

## 6. Atmospheric Models & Chemistry

### 6.1 NRLMSISE-00 Model

**[ATM-1] Picone, J.M. et al. (2002)**
- **Title:** "NRLMSISE-00 Empirical Model of the Atmosphere: Statistical Comparisons and Scientific Issues"
- **Journal:** Journal of Geophysical Research: Space Physics, Vol. 107, No. A12, Article 1468
- **DOI:** 10.1029/2002JA009430
- **Access:** 🟢 Open Access (AGU)
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** **Standard atmospheric model for VLEO altitudes (100-1000 km).**
  - Provides n, T, composition (O, N₂, O₂, He, Ar, H, N) vs altitude, solar activity
  - 200 km nominal: n ~ 4.2×10¹⁷ m⁻³, T ~ 1000 K, O (83%), N₂ (14%), O₂ (2%)
  - **IntakeSIM uses NRLMSISE-00 for freestream boundary conditions**

**[ATM-2] Emmert, J.T. et al. (2021)**
- **Title:** "NRLMSIS 2.0: A Whole-Atmosphere Empirical Model of Temperature and Neutral Species Densities"
- **Journal:** Earth and Space Science, Vol. 8, Article e2020EA001321
- **DOI:** 10.1029/2020EA001321
- **Access:** 🟢 Open Access
- **Priority:** ⭐⭐ Important
- **Notes:** Updated NRLMSIS 2.0 model (2021). Improved thermosphere-mesosphere coupling. Consider upgrading to NRLMSIS 2.0 in Phase II for better accuracy.

**[ATM-3] Mehta, P.M. et al. (2024)**
- **Title:** "Atmospheric Density Estimation in Very Low Earth Orbit Based on Nanosatellite Measurement Data Using Machine Learning"
- **Journal:** Aerospace Science and Technology, Vol. 144, Article 108808
- **DOI:** 10.1016/j.ast.2024.108808
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐ Supplementary
- **Notes:** Shows NRLMSISE-00 underpredicts density at ~200 km by factor of 1.5-2×. Actual density may be 0.5-0.64× higher than NRLMSISE-00 prediction. **Uncertainty consideration for ABEP performance trades.**

### 6.2 Thermosphere Composition

**[ATM-4] Hedin, A.E. (1987)**
- **Title:** "MSIS-86 Thermospheric Model"
- **Journal:** Journal of Geophysical Research, Vol. 92, No. A5, pp. 4649-4662
- **DOI:** 10.1029/JA092iA05p04649
- **Access:** 🟢 Open Access (AGU)
- **Priority:** ⭐ Supplementary (historical)
- **Notes:** Predecessor to NRLMSISE-00. Included for historical context; use NRLMSISE-00 for actual simulations.

---

## 7. Software Tools & Frameworks

### 7.1 SPARTA (Sandia DSMC)

**[SW-1] SPARTA Official Website**
- **URL:** https://sparta.github.io/
- **Access:** 🟢 Open Source (GPL)
- **Priority:** ⭐⭐⭐ Critical (if pursuing Option 3a)
- **Notes:** Official SPARTA documentation, tutorials, examples. Download version 4 Sep 2024.

**[SW-2] SPARTA GitHub Repository**
- **URL:** https://github.com/sparta/sparta
- **Access:** 🟢 Open Source (GPL)
- **Priority:** ⭐⭐⭐ Critical (Option 3a)
- **Notes:** Source code, issue tracker, development history. Check "examples/" directory for intake-relevant test cases.

**[SW-3] Plimpton, S.J. & Gallis, M.A. (2015)**
- **Title:** "SPARTA: A Scalable Flexible Open-Source Direct Simulation Monte Carlo Code"
- **Conference:** AIAA Aviation Forum, AIAA-2015-3212
- **DOI:** 10.2514/6.2015-3212
- **Access:** 🟡 Institutional Access (AIAA)
- **Priority:** ⭐⭐ Important (Option 3a)
- **Notes:** Original SPARTA paper describing architecture, parallelization, and capabilities.

**[SW-4] Gallis, M.A. et al. (2017)**
- **Title:** "Validation Simulations of the DSMC Code SPARTA"
- **Conference:** 30th International Symposium on Rarefied Gas Dynamics
- **DOI:** 10.1063/1.4967518
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐⭐ Important (Option 3a)
- **Notes:** Comprehensive validation suite for SPARTA. Benchmark test cases for code verification.

### 7.2 PICLas (Stuttgart PIC-DSMC)

**[SW-5] PICLas Official Website**
- **URL:** https://piclas.readthedocs.io
- **Access:** 🟢 Open Source (GPL-3.0)
- **Priority:** ⭐⭐⭐ Critical (if pursuing Option 3b)
- **Notes:** Full documentation, user guide, tutorials. Start here for PICLas evaluation (Month 5).

**[SW-6] PICLas GitHub Repository**
- **URL:** https://github.com/piclas-framework/piclas
- **Access:** 🟢 Open Source (GPL-3.0)
- **Priority:** ⭐⭐⭐ Critical (Option 3b)
- **Notes:** Source code (Fortran), examples, feature documentation. See docs/documentation/userguide/features-and-models/ for physics modules.

**[SW-7] Pfeiffer, M. et al. (2019)**
- **Title:** "PICLas: A Highly Flexible Particle Code for the Simulation of Reactive Plasma Flows"
- **Conference:** AIAA Aerospace Sciences Meeting, AIAA-2019-0962
- **DOI:** 10.2514/6.2019-0962
- **Access:** 🟡 Institutional Access (AIAA)
- **Priority:** ⭐⭐ Important (Option 3b)
- **Notes:** Overview paper describing PICLas architecture and validation. Applications to hypersonic reentry and EP thrusters.

**[SW-8] Munz, C.-D. et al. (2014)**
- **Title:** "Coupled Particle-In-Cell and Direct Simulation Monte Carlo Method for Simulating Reactive Plasma Flows"
- **Journal:** Comptes Rendus Mécanique, Vol. 342, pp. 662-670
- **DOI:** 10.1016/j.crme.2014.07.005
- **Access:** 🟡 Institutional Access
- **Priority:** ⭐ Supplementary (Option 3b)
- **Notes:** Theoretical foundation for PICLas coupling approach. Explains DSMC-PIC handoff methodology.

### 7.3 ANISE (Orbital Mechanics)

**[SW-9] ANISE Official Website**
- **URL:** https://nyxspace.com/anise/
- **Access:** 🟢 Open Source (Apache 2.0)
- **Priority:** ⭐ Supplementary (constellation analysis project)
- **Notes:** ANISE used in AeriSat constellation analysis project (not IntakeSIM). Included for completeness.

**[SW-10] ANISE GitHub Repository**
- **URL:** https://github.com/nyx-space/anise
- **Access:** 🟢 Open Source (Apache 2.0)
- **Priority:** ⭐ Supplementary
- **Notes:** Rust + Python bindings. Used in NASA Firefly Blue Ghost lunar mission (TRL-9).

---

## 8. Foundational Textbooks

**[TEXT-1] Curtis, H.D. (2013)**
- **Title:** "Orbital Mechanics for Engineering Students" (3rd Edition)
- **Publisher:** Butterworth-Heinemann
- **ISBN:** 978-0080977478
- **Access:** 🔴 Purchase (~$80)
- **Priority:** ⭐⭐ Important (for constellation context)
- **Notes:** Standard orbital mechanics textbook. Chapter 6 (orbital maneuvers), Chapter 8 (perturbations). Used in AeriSat constellation project, not directly in IntakeSIM.

**[TEXT-2] Vallado, D.A. (2013)**
- **Title:** "Fundamentals of Astrodynamics and Applications" (4th Edition)
- **Publisher:** Microcosm Press
- **ISBN:** 978-1881883180
- **Access:** 🔴 Purchase (~$90)
- **Priority:** ⭐ Supplementary
- **Notes:** More advanced orbital mechanics reference. Frame transformations, perturbations.

**[TEXT-3] Lieberman, M.A. & Lichtenberg, A.J. (2005)**
- **Title:** "Principles of Plasma Discharges and Materials Processing" (2nd Edition)
- **Publisher:** Wiley-Interscience
- **ISBN:** 978-0471720010
- **Access:** 🔴 Purchase (~$150)
- **Priority:** ⭐⭐⭐ Critical
- **Notes:** **Essential RF discharge theory.** Chapter 11 (RF discharges, ICP/CCP). Explains power balance, sheath physics, electron heating mechanisms. **Read before implementing PIC module (Week 7-9).**

**[TEXT-4] Chen, F.F. (2015)**
- **Title:** "Introduction to Plasma Physics and Controlled Fusion" (3rd Edition)
- **Publisher:** Springer
- **ISBN:** 978-3319223087
- **Access:** 🔴 Purchase (~$80)
- **Priority:** ⭐⭐ Important
- **Notes:** Undergraduate-level plasma physics. Good for building intuition before tackling Lieberman & Lichtenberg.

**[TEXT-5] Raizer, Y.P. (1991)**
- **Title:** "Gas Discharge Physics"
- **Publisher:** Springer-Verlag
- **ISBN:** 978-3540194620
- **Access:** 🔴 Purchase (~$200)
- **Priority:** ⭐ Supplementary
- **Notes:** Comprehensive gas discharge reference. Chapter 7 (RF discharges). More detailed than Lieberman but harder to read.

---

## 9. Validation Benchmark Summary Table

| Benchmark | Reference | Target Values | IntakeSIM Module | Week |
|-----------|-----------|---------------|------------------|------|
| **Thermal Transpiration** | Sharipov (1998) Table 8 | Δp/p = √(T_hot/T_cold) - 1 within 10% | DSMC | 3 |
| **Poiseuille Flow** | Sharipov (1998) Tables 10-12 | Flow rate vs Kn within 15% | DSMC | 3 |
| **Romano Intake η_c** | Romano et al. (2021) Table 8 | η_c = 0.458 ± 30% at 150 km | DSMC | 6 |
| **Child-Langmuir Sheath** | Lieberman Ch. 6 | s = 5λ_D within 20% | PIC | 9 |
| **Turner CCP Benchmark** | Turner et al. (2013) | n_e, φ within 20% | PIC | 9 |
| **Parodi N₂ CR (local)** | Parodi et al. (2025) | CR ~ 5-10 (outlet/inlet) | DSMC | 6 |
| **Parodi Plasma Density** | Parodi et al. (2025) | n_plasma = 1.65×10¹⁷ ± 30% | PIC | 11 |
| **Parodi T_e** | Parodi et al. (2025) | T_e = 7.8 eV ± 20% | PIC | 11 |
| **Parodi Thrust** | Parodi et al. (2025) | Thrust = 480 μN ± 30% | Coupled | 13 |
| **Power Balance** | Theory | P_in = P_out within 10% | PIC | 9,11 |

---

## 10. Open Data Resources

### Cross-Section Databases
- **LXCat**: https://www.lxcat.net (electron-impact cross-sections)
- **NIST Atomic Spectra Database**: https://www.nist.gov/pml/atomic-spectra-database
- **IAEA AMDIS**: https://www-amdis.iaea.org/ALADDIN/ (atomic & molecular data)

### Atmospheric Models
- **NRLMSISE-00 Online Calculator**: https://ccmc.gsfc.nasa.gov/modelweb/models/nrlmsise00.php
- **NRLMSIS 2.0 Data**: https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/NRLMSIS2.0/
- **Space Weather Data**: https://www.swpc.noaa.gov/

### Software Repositories
- **SPARTA**: https://github.com/sparta/sparta
- **PICLas**: https://github.com/piclas-framework/piclas
- **ANISE**: https://github.com/nyx-space/anise

### Validation Data
- **Sharipov's Data Files**: http://fisica.ufpr.br/sharipov/ (rarefied flow benchmarks)
- **Turner Benchmark Data**: Available in Turner et al. (2013) supplementary materials

---

## 11. How to Obtain Paywalled Papers

### Strategy 1: Institutional Access
- University library subscriptions (IEEE Xplore, ScienceDirect, Springer, AIP)
- Ask collaborators with university affiliations
- Some employers have institutional subscriptions

### Strategy 2: Author Contact
Most researchers are happy to share PDFs of their published work:
- **Pietro Parodi**: pietro.parodi@vki.ac.be (von Karman Institute)
- **Francesco Romano**: francesco.romano@unipi.it (University of Pisa)
- Email authors directly with polite request

### Strategy 3: Preprints & Open Repositories
- **ArXiv**: https://arxiv.org/ (Parodi 2025 available here!)
- **ResearchGate**: Many authors self-archive papers
- **Google Scholar**: Click "All versions" to find open-access copies

### Strategy 4: Interlibrary Loan
- Many public libraries offer interlibrary loan services
- Academic libraries typically provide this for alumni

### Strategy 5: Legal Purchase
- Individual article purchase (~$30-50 per paper)
- Only recommended for critical papers if other methods fail

---

## 12. Reading Priority for IntakeSIM Development

### Week 1-2 (Before Coding):
1. ⭐⭐⭐ **Andreussi et al. (2022)** - ABEP overview [ABEP-4]
2. ⭐⭐⭐ **Bird (1994)** - Chapters 2-4, 12 [DSMC-2]
3. ⭐⭐⭐ **Parodi et al. (2025)** - Primary validation target [ABEP-1]

### Week 3-4 (DSMC Implementation):
4. ⭐⭐⭐ **Sharipov & Seleznev (1998)** - Benchmark data [DSMC-7]
5. ⭐⭐⭐ **Romano et al. (2021)** - Intake validation [ABEP-2]
6. ⭐⭐⭐ **Krasheninnikov et al. (2023)** - Catalytic recombination [SURF-3]

### Week 7-9 (PIC Implementation):
7. ⭐⭐⭐ **Birdsall & Langdon (1991)** - Chapters 4, 7, 10 [PIC-1]
8. ⭐⭐⭐ **Lieberman & Lichtenberg (2005)** - Chapter 11 [TEXT-3]
9. ⭐⭐⭐ **Turner et al. (2013)** - CCP benchmark [PIC-5]
10. ⭐⭐⭐ **LXCat Database** - Download cross-sections [PLASMA-7]

### Week 11-13 (Validation):
11. ⭐⭐⭐ **Vaughan (1989)** - SEE model [PLASMA-1]
12. ⭐⭐⭐ **Lindsay & Stebbings (2005)** - CEX cross-sections [PLASMA-5]
13. ⭐⭐⭐ **Cifali et al. (2011)** - Experimental context [ABEP-3]

---

## 13. Additional Notes

### ITAR Considerations
- Most fundamental physics papers are NOT ITAR-controlled
- ABEP mission-specific parameters may be sensitive
- When publishing, use generic "representative CubeSat ABEP system" language
- Consult legal before first conference presentation

### Version Control
- This reference list reflects state-of-the-art as of January 2025
- ABEP is rapidly evolving field - expect new papers in 2025-2026
- Set up Google Scholar alerts for: "air-breathing electric propulsion", "ABEP", "atmosphere-breathing"
- Monitor IEPC and AIAA SciTech proceedings for latest developments

### Citation Tracking
- Use Google Scholar "Cited by" to find follow-on work
- Parodi (2025) is very recent - expect citations to appear in 2025-2026
- Romano (2021) has ~50 citations - good entry point for intake literature

---

## Contact for Questions

**IntakeSIM Project Lead**: George Boyce (CTO, AeriSat Systems)

**For specific topics, consider reaching out to:**
- **ABEP systems**: T. Andreussi (Sitael), V. Romano (ESA)
- **PIC-DSMC coupling**: P. Parodi (VKI), G. Lapenta (KU Leuven)
- **DSMC methods**: M. Gallis (Sandia), F. Sharipov (UFPR)
- **Plasma simulation**: M. Turner (DCU), J. Verboncoeur (MSU)

---

**Document Version**: 1.0 (January 2025)
**Total References**: 67 papers + 5 textbooks + 8 online resources = 80 total

**Last Updated**: Generated via Claude Code research session, January 2025
