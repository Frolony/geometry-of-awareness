# MATHEMATICAL APPENDIX (v1.2)
## The Geometry of Awareness Framework — Discrete Metric Learning Formulation

### Purpose and Scope
This appendix defines a substrate-neutral geometric model of awareness in which (i) awareness is represented as a state in a structured space, (ii) learning updates a coherence tensor, (iii) coherence induces a metric, and (iv) behavior evolves by metric-aware descent on a potential landscape. The primary formulation is discrete-time and corresponds to the implemented simulation engine. Continuous-time and full differential-geometric extensions are included explicitly as optional.

---

## A. Awareness State Space

Let \( \mathcal{M} \subseteq \mathbb{R}^n \) be a bounded state space representing configurations of awareness.

A state \( x(t) = (x^1(t), x^2(t), \dots, x^n(t)) \in \mathcal{M} \) is an instantaneous awareness configuration in a chosen coordinate system.

Example coordinate interpretations:
- \(x^1\): emotion / affect intensity or valence proxy  
- \(x^2\): memory accessibility/integration  
- \(x^3\): narrative coherence  
- \(x^4\): belief constraint / interpretive rigidity  
- \(x^5\): identity continuity/stability  
- \(x^6\): archetypal activation/constraint  
- \(x^7\): sensory integration/salience bandwidth  

Dimensionality \(n\) is system-dependent. The coordinate chart is not unique; the dynamics are chart-invariant under smooth reparameterization.

---

## B. Coherence Tensor as the Learned Relational State

Define a symmetric coherence matrix (rank-2 tensor)  
\( C(t) \in \mathbb{R}^{n \times n} \), with \( C_{ij}(t) = C_{ji}(t) \).

Interpretation:
- Large \( C_{ij} \): dimensions \(i\) and \(j\) co-activate coherently.
- Small \( C_{ij} \): weak integration.

In the baseline implementation, \( C_{ij}(t) \ge 0 \).

---

### B.1 Exposure and Salience Gate

Each timestep produces:
- state \( x(t) \),
- optional context \( s(t) \),
- salience gate \( \lambda(t) \in [0,1] \).

General form:

\( \lambda(t) = \sigma\!\big(a |x^1(t)| + b \|\nabla V(x(t))\| + c\,\mathrm{surprisal}(t) + d\,\mathrm{trust}(t) - \theta\big) \)

where \( \sigma(\cdot) \) is logistic.

High \( \lambda \) enables rapid structural update; low \( \lambda \) yields gradual drift.

---

### B.2 Hebbian-with-Decay Update Rule

Let \( \tilde{x}(t) \) be normalized coordinates (e.g., \( \tanh(x) \)). Define co-activation:

\( \Delta_{ij}(t) = \tilde{x}_i(t)\tilde{x}_j(t) \)

Update rule:

\( C_{ij}(t+1) = (1-\rho) C_{ij}(t) + \eta \lambda(t) [\Delta_{ij}(t)]_+ \)

where:
- \( \rho \in (0,1) \): decay  
- \( \eta > 0 \): learning rate  
- \( [\cdot]_+ \): optional rectification  

This yields instant encoding when \( \lambda \approx 1 \), slow consolidation otherwise.

---

### B.3 Context-Conditioned Plasticity (Optional)

\( \eta(t) = \eta_0 \lambda(t) h(k(t)) \)

where \( h(\cdot) \) is a context-dependent modulation.

---

### B.4 Signed Coupling Variant (Optional)

Allow \( C_{ij}(t) \in \mathbb{R} \). Positive components may feed metric construction; negative couplings may enter potential-field terms separately.

---

## C. Metric Construction from Coherence

Geometry is defined by an SPD metric \( g(t) \in \mathbb{R}^{n \times n} \).

### C.1 Laplacian SPD Metric (Implemented)

Let \( W(t) = \max(C(t),0) \). Define degree matrix \( D_{ii}(t) = \sum_j W_{ij}(t) \). Graph Laplacian:

\( L(t) = D(t) - W(t) \)

Metric:

\( g(t) = I + \alpha L(t), \quad \alpha > 0 \)

Properties:
- Symmetric  
- Positive definite  
- Invertible  
- Numerically stable  

Interpretation: learned coherence reshapes traversal costs between awareness dimensions.

---

### C.2 Entrywise Monotone Mapping (Optional)

\( g_{ij}(t) = \Phi(C_{ij}(t)) \), with \( \partial \Phi / \partial C < 0 \). SPD must be enforced separately if used.

---

## D. Potential Landscape

Define \( V: \mathcal{M} \to \mathbb{R} \).

Attractors: \( \nabla V(x^*) = 0 \), \( \nabla^2 V(x^*) \succ 0 \).  
Repulsor barriers: large \( \|\nabla V\| \).

\( V(x) \) may include:
- healthy integration basin  
- rigid/defensive basin  
- trauma barrier term  

---

## E. Discrete-Time Dynamics (Implemented Core)

Update rule:

\( x(t+1) = x(t) - \Delta t\, g(t)^{-1} \nabla V(x(t)) + \xi(t) \)

where:
- \( \Delta t > 0 \): step size  
- \( \xi(t) \): optional noise  

Continuous limit: \( \dot{x} = -g^{-1}\nabla V(x) \).

Optional inertial extension:

\( \ddot{x}^k + \Gamma^k_{ij}\dot{x}^i\dot{x}^j = -g^{kl}\partial_l V - \gamma \dot{x}^k \)

(not used in implemented engine).

---

## F. Local Stability and Lyapunov Criterion

Discrete map \( F(x) = x - \Delta t\, g^{-1}\nabla V(x) \).

Jacobian:

\( J(x^*) = I - \Delta t\, g^{-1} \nabla^2 V(x^*) \)

Local stability requires spectral radius:

\( \rho(J) = \max_i |\lambda_i(J)| < 1 \)

This is the implemented Lyapunov stability test.

---

## G. Trauma, Rigidity, and Presence Diagnostics

Trauma-like regions exhibit:
- large \( \|\nabla V\| \)
- sharp avoidance trajectories
- deformation spikes \( \|g(t)-g(t-1)\|_F \)

Anisotropy proxy:

\( \kappa(g(t)) = \mathrm{cond}(g(t)) \)

Connection proxy (if \( g(x,t) \)):

\( \Gamma^k_{ij} = \frac{1}{2} g^{kl}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}) \)

If \( g = g(t) \) only, then \( \Gamma \approx 0 \).

---

## H. Bias as Relational Deformation

Let baseline \( T^{(0)} \). Strain:

\( B_{ij}(t) = T_{ij}(t) - T^{(0)}_{ij} \)

Normalized bias:

\( \beta_{ij}(t) = \frac{B_{ij}(t)}{T^{(0)}_{ij}} \)

Bias correction = reshaping \( C(t) \) and therefore \( g(t) \).

---

## I. Healing as Controlled Metric Learning

Healing corresponds to structured exposure sequences that reshape \( C(t) \), enabling traversal toward integrative basins and reducing deformation spikes and anisotropy burden.

Ricci-flow analogy is conceptual only; the implemented mechanism is discrete metric learning.

---

## J. Presence as Minimal Distortion Regime

Presence corresponds to:
- stable basin capture  
- reduced deformation energy  
- lower anisotropy burden  
- flexible traversal between dimensions  

---

## K. Ontological Neutrality

The framework is a structural description. It does not assume neural, computational, phenomenological, or metaphysical substrate. The manifold and metric are modeling tools.

---

## L. Generalized Relational Law

Let:
- \( M(x) \): salience scalar  
- \( \|C(x)\| \): integration magnitude  
- \( \mathcal{R}(x) \): relational amplification operator  

Then:

\( E(x) = M(x)\|C(x)\|^2 \mathcal{R}(x) \)

Here \( \mathcal{R} \) denotes resonance, not curvature.

---

## M. Summary of Implemented Structure

State: \( x(t) \in \mathcal{M} \subseteq \mathbb{R}^n \)  
Coherence: \( C(t) \) via salience-gated Hebbian-with-decay  
Metric: \( g(t) = I + \alpha L(\max(C(t),0)) \)  
Potential: multi-basin \( V(x) \)  
Dynamics: \( x(t+1) = x(t) - \Delta t g^{-1}\nabla V(x(t)) + \xi(t) \)  
Stability: \( J = I - \Delta t g^{-1}\nabla^2 V \), stable if \( \rho(J) < 1 \)  
Diagnostics: \( \kappa(g) \), deformation norm, gradient magnitude, basin probabilities  

Optional extensions include state-dependent metrics \( g(x,t) \), connection terms, and full curvature tensors. The above defines the core falsifiable, simulation-ready framework.
