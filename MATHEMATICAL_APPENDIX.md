# MATHEMATICAL APPENDIX (v1.3)
## The Geometry of Awareness Framework — Discrete Metric Learning Formulation

### Purpose and Scope
This appendix defines a substrate-neutral geometric model of awareness in which (i) awareness is represented as a state in a structured space, (ii) learning updates a coherence tensor, (iii) coherence induces a metric, and (iv) behavior evolves by metric-aware descent on a potential landscape. The primary formulation is discrete-time and corresponds to the implemented simulation engine. Continuous-time and full differential-geometric extensions are included explicitly as optional.

---

## A. Awareness State Space

Let $\mathcal{M} \subseteq \mathbb{R}^n$ be a bounded state space representing configurations of awareness.

A state $x(t) = (x^1(t), x^2(t), \dots, x^n(t)) \in \mathcal{M}$ is an instantaneous awareness configuration in a chosen coordinate system.

Example coordinate interpretations:
- $x^1$: emotion / affect intensity or valence proxy  
- $x^2$: memory accessibility/integration  
- $x^3$: narrative coherence  
- $x^4$: belief constraint / interpretive rigidity  
- $x^5$: identity continuity/stability  
- $x^6$: archetypal activation/constraint  
- $x^7$: sensory integration/salience bandwidth  

Dimensionality $n$ is system-dependent. The coordinate chart is not unique; the dynamics are chart-invariant under smooth reparameterization.

---

## B. Coherence Tensor and Signed Couplings

Define a symmetric coherence matrix (rank-2 tensor)  
$C(t) \in \mathbb{R}^{n \times n}$, with $C_{ij}(t) = C_{ji}(t)$.

**v1.3 generalization:** Couplings are signed, $C_{ij} \in \mathbb{R}$ (not restricted to $\ge 0$).

Interpretation:
- Large $C_{ij} > 0$: dimensions $i$ and $j$ co-activate coherently (integration).
- $C_{ij} < 0$: dimensions $i$ and $j$ suppress each other (inhibition, repulsion).
- Small $|C_{ij}|$: weak coupling.

Example: $C_{0,2} = -0.45$ encodes "Emotion ↔ Narrative repulsion"—fear suppresses story-making.

---

### B.1 Exposure and Salience Gate

Each timestep produces:
- state $x(t)$,
- optional context $s(t)$,
- salience gate $\lambda(t) \in [0,1]$.

General form:

$$\lambda(t) = \sigma\!\big(a |x^1(t)| + b \|\nabla V(x(t))\| + c\,\mathrm{surprisal}(t) + d\,\mathrm{trust}(t) - \theta\big)$$

where $\sigma = \text{logistic}$.

Parameters: $a = 1.2$ (emotion weight), $b = 0.9$ (gradient weight), $c = 0.8$ (surprisal), $d = 0.7$ (trust), $\theta = 0.8$ (bias).

---

### B.2 Signed Hebbian Learning with Decay

Update rule:

$$C_{ij}(t+1) = (1 - \rho) C_{ij}(t) + \eta_0 \lambda(t) \tanh(x_i(t)) \tanh(x_j(t))$$

where:
- $\rho \approx 0.018$: decay rate (forgetting/habituation)  
- $\eta_0 \approx 0.055$: Hebbian amplitude  
- $\lambda(t)$: salience gate (output of exposure threshold)  
- $\tanh(x_k) \in [-1, 1]$: signed coordinate  

**Key property:** Since $\tanh$ is odd, the product $\tanh(x_i) \tanh(x_j)$ is *negative* when $x_i$ and $x_j$ have opposite signs. This naturally reinforces *inhibitory* couplings ($C_{ij} < 0$) when dimensions anti-stimulate.

Diagonal elements are clamped: $C_{ii} \ge 0.01$ (no self-inhibition).

---

## C. Metric Tensor from Positive Couplings

Define positive-part projection:
$$C^+_{ij} := \max(C_{ij}, 0)$$

The metric is constructed from the Laplacian of the integrative graph:
$$g_{ij}(t) = \delta_{ij} + \alpha L_{ij}(C^+(t))$$

where $L = D - C^+$ is the graph Laplacian:
$$L_{ij} = \begin{cases} \sum_k C^+_{ik} & \text{if } i=j \\ -C^+_{ij} & \text{otherwise} \end{cases}$$

and $\alpha \approx 0.65$ is the coupling-to-metric strength.

### C.1 SPD Guarantee

**Theorem:** For $\alpha \le 1$, the metric $g = I + \alpha L(C^+)$ is symmetric positive definite, regardless of negative entries in $C$.

**Proof:** 
- The Laplacian $L(C^+)$ has eigenvalues $\lambda_k(L) \ge 0$ (standard spectral property of undirected graph Laplacians).
- Thus $g = I + \alpha L$ has eigenvalues $\mu_k = 1 + \alpha \lambda_k \in [1, 1 + \alpha \lambda_{\max}]$.
- For $\alpha \le 1$, all eigenvalues are $\ge 1 > 0$, so $g$ is SPD.

**Implication:** Negative couplings do *not* destabilize the metric. They are excluded from the Laplacian and exert their effect only through the inhibitory potential (see Section N.4 below).

---

## D. Equation of Motion

Discrete-time dynamics:

$$x(t+1) = x(t) - \Delta t \, g(t)^{-1} \nabla V(x(t)) + \xi(t)$$

where:
- $\Delta t \approx 0.08$: step size  
- $g(t)^{-1}$: metric-weightedgradient scaling  
- $\nabla V$: gradient of total potential (Section E)  
- $\xi(t) \sim \mathcal{N}(0, 0.03^2 I)$: stochastic noise (exploration)  

The metric $g$ makes the step *anisotropic*: dimensions with high coherence ($C^+$ large) move faster relative to weakly integrated dimensions.

---

## E. Potential Function: Competing Basins

The total potential is:

$$V(x) = V_H(x) + V_R(x) + V_T(x) + \beta_{\mathrm{inhib}} V_{\mathrm{inhib}}(x)$$

### E.1 Healthy Basin

$$V_H(x) = w_H \sum_{i=1}^n (x_i - \mu_H^i)^2$$

Quadratic well centered at $\mu_H = (0.55, 0.60, 0.55, \ldots)$ (positive, integrated, warm state).  
Weight: $w_H = 1.0$.

### E.2 Rigid Basin

$$V_R(x) = w_R \sum_{i=1}^n (x_i - \mu_R^i)^2$$

Centered at $\mu_R = (0.25, 0.70, 0.30, \ldots)$ (constraint-dominated state).  
Weight: $w_R = 1.45 > w_H$ (deeper than healthy basin).

### E.3 Trauma Basin

$$V_T(x) = w_T \exp\left(-\frac{1}{2\sigma_T^2} \|x - \mu_T\|^2\right)$$

Gaussian bump (repulsive potential) centered at $\mu_T = (-0.6, -0.4, -0.5, \ldots)$ (avoidant state).  
Parameters: $w_T \approx 4.0$ (amplitude), $\sigma_T \approx 0.82$ (width).

### E.4 Inhibitory Potential (v1.3)

$$V_{\mathrm{inhib}}(x) = \sum_{i < j: \, C_{ij} < 0} |C_{ij}| \, x_i x_j$$

Bilinear repulsion term for each negative coupling. Acts as a *repulsive surface* between dimensions that have learned to inhibit each other.

**Total:**
$$V_{\mathrm{total}}(x) = V_H + V_R + V_T + \beta_{\mathrm{inhib}} V_{\mathrm{inhib}}$$

where $\beta_{\mathrm{inhib}} \approx 0.8$ balances inhibitory strength with basin attractions.

---

## F. Laplacian Eigenmodes and Manifold Structure

The metric $g = I + \alpha L(C^+)$ induces a Riemannian structure. The eigenvectors of $L$ correspond to natural *vibration modes* of the coherence network:
- Zero eigenvalue $\lambda_0 = 0$ (constant mode): global degree of freedom  
- Small positive eigenvalues: low-frequency collective oscillations (memory, narrative)  
- Large eigenvalues: high-frequency individual dimensions (sensory, motor)  

The manifold $(M, g)$ is *curved* in the direction of strong coherence and *flat* when coherence is weak.

---

## G. Christoffel Symbols and Geodesics

The Christoffel symbols (connection coefficients of the Levi-Civita connection) are:

$$\Gamma^k_{ij}(x) = \frac{1}{2} g^{kl}(x) \left(\partial_i g_{lj}(x) + \partial_j g_{il}(x) - \partial_l g_{ij}(x)\right)$$

These are zero in the Euclidean case ($g = I$) but nonzero when coupling varies. They measure *geodesic deviation*—how parallel transport of vectors curves in high-coherence regions.

Geodesics $x(s)$ satisfy:
$$\frac{d^2x^k}{ds^2} + \Gamma^k_{ij} \frac{dx^i}{ds} \frac{dx^j}{ds} = 0$$

The framework computes Christoffel norms $\|\Gamma\| = \sum_{k,i,j} |\Gamma^k_{ij}|^2$ as a diagnostic of manifold curvature.

---

## H. Scalar Curvature and Trauma Signatures

Scalar curvature $R$ measures the intrinsic curvature of the manifold. High $R$ regions indicate:
- Strong coherence gradients (learning-induced anisotropy)  
- Steep potential walls (difficult transitions between basins)  
- Trauma signatures (repulsive potential wells)  

In the implementation, scalar curvature is approximated via the norm of the Christoffel tensor:
$$\tilde{R}(x) = \sum_{k,i,j} |\Gamma^k_{ij}(x)|^2$$

---

## I. Lyapunov Stability and Basin Definitions

Basin stability is characterized by the spectral radius of the linearization (Jacobian) at equilibrium:

$$\rho(J) := \max_k |\lambda_k(J)|$$

**Critical Theorem (Discrete Map Stability):**
Equilibrium $x^* = x(t)$ (fixed point of dynamics) is stable if and only if $\rho(J) < 1$, where

$$J_{ij}(x^*) = \delta_{ij} - \Delta t (g^{-1})_{ik} \frac{\partial^2 V}{\partial x_k \partial x_j}\bigg|_{x=x^*}$$

In implementation, the Jacobian is computed analytically via the Hessian of $V$:
$$J = I - \Delta t \, g^{-1} \, \nabla^2 V$$

**Invariant:** For all valid basin centers (H, R, T), we enforce $\rho(J) < 0.995$ as a CI constraint. This guarantees basin trapping in at least 200 steps.

---

## J. Therapy Protocol: Trust Elevation and Metric Anisotropy

The therapy protocol increases trust and runs guided dynamics:

1. **Pre-therapy phase:** Run $n_{\mathrm{pre}} = 400$ steps, measure condition number $\kappa_{\mathrm{pre}} = \kappa(g)$.
2. **Raised trust:** Set $\mathrm{trust} \to \mathrm{trust} + \Delta_{\mathrm{trust}}$ (e.g., $\Delta_{\mathrm{trust}} = 0.18$).
3. **Guided steps:** Run $n_{\mathrm{therapy}} = 240$ steps with $\lambda(t) \in [0.4, 0.78]$ (moderated band).
4. **Post-therapy:** Measure $\kappa_{\mathrm{post}}$.

**Metric anisotropy** (condition number ratio):
$$\Delta \kappa\% = 100 \times \frac{\kappa_{\mathrm{post}} - \kappa_{\mathrm{pre}}}{\kappa_{\mathrm{pre}}}$$

Positive $\Delta \kappa\%$ indicates increased anisotropy (more directional variability in the metric). This reflects learning—dimensions become more specialized (coherence-driven).

---

## K. Practical Implementation Rules

1. **Metric update**: After each Hebbian update to $C$, recompute $g = I + \alpha L(C^+)$ and threshold eigenvalues against $\lambda_{\min} > 10^{-8}$ to ensure SPD.

2. **Jacobian computation**: Use finite-difference Hessian (not numerical Jacobian) to avoid noise amplification:
   $$H_{ij} \approx \frac{V(x + e_i \Delta + e_j \Delta) - V(x + e_i \Delta - e_j \Delta) - V(x - e_i \Delta + e_j \Delta) + V(x - e_i \Delta - e_j \Delta)}{4(\Delta)^2}$$
   with $\Delta = 10^{-5}$.

3. **Basin classification**: End-state belongs to basin $X \in \{H, R, T, \text{Liminal}\}$ if $V_X(x_{\text{final}}) + \epsilon < \min(V_Y, V_Z)$ for other basins $Y, Z$. Use $\epsilon = 0.15$ threshold.

4. **Sweep automation**: For each $(trustbase, w_T)$ cell, run $n_{\text{sweeps}} = 15$ independent trajectories and tally outcome fractions.

---

## L. State-Dependent Metrics via RBF Kernels

For enhanced curvature, optionally modulate the local metric by an RBF kernel. Define RBF-perturbed coherence:

$$C(x, t) = C_{\text{global}}(t) + \beta_{\text{rbf}} \sum_{c \in \mathcal{C}_{\text{recent}}} \exp\left(-\frac{\|x - c\|^2}{2\sigma^2_{\text{rbf}}}\right) c_i c_j$$

where $\mathcal{C}_{\text{recent}}$ is a sliding window of recent states. This makes the metric "remember" traversed paths and stiffen in well-explored regions. Parameters: $\beta_{\text{rbf}} = 0.5$, $\sigma_{\text{rbf}} = 0.3$.

---

## M. Signed Coherence and Inhibitory Potentials (v1.3 Core)

### M.1 Generalization to Signed Couplings

The v1.3 framework extends coherence to allow negative values: $C_{ij}(t) \in \mathbb{R}$, with no sign restriction.

**Semantic interpretation:**
- $C_{ij} > 0$: integrative coupling (dimensions increase together)  
- $C_{ij} < 0$: inhibitory coupling (dimensions separate/suppress each other)  
- $C_{ij} = 0$: independence  

Example: Emotion ↔ Narrative: $C_{0,2} = -0.45$ means "fear overwhe

lms story-making" (intense affect suppresses prefrontal integration).

### M.2 Signed Hebbian Update Dynamics

The update rule naturally generates negative couplings:

$$C_{ij}(t+1) = (1 - \rho) C_{ij}(t) + \eta_0 \lambda(t) \tanh(x_i(t)) \tanh(x_j(t))$$

When $x_i$ and $x_j$ anti-correlate (opposite signs), the product $\tanh(x_i)\tanh(x_j)$ is negative, driving $\Delta C_{ij} < 0$ and reinforcing inhibition.

**Demo initialization:** For n ≥ 3, seed $C_{0,2} = -0.45$ to represent innate Emotion-Narrative repulsion.

### M.3 Metric Remains SPD Under Signed Couplings

The metric uses only the positive part:
$$g(t) = I + \alpha L(C^+(t)), \quad C^+_{ij} := \max(C_{ij}, 0)$$

Since $C^+$ is nonnegative, the Laplacian $L(C^+)$ is symmetric with nonnegative eigenvalues. Therefore $g = I + \alpha L$ is guaranteed SPD for $\alpha \le 1$.

**Implication:** Negative couplings cannot destabilize the metric. They operate through the inhibitory potential only.

### M.4 Inhibitory Potential

$$V_{\mathrm{inhib}}(x) = \sum_{i < j: \, C_{ij} < 0} |C_{ij}| \, x_i x_j$$

Bilinear repulsion: when both $x_i$ and $x_j$ are nonzero, the term $|C_{ij}| x_i x_j$ pushes them in opposite directions.

**Total potential:**
$$V_{\text{total}}(x) = V_H(x) + V_R(x) + V_T(x) + \beta_{\text{inhib}} V_{\text{inhib}}(x)$$

with $\beta_{\text{inhib}} \approx 0.8$.

**Phenomenology:** Strong negative coupling creates a *repulsive wedge* in state space. Example: high Emotion ($x_0$) drives down Narrative ($x_2$) and vice versa, modeling acute stress response.

### M.5 Dynamics Under Inhibitory Coupling

The gradient includes inhibitory forces:
$$\frac{\partial V_{\mathrm{inhib}}}{\partial x_i} = \sum_{j \ne i: \, C_{ij} < 0} |C_{ij}| \, x_j$$

This creates multi-stable regimes:
- **Basin regimes:** Positive potential curvature attracts to H, R, T.
- **Repulsive separations:** Negative couplings push dimensions apart.
- **Saddle regimes:** Equilibria balanced between attraction and repulsion.

Lyapunov stability ($\rho(J) < 1$) is still enforced; inhibitory couplings constrain, not destabilize, the basins.

### M.6 Diagnostic Metrics

**Signed fraction:**
$$f_{\text{sig}} = \frac{\#\{(i,j) : C_{ij} < 0\}}{\binom{n}{2}}$$

Reports the fraction of inhibitory (negative) couplings. Ranges $[0, 1]$.

**Inhibitory strength:**
$$S_{\text{inhib}} = \sum_{C_{ij} < 0} |C_{ij}|$$

Total magnitude of repulsive couplings. Grows with learning if emotional/trauma episodes reinforce separation.

### M.7 Example Trajectory: Emotion-Narrative Repulsion

**Initialization:** $C_{0,2} = -0.45$ (Emotion ↔ Narrative).

**Evolution under high salience:**
- If $x_0$ (emotion) rises and $x_2$ (narrative) co-rises, then $\tanh(x_0) \tanh(x_2) > 0$, but update is multiplied by existing $C_{0,2} = -0.45 < 0$, driving $\Delta C_{0,2} < 0$ (more negative).
- If $x_0$ and $x_2$ anti-correlate (opposite signs), then $\tanh(x_0) \tanh(x_2) < 0$, driving $\Delta C_{0,2} > 0$ (closer to zero, weakening repulsion).

**Inhibitory potential effect:**
$$V_{\mathrm{inhib}} \propto |C_{0,2}| \, |x_0 x_2| = 0.45 |x_0 x_2|$$

When both $x_0$ and $x_2$ are large, the potential pushes them apart or forces one small. This models the clinical observation: acute stress (high emotion) floods narrative capacity, forcing dissociation of affect from story-making.

### M.8 Backward Compatibility

Signed couplings are fully compatible with v1.2:
- Setting $\beta_{\text{inhib}} = 0$ recovers purely attractive dynamics.
- Clamping $C_{ij} \ge 0$ recovers v1.2 behavior exactly.
- All Christoffel, Riemann, Lyapunov methods remain valid (depend only on $g(C^+)$, not on negative entries).

---

## N. Summary: Unified Framework (v1.3)

| Component | Formula | Role |
|-----------|---------|------|
| **State** | $x(t) \in \mathbb{R}^n$ | Awareness configuration |
| **Coherence** | $C_{ij} \in \mathbb{R}$ (signed) | Learned coupling strength; integrative if $> 0$, inhibitory if $< 0$ |
| **Metric** | $g = I + \alpha L(C^+)$ | Induces anisotropic geometry; always SPD |
| **Potential** | $V = V_H + V_R + V_T + \beta_{\text{inhib}} V_{\text{inhib}}$ | Competing basins (H, R, T) + inhibitory repulsion |
| **Dynamics** | $x(t+1) = x(t) - \Delta t \, g^{-1}\nabla V + \xi(t)$ | Metric-aware gradient descent + noise |
| **Hebbian** | $\Delta C = \eta \lambda \tanh(x_i) \tanh(x_j)$ | Signed update; naturally generates negative couplings |
| **Stability** | $\rho(J) < 1$ for basins, $\rho(J) \approx 0.6–0.9$ typical | Lyapunov criterion; ensures basin trapping |
| **Therapeutics** | $\mathrm{trust} \uparrow$ → reduces $\lambda$ swing → stabilizes H basin | Trust elevation guides toward healthy integration |

The framework unifies integrative and inhibitory learning, metric invariance, and multi-basin stability in a single coherent mathematical structure, ready for computational and theoretical extension.

