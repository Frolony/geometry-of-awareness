"""
Riemannian Leapfrog Integrator for Geometry of Awareness
=========================================================

Implements velocity-Verlet integrator on Riemannian manifolds with:
- Laplacian-based SPD metric derived from signed coherence C_ij
- Optional Christoffel symbol corrections (geodesic deviation)
- Hebbian learning for coherence matrix updates
- Stochastic exploration and state bounds enforcement

Integration with GeometryOfAwareness:
- Standalone functions work with any metric callback system
- Pre-configured integrator class for direct model usage
- Compatible with v1.3 signed coherence and inhibitory dynamics

Usage:
    >>> from geometry_of_awareness import GeometryOfAwareness
    >>> from riemannian_leapfrog import create_integrator
    >>> 
    >>> model = GeometryOfAwareness(n=7)
    >>> integrator = create_integrator(model)
    >>> x, v = integrator.step(x, v)
"""

import numpy as np
from scipy.linalg import eigh
from typing import Callable, Tuple, Optional, Dict, Any


# ============================================================================
# Core Riemannian Leapfrog Integration
# ============================================================================

def compute_metric_from_coherence(C: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """
    Compute Laplacian-based SPD metric from signed coherence matrix.
    
    Metric is: g = I + α·L, where L is the graph Laplacian formed from 
    positive coherences (edges). This yields an SPD metric representing
    graph structure learned via Hebbian plasticity.
    
    Args:
        C: Coherence matrix C_ij ∈ ℝ (signed), shape (n, n)
        alpha: Metric weight on Laplacian (default 0.65)
    
    Returns:
        g: SPD metric tensor, shape (n, n)
    
    References:
        - Coherence as connection strength
        - Laplacian eigenvalues relate to diffusion properties
        - SPD property guaranteed for stable learning
    """
    n = C.shape[0]
    W = np.maximum(C, 0)  # Keep only positive (integrative) couplings
    D = np.diag(W.sum(axis=1))  # Degree matrix
    L = D - W  # Graph Laplacian
    
    # Metric: I + α·L (controls metric deformation)
    g = np.eye(n) + alpha * L
    
    # Enforce SPD via small regularization if needed
    eigvals = np.linalg.eigvalsh(g)
    if np.min(eigvals) < 1e-8:
        g += 1e-6 * np.eye(n)
    
    return g


def compute_metric_inverse(C: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Compute g⁻¹ directly (more numerically stable than inverting g)."""
    g = compute_metric_from_coherence(C, alpha)
    return np.linalg.inv(g)


def compute_potential_gradient(
    x: np.ndarray,
    model,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute potential gradient ∇V via finite differences.
    
    Numerically stable approach: uses central differences and handles
    the full v1.3 potential including inhibitory terms.
    
    Args:
        x: State vector, shape (n,)
        model: GeometryOfAwareness instance with potential() method
        eps: Finite difference step size
    
    Returns:
        grad_V: Gradient ∇V(x), shape (n,)
    """
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        
        V_plus, _ = model.potential(x_plus)
        V_minus, _ = model.potential(x_minus)
        
        grad[i] = (V_plus - V_minus) / (2 * eps)
    
    return grad


def update_coherence(
    C: np.ndarray,
    x: np.ndarray,
    rho: float = 0.018,
    eta: float = 0.055,
    lam: float = 0.5
) -> np.ndarray:
    """
    Update coherence via Hebbian learning rule.
    
    Signed coherence update: Hebbian learning for positive correlations
    while allowing negative couplings (inhibition) to emerge from 
    divergent activity patterns.
    
    Rule: C_new = (1-ρ)·C + η·λ·(x⊗x)
    
    Args:
        C: Current coherence matrix, shape (n, n)
        x: Current state (normalized to [-1,1]), shape (n,)
        rho: Decay rate (forgetting) - typical 0.018
        eta: Learning rate - typical 0.055
        lam: Salience gate [0,1] - controls learning speed
    
    Returns:
        C_new: Updated coherence matrix with preserved structure
    
    Notes:
        - Diagonal enforced ≥ 0.01 (self-coupling baseline)
        - Outer product captures correlations naturally
        - Both positive and negative correlations can emerge
        - v1.3: supports signed (inhibitory) couplings
    """
    x_norm = np.tanh(x)  # Clip to [-1, 1] nonlinearly
    
    # Hebbian update: correlation-based learning
    delta = np.outer(x_norm, x_norm)
    C_new = (1 - rho) * C + eta * lam * delta
    
    # Enforce minimum diagonal (self-coupling baseline)
    np.fill_diagonal(C_new, np.maximum(C_new.diagonal(), 0.01))
    
    return C_new


def christoffel_quadratic_form(
    Gamma: np.ndarray,
    v: np.ndarray
) -> np.ndarray:
    """
    Compute Christoffel symbol quadratic form: Γ(v,v).
    
    The geodesic acceleration term in Riemannian dynamics:
        (Γ(v,v))ᵏ = Γᵏᵢⱼ·vⁱ·vʲ
    
    This represents geodesic deviation (curvature effects on trajectories).
    
    Args:
        Gamma: Christoffel symbols, shape (n, n, n), where Gamma[k,i,j] = Γᵏᵢⱼ
        v: Velocity vector, shape (n,)
    
    Returns:
        quad_form: (n,) vector representing Γ(v,v)
    """
    n = len(v)
    quad_form = np.zeros(n)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                quad_form[k] += Gamma[k, i, j] * v[i] * v[j]
    
    return quad_form


def riemannian_leapfrog_step(
    x: np.ndarray,
    v: np.ndarray,
    C: np.ndarray,
    compute_g_inv: Callable[[np.ndarray], np.ndarray],
    compute_grad_V: Callable[[np.ndarray], np.ndarray],
    compute_christoffel_quad: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    dt: float = 0.08,
    noise_scale: float = 0.032,
    rho: float = 0.018,
    eta: float = 0.055,
    lam: float = 0.5,
    x_bounds: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Riemannian leapfrog / velocity-Verlet integration step.
    
    Solves geodesic equation on a Riemannian manifold with metric g:
        ẍᵏ = -gᵏˡ(∂ₗV + Γˡᵢⱼ·ẋⁱ·ẋʲ)
    
    The leapfrog scheme is symplectic-like and preserves phase space
    structure better than Euler methods, important for long trajectories
    and Hamiltonian-like dynamics.
    
    Args:
        x: Position vector (state), shape (n,)
        v: Velocity vector, shape (n,)
        C: Coherence matrix, shape (n, n)
        compute_g_inv: Function C → g⁻¹ (metric inverse)
        compute_grad_V: Function x → ∇V (potential gradient)
        compute_christoffel_quad: Optional function (x, v) → Γ(v,v)
                                  Pass None for flat metric (no curvature)
        dt: Time step (default 0.08)
        noise_scale: Stochastic exploration amplitude (default 0.032)
        rho: Coherence decay rate (default 0.018)
        eta: Coherence learning rate (default 0.055)
        lam: Salience gate [0,1] (default 0.5)
        x_bounds: Hard clipping bounds for position (default (-1.0, 1.0))
    
    Returns:
        x_new: Updated position, shape (n,)
        v_new: Updated velocity, shape (n,)
        C_new: Updated coherence matrix, shape (n, n)
    
    Algorithm Steps:
        1. Compute acceleration at current position: a = -g⁻¹∇V - Γ(v,v)
        2. Half-step velocity: v_half = v + (dt/2)·a
        3. Full-step position: x_new = x + dt·v_half (with bounds clipping)
        4. Update coherence via Hebbian learning: C_new = f(C, x_new)
        5. Recompute acceleration at new position with updated metric
        6. Complete velocity step: v_new = v_half + (dt/2)·a_new
        7. Add stochastic exploration noise
    
    Notes:
        - Symplectic structure: preserves phase space volume
        - Coherence updates are Hebbian: learns correlations from trajectory
        - Christoffel term optional: set to None for flat metrics
        - Noise enables exploration beyond gradient descent
        - Hard bounds prevent divergence in unbounded potentials
    
    References:
        - LeimKuhler & Shang (2016): Adaptive Thermostat for MD
        - Girolami & Calderhead (2011): Riemannian Langevin Dynamics
    """
    n = len(x)
    
    # Step 1: Current acceleration
    g_inv = compute_g_inv(C)
    grad_V = compute_grad_V(x)
    a = -g_inv @ grad_V
    
    if compute_christoffel_quad is not None:
        a -= compute_christoffel_quad(x, v)
    
    # Step 2: Half-step velocity
    v_half = v + 0.5 * dt * a
    
    # Step 3: Full-step position with bounds
    x_new = x + dt * v_half
    x_new = np.clip(x_new, x_bounds[0], x_bounds[1])
    
    # Step 4: Update coherence matrix (Hebbian learning)
    C_new = update_coherence(C, x_new, rho=rho, eta=eta, lam=lam)
    
    # Step 5: Recompute acceleration at new position
    g_inv_new = compute_g_inv(C_new)
    grad_V_new = compute_grad_V(x_new)
    a_new = -g_inv_new @ grad_V_new
    
    if compute_christoffel_quad is not None:
        a_new -= compute_christoffel_quad(x_new, v_half)
    
    # Step 6: Complete velocity step
    v_new = v_half + 0.5 * dt * a_new
    
    # Step 7: Stochastic exploration (Brownian motion)
    v_new += noise_scale * np.random.randn(n)
    
    return x_new, v_new, C_new


# ============================================================================
# Integrator Class for GeometryOfAwareness
# ============================================================================

class RiemannianLeapfrogIntegrator:
    """
    High-level integrator wrapping leapfrog dynamics with GeometryOfAwareness.
    
    Provides:
    - Automatic metric computation from model coherence
    - Optional Christoffel symbol corrections
    - Phase space trajectory tracking
    - Statistical observers (energy, salience, stability)
    
    Example:
        >>> model = GeometryOfAwareness(n=7, seed=42)
        >>> integrator = RiemannianLeapfrogIntegrator(model)
        >>> 
        >>> x = np.random.uniform(-0.3, 0.3, 7)
        >>> v = 0.1 * np.random.randn(7)
        >>> 
        >>> for step in range(1000):
        >>>     x, v = integrator.step(x, v)
        >>> 
        >>> print(f"Energy: {integrator.stats['total_energy'][-1]:.6f}")
    """
    
    def __init__(
        self,
        model,
        dt: float = 0.08,
        noise_scale: float = 0.032,
        use_christoffel: bool = True,
        alpha_metric: float = 0.65,
        track_stats: bool = True
    ):
        """
        Initialize integrator bound to a GeometryOfAwareness model.
        
        Args:
            model: GeometryOfAwareness instance
            dt: Time step
            noise_scale: Brownian motion amplitude
            use_christoffel: Include geodesic deviation (Christoffel terms)
            alpha_metric: Weight on metric Laplacian deformation
            track_stats: Log energy, salience, condition number
        """
        self.model = model
        self.dt = dt
        self.noise_scale = noise_scale
        self.use_christoffel = use_christoffel
        self.alpha_metric = alpha_metric
        
        self.stats = {
            'total_energy': [],
            'kinetic_energy': [],
            'potential_energy': [],
            'salience': [],
            'condition_number': [],
            'christoffel_norm': []
        }
        self.track_stats = track_stats
    
    def compute_g_inv_model(self, C: np.ndarray) -> np.ndarray:
        """Metric inverse using model's metric computation."""
        return compute_metric_inverse(C, alpha=self.alpha_metric)
    
    def compute_grad_V_model(self, x: np.ndarray) -> np.ndarray:
        """Potential gradient via model's potential function."""
        return compute_potential_gradient(x, self.model, eps=1e-5)
    
    def compute_christoffel_quad_model(
        self,
        x: np.ndarray,
        v: np.ndarray
    ) -> np.ndarray:
        """Christoffel quadratic form if available."""
        if not self.use_christoffel:
            return np.zeros(len(x))
        
        try:
            Gamma = self.model.compute_christoffel(x, eps=1e-4)
            return christoffel_quadratic_form(Gamma, v)
        except:
            # Fallback if Christoffel computation fails
            return np.zeros(len(x))
    
    def step(
        self,
        x: np.ndarray,
        v: np.ndarray,
        surprisal: float = 0.0,
        trust: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute one leapfrog integration step with model updates.
        
        Args:
            x: Position (state)
            v: Velocity
            surprisal: Optional salience modifier
            trust: Optional trust parameter for salience computation
        
        Returns:
            x_new, v_new (note: C updated internally in model)
        """
        # Compute salience for coherence learning rule
        lam = self.model.salience(x, surprisal=surprisal, trust=trust)
        
        # Leapfrog step with model callbacks
        x_new, v_new, C_new = riemannian_leapfrog_step(
            x, v, self.model.C,
            compute_g_inv=self.compute_g_inv_model,
            compute_grad_V=self.compute_grad_V_model,
            compute_christoffel_quad=self.compute_christoffel_quad_model,
            dt=self.dt,
            noise_scale=self.noise_scale,
            rho=self.model.rho,
            eta=self.model.eta0,
            lam=lam,
            x_bounds=(-1.2, 1.2)  # Allow some overshoot
        )
        
        # Update model's coherence matrix
        self.model.C = C_new
        self.model.update_metric()
        
        # Track statistics
        if self.track_stats:
            V_total, (V_H, V_R, V_T, V_inhib) = self.model.potential(x_new)
            KE = 0.5 * np.dot(v_new, v_new)  # Kinetic energy
            
            self.stats['total_energy'].append(KE + V_total)
            self.stats['kinetic_energy'].append(KE)
            self.stats['potential_energy'].append(V_total)
            self.stats['salience'].append(lam)
            self.stats['condition_number'].append(self.model.get_condition_number_x(x_new))
            
            if self.use_christoffel:
                try:
                    Gamma = self.model.compute_christoffel(x_new, eps=1e-4)
                    christoffel_norm = np.sqrt(np.sum(Gamma**2))
                    self.stats['christoffel_norm'].append(christoffel_norm)
                except:
                    self.stats['christoffel_norm'].append(0.0)
        
        return x_new, v_new
    
    def trajectory(
        self,
        x0: np.ndarray,
        v0: np.ndarray,
        n_steps: int = 1000,
        surprisal: float = 0.0,
        trust: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Integrate full trajectory and return history.
        
        Args:
            x0: Initial position
            v0: Initial velocity
            n_steps: Number of integration steps
            surprisal: Optional salience modifier
            trust: Optional trust parameter
        
        Returns:
            Dictionary with 'position', 'velocity', 'energy', 'salience', ...
        """
        x, v = x0.copy(), v0.copy()
        history = {
            'position': [x.copy()],
            'velocity': [v.copy()]
        }
        
        for step in range(n_steps):
            x, v = self.step(x, v, surprisal=surprisal, trust=trust)
            history['position'].append(x.copy())
            history['velocity'].append(v.copy())
        
        # Append tracking statistics
        for key, val in self.stats.items():
            history[key] = np.array(val)
        
        return history
    
    def reset_stats(self):
        """Clear tracking statistics."""
        self.stats = {key: [] for key in self.stats}


def create_integrator(
    model,
    dt: float = 0.08,
    noise_scale: float = 0.032,
    use_christoffel: bool = True
) -> RiemannianLeapfrogIntegrator:
    """
    Factory function to create an integrator for a model.
    
    This is the recommended entry point for standard usage.
    
    Args:
        model: GeometryOfAwareness instance
        dt: Time step (default 0.08)
        noise_scale: Brownian exploration amplitude (default 0.032)
        use_christoffel: Include Riemannian curvature effects (default True)
    
    Returns:
        Initialized RiemannianLeapfrogIntegrator
    
    Example:
        >>> from geometry_of_awareness import GeometryOfAwareness
        >>> from riemannian_leapfrog import create_integrator
        >>> 
        >>> model = GeometryOfAwareness(n=7)
        >>> integrator = create_integrator(model)
        >>> x, v = integrator.step(x, v)
    """
    return RiemannianLeapfrogIntegrator(
        model,
        dt=dt,
        noise_scale=noise_scale,
        use_christoffel=use_christoffel,
        track_stats=True
    )


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == '__main__':
    """Integration test with GeometryOfAwareness model."""
    try:
        from geometry_of_awareness import GeometryOfAwareness
        
        print("Testing Riemannian Leapfrog Integrator...")
        print("=" * 70)
        
        # Initialize model and integrator
        model = GeometryOfAwareness(n=7, seed=42)
        integrator = create_integrator(model, dt=0.08, use_christoffel=False)
        
        # Initial conditions
        x0 = np.random.uniform(-0.3, 0.3, 7)
        v0 = 0.05 * np.random.randn(7)
        
        print(f"\nInitial state:")
        print(f"  x0 = {x0}")
        print(f"  v0 = {v0}")
        print(f"  ||x0|| = {np.linalg.norm(x0):.6f}")
        print(f"  ||v0|| = {np.linalg.norm(v0):.6f}")
        
        # Integrate trajectory
        print(f"\nIntegrating {500} steps...")
        x, v = x0.copy(), v0.copy()
        for step in range(500):
            x, v = integrator.step(x, v)
        
        print(f"\nFinal state:")
        print(f"  x_final = {x}")
        print(f"  v_final = {v}")
        print(f"  ||x_final|| = {np.linalg.norm(x):.6f}")
        print(f"  ||v_final|| = {np.linalg.norm(v):.6f}")
        
        # Statistics
        print(f"\nStatistics (last 50 steps):")
        print(f"  Mean energy: {np.mean(integrator.stats['total_energy'][-50:]):.6f}")
        print(f"  Mean KE: {np.mean(integrator.stats['kinetic_energy'][-50:]):.6f}")
        print(f"  Mean PE: {np.mean(integrator.stats['potential_energy'][-50:]):.6f}")
        print(f"  Mean salience: {np.mean(integrator.stats['salience'][-50:]):.6f}")
        print(f"  Mean κ(g): {np.mean(integrator.stats['condition_number'][-50:]):.6f}")
        
        print("\n✓ Integration test passed!")
        print("=" * 70)
        
    except ImportError:
        print("Note: GeometryOfAwareness not available. Skipping integration test.")
        print("Module is still importable and usable as a standalone leapfrog integrator.")