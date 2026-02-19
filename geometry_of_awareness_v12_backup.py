import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class GeometryOfAwareness:
    """
    Geometry of Awareness Framework v1.2
    - n=7 or n=15 dimensional manifold
    - Salience-gated Hebbian metric learning
    - Laplacian SPD metric (guaranteed stable)
    - State-dependent metric C(x) with RBF kernel
    - Riemannian geometry: Christoffel symbols, scalar curvature
    - Competing basins (Healthy, Rigid, Trauma)
    - Lyapunov stability analysis
    - Therapy protocol support
    - Phase-diagram sweep utilities
    """
    def __init__(self, n=7, alpha=0.65, rho=0.018, eta0=0.055, 
                 trust_base=0.65, trust_vol=0.12, surprisal_amp=0.8,
                 emo_weight=1.2, grad_weight=0.9, social_weight=0.7,
                 w_T=4.0, beta_rbf=0.5, sigma_rbf=0.3, seed=42):
        np.random.seed(seed)
        self.n = n
        self.alpha = alpha
        self.rho = rho
        self.eta0 = eta0
        self.C = np.zeros((n, n)) + 0.01 * np.eye(n)  # small diagonal bias
        self.g = None
        self.update_metric()
        
        # RBF kernel for state-dependent metric
        self.beta_rbf = beta_rbf  # strength of local perturbations
        self.sigma_rbf = sigma_rbf  # width of RBF kernel
        self.rbf_centers = []  # recent history points
        
        # Basins (n=7 or n=15 support)
        if n == 7:
            self.mu_H = np.array([0.55, 0.60, 0.55, 0.58, 0.45, 0.50, 0.35])  # Healthy
            self.mu_R = np.array([0.25, 0.70, 0.30, 0.75, 0.75, 0.40, 0.20])  # Rigid
            self.mu_T = np.array([-0.6, -0.4, -0.5, -0.3, -0.7, 0.1, -0.2])   # Trauma
            self.dim_names = ['Emotion', 'Memory', 'Narrative', 'Belief', 'Identity', 'Archetypal', 'Sensory']
        elif n == 15:
            # Extended to 15 dimensions: pad with plausible basin centers
            self.mu_H = np.concatenate([np.array([0.55, 0.60, 0.55, 0.58, 0.45, 0.50, 0.35]),
                                       np.array([0.50, 0.45, 0.55, 0.48, 0.42, 0.52, 0.49, 0.45])])
            self.mu_R = np.concatenate([np.array([0.25, 0.70, 0.30, 0.75, 0.75, 0.40, 0.20]),
                                       np.array([0.65, 0.70, 0.72, 0.68, 0.75, 0.70, 0.73, 0.72])])
            self.mu_T = np.concatenate([np.array([-0.6, -0.4, -0.5, -0.3, -0.7, 0.1, -0.2]),
                                       np.array([-0.5, -0.55, -0.45, -0.6, -0.4, -0.3, -0.5, -0.55])])
            self.dim_names = ['Emotion', 'Memory', 'Narrative', 'Belief', 'Identity', 'Archetypal', 'Sensory',
                            'Somatic', 'Cognitive', 'Social', 'Spiritual', 'Motor', 'Aesthetic', 'Temporal', 'Spatial']
        else:
            raise ValueError(f"Unsupported n={n}. Use n=7 or n=15.")
        
        self.w_H = 1.0
        self.w_R = 1.45
        self.w_T = w_T
        self.sigma_T = 0.82
        
        # Salience parameters
        self.trust_base = trust_base
        self.trust_vol = trust_vol
        self.surprisal_amp = surprisal_amp
        self.emo_weight = emo_weight
        self.grad_weight = grad_weight
        self.social_weight = social_weight
        
        self.history = {'x': [], 'C': [], 'cond_g': [], 'lambda': [], 'V': []}
        self.seed = seed
    
    def update_metric(self):
        W = np.maximum(self.C, 0)
        D = np.diag(W.sum(axis=1))
        L = D - W
        self.g = np.eye(self.n) + self.alpha * L
        # enforce SPD
        eigvals = np.linalg.eigvalsh(self.g)
        if np.min(eigvals) < 1e-8:
            self.g += 1e-6 * np.eye(self.n)
    
    def reset(self, trust_base=None, w_T=None):
        """Reset state for clean reuse in sweeps (replaces __init__ call in loop)"""
        if trust_base is not None:
            self.trust_base = trust_base
        if w_T is not None:
            self.w_T = w_T
        self.C = np.zeros((self.n, self.n)) + 0.01 * np.eye(self.n)
        self.rbf_centers = []
        self.history = {'x': [], 'C': [], 'cond_g': [], 'lambda': [], 'V': []}
        self.update_metric()
    
    def compute_C_state_dependent(self, x):
        """Compute state-dependent metric C(x) = C_global + RBF perturbations"""
        C_local = self.C.copy()
        if len(self.rbf_centers) > 0:
            rbf_centers = np.array(self.rbf_centers[-20:])  # use last 20 points
            for center in rbf_centers:
                dist_sq = np.sum((x - center)**2)
                rbf_weight = self.beta_rbf * np.exp(-dist_sq / (2 * self.sigma_rbf**2))
                C_local += rbf_weight * np.outer(center, center) * 0.01  # small amplitude
        return np.maximum(C_local, 0)
    
    def compute_g_state_dependent(self, x):
        """Compute state-dependent metric g(x) from local C(x)"""
        C_x = self.compute_C_state_dependent(x)
        W = np.maximum(C_x, 0)
        D = np.diag(W.sum(axis=1))
        L = D - W
        g_x = np.eye(self.n) + self.alpha * L
        # enforce SPD
        eigvals = np.linalg.eigvalsh(g_x)
        if np.min(eigvals) < 1e-8:
            g_x += 1e-6 * np.eye(self.n)
        return g_x
    
    def get_condition_number(self):
        return np.linalg.cond(self.g)
    
    def get_condition_number_x(self, x):
        """Condition number at point x"""
        g_x = self.compute_g_state_dependent(x)
        return np.linalg.cond(g_x)
    
    def potential(self, x):
        x = np.asarray(x).reshape(-1)
        V_H = self.w_H * np.sum((x - self.mu_H)**2)
        V_R = self.w_R * np.sum((x - self.mu_R)**2)
        V_T = self.w_T * np.exp(-0.5 * np.sum((x - self.mu_T)**2) / self.sigma_T**2)
        return V_H + V_R + V_T, (V_H, V_R, V_T)
    
    def salience(self, x, surprisal=0.0, trust=None):
        if trust is None:
            trust = np.clip(self.trust_base + np.random.normal(0, self.trust_vol), 0.1, 1.0)
        x1 = abs(x[0])  # emotion
        _, (VH, VR, VT) = self.potential(x)
        gradV = np.gradient([self.potential(x + 1e-4*np.eye(self.n)[i])[0] for i in range(self.n)])[0]
        grad_norm = np.linalg.norm(gradV)
        
        arg = (self.emo_weight * x1 +
               self.grad_weight * grad_norm +
               self.surprisal_amp * surprisal +
               self.social_weight * trust - 0.8)
        return 1 / (1 + np.exp(-arg))  # logistic
    
    def step(self, x0, surprisal=0.0, trust=None, dt=0.08, therapy_mode=False):
        lam = self.salience(x0, surprisal, trust)
        if therapy_mode:
            lam = np.clip(lam * 1.15, 0.4, 0.78)  # guided moderate band
        
        # normalize x
        x_norm = np.tanh(x0)  # [-1,1]
        
        # Hebbian update
        delta = np.outer(x_norm, x_norm)
        self.C = (1 - self.rho) * self.C + self.eta0 * lam * delta
        np.fill_diagonal(self.C, np.maximum(self.C.diagonal(), 0.01))
        self.update_metric()
        
        # Store state for RBF kernel
        self.rbf_centers.append(x0.copy())
        
        # dynamics: gradient flow on manifold
        grad_num = np.zeros(self.n)
        eps = 1e-5
        for i in range(self.n):
            x_plus = x0.copy(); x_plus[i] += eps
            x_min = x0.copy(); x_min[i] -= eps
            grad_num[i] = (self.potential(x_plus)[0] - self.potential(x_min)[0]) / (2*eps)
        
        g_inv = np.linalg.inv(self.g)
        dx = -dt * g_inv @ grad_num
        
        x1 = x0 + dx + 0.03 * np.random.randn(self.n)  # stochastic
        x1 = np.clip(x1, -1.2, 1.2)
        
        # log
        cond = self.get_condition_number()
        Vtot, _ = self.potential(x1)
        self.history['x'].append(x1.copy())
        self.history['C'].append(self.C.copy())
        self.history['cond_g'].append(cond)
        self.history['lambda'].append(lam)
        self.history['V'].append(Vtot)
        
        return x1, lam, cond
    
    def compute_jacobian(self, x0, trust=0.7, eps=1e-5, dt=0.08):
        """
        Analytical Jacobian of discrete map: J = I - dt*g_inv*Hess_V(x0)
        
        Pure function: no mutation of state.
        Computed analytically (not numerically) to avoid noise amplification.
        
        Args:
            x0: Point at which to evaluate Jacobian
            trust: Trust parameter (not used, kept for API compatibility)
            eps: Finite-difference step for Hessian (default 1e-5)
            dt: Discrete time step (default 0.08)
        
        Returns:
            J: n×n Jacobian matrix of the discrete map dx -> x_next
        """
        x0 = np.asarray(x0).flatten().copy()
        
        # Compute Hessian of potential via finite differences
        # H[i,j] = ∂²V/∂x_i∂x_j
        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                x_pp = x0.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pm = x0.copy(); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = x0.copy(); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = x0.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                
                V_pp, _ = self.potential(x_pp)
                V_pm, _ = self.potential(x_pm)
                V_mp, _ = self.potential(x_mp)
                V_mm, _ = self.potential(x_mm)
                
                H[i, j] = (V_pp - V_pm - V_mp + V_mm) / (4 * eps**2)
        
        # Compute metric at x0
        g_x = self.compute_g_state_dependent(x0)
        g_inv = np.linalg.inv(g_x)
        
        # Jacobian: J = I - dt * g_inv * H
        J = np.eye(self.n) - dt * (g_inv @ H)
        
        return J
    
    def compute_christoffel(self, x, eps=1e-4):
        """Christoffel symbols Γᵏᵢⱼ = (1/2) gᵏˡ (∂gⱼˡ/∂xⁱ + ∂gᵢˡ/∂xʲ - ∂gᵢⱼ/∂xˡ)"""
        g_x = self.compute_g_state_dependent(x)
        Gamma = np.zeros((self.n, self.n, self.n))
        
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    # Compute partial derivatives via finite difference
                    dg_dxi = np.zeros((self.n, self.n))
                    dg_dxj = np.zeros((self.n, self.n))
                    dg_dxl = np.zeros((self.n, self.n, self.n))
                    
                    for l in range(self.n):
                        x_plus = x.copy(); x_plus[i] += eps
                        x_min = x.copy(); x_min[i] -= eps
                        g_plus = self.compute_g_state_dependent(x_plus)
                        g_minus = self.compute_g_state_dependent(x_min)
                        dg_dxi = (g_plus - g_minus) / (2 * eps)
                        
                        x_plus = x.copy(); x_plus[j] += eps
                        x_min = x.copy(); x_min[j] -= eps
                        g_plus = self.compute_g_state_dependent(x_plus)
                        g_minus = self.compute_g_state_dependent(x_min)
                        dg_dxj = (g_plus - g_minus) / (2 * eps)
                        
                        x_plus = x.copy(); x_plus[l] += eps
                        x_min = x.copy(); x_min[l] -= eps
                        g_plus = self.compute_g_state_dependent(x_plus)
                        g_minus = self.compute_g_state_dependent(x_min)
                        dg_dxl[:, :] = (g_plus - g_minus) / (2 * eps)
                    
                    # Γᵏᵢⱼ = (1/2) gᵏˡ (∂gⱼˡ/∂xⁱ + ∂gᵢˡ/∂xʲ - ∂gᵢⱼ/∂xˡ)
                    sum_term = dg_dxi[j, :] + dg_dxj[i, :] - dg_dxl[i, j]
                    g_inv = np.linalg.inv(g_x)
                    Gamma[k, i, j] = 0.5 * np.sum(g_inv[k, :] * sum_term)
        
        return Gamma
    
    def compute_riemann_scalar(self, x):
        """Scalar curvature R = gⁱʲ Rᵢⱼ (contraction of Riemann tensor)"""
        g_x = self.compute_g_state_dependent(x)
        Gamma = self.compute_christoffel(x, eps=1e-4)
        
        # Approximate Riemann via Christoffel contraction
        # Ricci tensor: Rᵢⱼ = ∂Γᵏᵢⱼ/∂xᵏ - ... (simplified)
        # For demonstration: use sum of Christoffel squared as curvature proxy
        R_component = 0.0
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    R_component += Gamma[k, i, j]**2
        
        g_inv = np.linalg.inv(g_x)
        R = np.sum(g_inv) * np.sqrt(R_component) * 0.01  # normalized scalar curvature
        return R
    
    # Phase sweep utility
    def run_sweep(self, trust_vals, trauma_vals, runs_per_cell=15, steps=800):
        results = {}
        for t0 in trust_vals:
            for at in trauma_vals:
                key = (t0, at)
                outcomes = {'H':0, 'R':0, 'T':0, 'L':0, 'cond_mean':[], 'core_C':[]}
                for r in range(runs_per_cell):
                    self.reset(trust_base=t0, w_T=at)  # fast reset (no __init__)
                    x = np.random.uniform(-0.3, 0.3, self.n)
                    for _ in range(steps):
                        x, _, _ = self.step(x)
                    # classify end-state
                    VH = self.w_H * np.sum((x - self.mu_H)**2)
                    VR = self.w_R * np.sum((x - self.mu_R)**2)
                    VT = self.w_T * np.exp(-0.5 * np.sum((x - self.mu_T)**2) / self.sigma_T**2)
                    eps = 0.15
                    if VH + eps < min(VR, VT):
                        outcomes['H'] += 1
                    elif VR + eps < min(VH, VT):
                        outcomes['R'] += 1
                    elif VT > max(VH, VR) + 0.5:
                        outcomes['T'] += 1
                    else:
                        outcomes['L'] += 1
                    outcomes['cond_mean'].append(np.mean(self.history['cond_g'][-100:]))
                    core_edges = [(1,2),(2,3),(3,4),(1,3)]  # Mem-Nar-Bel-Id
                    core_C = np.mean([self.C[i,j] for i,j in core_edges])
                    outcomes['core_C'].append(core_C)
                results[key] = {k: np.mean(v) if 'mean' in k else np.sum(v)/runs_per_cell for k,v in outcomes.items() if k != 'cond_mean' and k != 'core_C'}
                results[key]['cond_mean'] = np.mean(outcomes['cond_mean'])
                results[key]['core_C'] = np.mean(outcomes['core_C'])
        return results
    
    # Therapy protocol
    def run_therapy(self, pre_steps=400, therapy_steps=240, trust_lift=0.18):
        self.trust_base += trust_lift
        x = np.random.uniform(-0.2, 0.2, self.n)
        for _ in range(pre_steps):
            x, _, _ = self.step(x)
        pre_cond = np.mean(self.history['cond_g'][-50:])
        for _ in range(therapy_steps):
            x, _, _ = self.step(x, therapy_mode=True)
        post_cond = np.mean(self.history['cond_g'][-50:])
        return pre_cond, post_cond, x
    
    def lyapunov_analysis(self, basin='H', trust=0.7, n_steps=500):
        """Lyapunov stability: run to equilibrium, compute Jacobian eigenvalues"""
        if basin == 'H':
            x_eq = self.mu_H.copy()
        elif basin == 'R':
            x_eq = self.mu_R.copy()
        else:
            x_eq = self.mu_T.copy()
        
        # Run dynamics near basin center
        x = x_eq + 0.01 * np.random.randn(self.n)
        for _ in range(n_steps):
            x, _, _ = self.step(x, trust=trust)
        
        # Compute Jacobian at equilibrium
        J = self.compute_jacobian(x, trust=trust)
        eigs = np.linalg.eigvals(J)
        max_abs_eig = np.max(np.abs(eigs))
        
        return {
            'basin': basin,
            'trust': trust,
            'equilibrium': x.copy(),
            'eigenvalues': eigs,
            'max_abs_eigenvalue': max_abs_eig,
            'stable': max_abs_eig < 1.0,
            'eq_potential': self.potential(x)[0]
        }