import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class GeometryOfAwareness:
    """
    Geometry of Awareness Framework v1.0
    - n=7 dimensional manifold
    - Salience-gated Hebbian metric learning
    - Laplacian SPD metric (guaranteed stable)
    - Competing basins (Healthy, Rigid, Trauma)
    - Therapy protocol support
    - Phase-diagram sweep utilities
    """
    def __init__(self, n=7, alpha=0.65, rho=0.018, eta0=0.055, 
                 trust_base=0.65, trust_vol=0.12, surprisal_amp=0.8,
                 emo_weight=1.2, grad_weight=0.9, social_weight=0.7,
                 w_T=4.0, seed=42):
        np.random.seed(seed)
        self.n = n
        self.alpha = alpha
        self.rho = rho
        self.eta0 = eta0
        self.C = np.zeros((n, n)) + 0.01 * np.eye(n)  # small diagonal bias
        self.g = None
        self.update_metric()
        
        # Basins (quadratic + Gaussian)
        self.mu_H = np.array([0.55, 0.60, 0.55, 0.58, 0.45, 0.50, 0.35])  # Healthy
        self.w_H = 1.0
        self.mu_R = np.array([0.25, 0.70, 0.30, 0.75, 0.75, 0.40, 0.20])  # Rigid (high Bel/Id)
        self.w_R = 1.45
        self.mu_T = np.array([-0.6, -0.4, -0.5, -0.3, -0.7, 0.1, -0.2])   # Trauma repulsor center
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
        self.dim_names = ['Emotion', 'Memory', 'Narrative', 'Belief', 'Identity', 'Archetypal', 'Sensory']
    
    def update_metric(self):
        W = np.maximum(self.C, 0)
        D = np.diag(W.sum(axis=1))
        L = D - W
        self.g = np.eye(self.n) + self.alpha * L
        # enforce SPD
        eigvals = np.linalg.eigvalsh(self.g)
        if np.min(eigvals) < 1e-8:
            self.g += 1e-6 * np.eye(self.n)
    
    def get_condition_number(self):
        return np.linalg.cond(self.g)
    
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
        
        # dynamics: gradient flow on manifold
        _, _ = self.potential(x0)
        grad = np.array([np.sum((x0 - self.mu_H) * self.w_H) + ...])  # full analytic grad omitted for brevity; implemented numerically
        # (full numeric grad in package)
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
    
    def compute_jacobian(self, x0, trust=0.7, eps=1e-6, dt=0.08):
        """Numerical Jacobian of the map x -> step(x, trust=trust)"""
        x0 = np.asarray(x0).flatten().copy()
        J = np.zeros((self.n, self.n))
        f0, _, _ = self.step(x0.copy(), trust=trust, dt=dt)  # one step
        
        for i in range(self.n):
            x_plus = x0.copy()
            x_plus[i] += eps
            f_plus, _, _ = self.step(x_plus, trust=trust, dt=dt)
            J[:, i] = (f_plus - f0) / eps
        return J
    
    # Phase sweep utility
    def run_sweep(self, trust_vals, trauma_vals, runs_per_cell=15, steps=800):
        results = {}
        for t0 in trust_vals:
            for at in trauma_vals:
                key = (t0, at)
                outcomes = {'H':0, 'R':0, 'T':0, 'L':0, 'cond_mean':[], 'core_C':[]}
                for r in range(runs_per_cell):
                    self.__init__(trust_base=t0, w_T=at)  # reset
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

