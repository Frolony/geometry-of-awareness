"""
Diagnostic script to isolate Jacobian scaling error.
Tests compute_jacobian against manual Hessian computation.
"""

import numpy as np
from scipy.linalg import eigh, eigvals
from geometry_of_awareness import GeometryOfAwareness

def manual_hessian_potential(model, x, eps=1e-5):
    """Compute Hessian of potential V(x) via finite differences"""
    n = model.n
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            
            V_pp, _ = model.potential(x_pp)
            V_pm, _ = model.potential(x_pm)
            V_mp, _ = model.potential(x_mp)
            V_mm, _ = model.potential(x_mm)
            
            H[i, j] = (V_pp - V_pm - V_mp + V_mm) / (4 * eps**2)
    
    return H

def main():
    print("="*70)
    print("JACOBIAN DIAGNOSTIC: Scale Error Analysis")
    print("="*70)
    
    # Initialize model
    model = GeometryOfAwareness(n=7, seed=42)
    model.reset()
    
    # Pick a test point (near Healthy basin)
    x_test = model.mu_H + 0.05 * np.random.randn(7)
    x_test = np.clip(x_test, -1.2, 1.2)
    
    print(f"\n[1] Test point: x = {x_test}")
    V_test, (VH, VR, VT) = model.potential(x_test)
    print(f"    Potential: V = {V_test:.4f} (VH={VH:.4f}, VR={VR:.4f}, VT={VT:.4e})")
    
    # Step 1: Compute Hessian of potential directly
    print(f"\n[2] Computing Hessian of potential V(x)...")
    H = manual_hessian_potential(model, x_test, eps=1e-5)
    H_eigs = np.linalg.eigvalsh(H)
    print(f"    Hessian eigenvalues: min={H_eigs.min():.4f}, max={H_eigs.max():.4f}")
    print(f"    Hessian condition number: {np.linalg.cond(H):.4f}")
    
    # Step 2: Compute metric at test point
    print(f"\n[3] Computing metric g(x)...")
    g_x = model.compute_g_state_dependent(x_test)
    g_inv = np.linalg.inv(g_x)
    g_eigs = np.linalg.eigvalsh(g_x)
    g_inv_eigs = np.linalg.eigvalsh(g_inv)
    print(f"    g eigenvalues: min={g_eigs.min():.4f}, max={g_eigs.max():.4f}")
    print(f"    g_inv eigenvalues: min={g_inv_eigs.min():.4f}, max={g_inv_eigs.max():.4f}")
    print(f"    g condition number: {np.linalg.cond(g_x):.4f}")
    
    # Step 3: Compute g_inv @ H (the key product)
    print(f"\n[4] Computing product g_inv @ Hess_V...")
    product = g_inv @ H
    product_eigs = np.linalg.eigvals(product)
    print(f"    (g_inv @ H) eigenvalue magnitudes: min={np.abs(product_eigs).min():.4f}, max={np.abs(product_eigs).max():.4f}")
    
    # Step 4: Expected Jacobian from theory
    print(f"\n[5] Expected Jacobian from theory: J = I - dt*g_inv*H")
    dt = 0.08
    J_expected = np.eye(7) - dt * product
    J_expected_eigs = np.linalg.eigvals(J_expected)
    print(f"    Expected J eigenvalues: min={np.abs(J_expected_eigs).min():.4f}, max={np.abs(J_expected_eigs).max():.4f}")
    
    # Step 5: Compute actual Jacobian from compute_jacobian()
    print(f"\n[6] Computing actual Jacobian via compute_jacobian()...")
    J_actual = model.compute_jacobian(x_test, trust=0.7, eps=1e-6, dt=0.08)
    J_actual_eigs = np.linalg.eigvals(J_actual)
    print(f"    Actual J eigenvalues: min={np.abs(J_actual_eigs).min():.4f}, max={np.abs(J_actual_eigs).max():.4f}")
    
    # Step 6: Comparison
    print(f"\n[7] DIAGNOSIS:")
    print(f"    Expected max|λ(J)|: {np.abs(J_expected_eigs).max():.4f}")
    print(f"    Actual max|λ(J)|:   {np.abs(J_actual_eigs).max():.4e}")
    ratio = np.abs(J_actual_eigs).max() / np.abs(J_expected_eigs).max()
    print(f"    Ratio (Actual/Expected): {ratio:.4e}")
    
    if ratio > 100:
        print(f"\n    ⚠️  SCALE ERROR DETECTED: Actual is {ratio:.0f}x too large")
        print(f"    This suggests a missing dt normalization or eps amplification")
    
    # Step 7: Lyapunov analysis for comparison
    print(f"\n[8] Running full lyapunov_analysis()...")
    lya = model.lyapunov_analysis(basin='H', trust=0.7, n_steps=200)
    lya_eigs = lya['eigenvalues']
    print(f"    Lyapunov max|λ|: {lya['max_abs_eigenvalue']:.4e}")
    print(f"    Stable? {lya['stable']}")
    
    print("\n" + "="*70)
    print("END DIAGNOSTIC")
    print("="*70)

if __name__ == "__main__":
    main()
