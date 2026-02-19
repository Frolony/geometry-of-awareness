"""
Visualization Dashboard for Geometry of Awareness v1.2

Generates comprehensive visualizations of:
- Dynamics (phase space, energy, salience, stability)
- Basin geometry (3D plots of potential landscape)
- Metric properties (condition number, eigenvalues, curvature)
- Lyapunov stability (eigenvalue distributions)
- Phase diagrams (trust vs trauma parameter space)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from geometry_of_awareness import GeometryOfAwareness

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'H': '#2ecc71', 'R': '#e74c3c', 'T': '#9b59b6', 'L': '#95a5a6'}

def plot_dynamics_trajectory(model, n_steps=1000, figsize=(15, 10)):
    """Plot phase space trajectory, energy, salience, and stability"""
    model.reset()
    x = np.random.uniform(-0.3, 0.3, model.n)
    
    for _ in range(n_steps):
        x, _, _ = model.step(x)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Dynamics Trajectory (n={model.n}, {n_steps} steps)', fontsize=14, fontweight='bold')
    
    traj = np.array(model.history['x'])
    
    # Phase space (first 3 dimensions)
    ax = axes[0, 0]
    ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.6, linewidth=0.5)
    ax.scatter(*traj[[0, -1], :2].T, c=['green', 'red'], s=100, zorder=5, label=['Start', 'End'])
    ax.set_xlabel('Emotion (x₀)')
    ax.set_ylabel('Memory (x₁)')
    ax.legend()
    ax.set_title('Phase-Space Trajectory (2D Projection)')
    ax.grid(True, alpha=0.3)
    
    # Potential energy
    ax = axes[0, 1]
    ax.plot(model.history['V'], 'k-', linewidth=1.5, label='Total V(x)')
    ax.fill_between(range(len(model.history['V'])), model.history['V'], alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Potential Energy V(x)')
    ax.set_title('Energy Landscape Descent')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Salience gating
    ax = axes[1, 0]
    ax.plot(model.history['lambda'], 'purple', linewidth=1, alpha=0.8, label='Salience λ(t)')
    ax.fill_between(range(len(model.history['lambda'])), model.history['lambda'], alpha=0.3, color='purple')
    ax.set_xlabel('Step')
    ax.set_ylabel('Salience λ')
    ax.set_ylim([0, 1.05])
    ax.set_title('Learning Rate Modulation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Metric stability (condition number)
    ax = axes[1, 1]
    ax.semilogy(model.history['cond_g'], 'orange', linewidth=1.5, label='cond(g)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Condition Number cond(g)')
    ax.set_title('Metric Stability')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_potential_landscape(model, n_grid=20):
    """Plot 3D potential surface for first two dimensions"""
    x0_range = np.linspace(-1, 1, n_grid)
    x1_range = np.linspace(-1, 1, n_grid)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    V = np.zeros_like(X0)
    
    for i in range(n_grid):
        for j in range(n_grid):
            x_test = np.zeros(model.n)
            x_test[0] = X0[i, j]
            x_test[1] = X1[i, j]
            V[i, j], _ = model.potential(x_test)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D surface
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X0, X1, V, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Emotion (x₀)')
    ax.set_ylabel('Memory (x₁)')
    ax.set_zlabel('Potential V(x)')
    ax.set_title(f'3D Potential Landscape (n={model.n})')
    
    # 2D contour with basins
    ax = fig.add_subplot(122)
    contour = ax.contour(X0, X1, V, levels=15, alpha=0.6, colors='gray')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Mark basins
    ax.scatter(model.mu_H[0], model.mu_H[1], s=200, c='green', marker='*', 
              edgecolors='black', linewidths=2, label='Healthy', zorder=5)
    ax.scatter(model.mu_R[0], model.mu_R[1], s=200, c='red', marker='X', 
              edgecolors='black', linewidths=2, label='Rigid', zorder=5)
    ax.scatter(model.mu_T[0], model.mu_T[1], s=200, c='purple', marker='s', 
              edgecolors='black', linewidths=2, label='Trauma', zorder=5)
    
    ax.set_xlabel('Emotion (x₀)')
    ax.set_ylabel('Memory (x₁)')
    ax.set_title('Basin Geometry (Contour Map)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_therapy_intervention(model, pre_steps=300, therapy_steps=150, figsize=(14, 10)):
    """Plot pre/post therapy intervention effects"""
    model.reset()
    pre_cond, post_cond, x_final = model.run_therapy(pre_steps=pre_steps, therapy_steps=therapy_steps)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Therapy Intervention Protocol (n={model.n})', fontsize=14, fontweight='bold')
    
    V_hist = np.array(model.history['V'])
    lambda_hist = np.array(model.history['lambda'])
    cond_hist = np.array(model.history['cond_g'])
    traj = np.array(model.history['x'])
    
    # Potential energy with intervention marker
    ax = axes[0, 0]
    ax.plot(range(pre_steps), V_hist[:pre_steps], 'b-', linewidth=2, label='Pre-therapy')
    ax.plot(range(pre_steps, len(V_hist)), V_hist[pre_steps:], 'g-', linewidth=2, label='Post-therapy')
    ax.axvline(pre_steps, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Intervention start')
    ax.set_xlabel('Step')
    ax.set_ylabel('Potential V(x)')
    ax.set_title('Energy Landscape: Before & After Therapy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Salience modulation
    ax = axes[0, 1]
    ax.plot(range(pre_steps), lambda_hist[:pre_steps], 'navy', linewidth=1.5, alpha=0.8, label='Pre')
    ax.plot(range(pre_steps, len(lambda_hist)), lambda_hist[pre_steps:], 'lime', linewidth=1.5, alpha=0.8, label='Post')
    ax.axvline(pre_steps, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.fill_between(range(pre_steps, len(lambda_hist)), 0.4, 0.78, alpha=0.1, color='lime', label='Therapy band')
    ax.set_xlabel('Step')
    ax.set_ylabel('Salience λ')
    ax.set_ylim([0, 1.05])
    ax.set_title('Salience Gating: Modulation During Therapy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Condition number (metric anisotropy)
    ax = axes[1, 0]
    ax.semilogy(range(pre_steps), cond_hist[:pre_steps], 'b.', alpha=0.5, label=f'Pre (mean={pre_cond:.2f})')
    ax.semilogy(range(pre_steps, len(cond_hist)), cond_hist[pre_steps:], 'g.', alpha=0.5, label=f'Post (mean={post_cond:.2f})')
    ax.axvline(pre_steps, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('cond(g)')
    
    # Report change in condition number accurately (no "improvement" label)
    cond_change = ((post_cond - pre_cond) / pre_cond) * 100
    direction = "↑ more anisotropic" if cond_change > 0 else "↓ more isotropic"
    ax.set_title(f'Metric Anisotropy: {abs(cond_change):.1f}% {direction}')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # State-space trajectory
    ax = axes[1, 1]
    ax.plot(traj[:pre_steps, 0], traj[:pre_steps, 1], 'b-', alpha=0.6, linewidth=0.8, label='Pre')
    ax.plot(traj[pre_steps:, 0], traj[pre_steps:, 1], 'g-', alpha=0.6, linewidth=0.8, label='Post')
    ax.scatter(model.mu_H[0], model.mu_H[1], s=150, c='green', marker='*', edgecolors='black', 
              linewidths=1.5, label='Healthy', zorder=5)
    ax.scatter(*traj[[0, -1], :2].T, c=['blue', 'lime'], s=100, zorder=4, marker='o')
    ax.set_xlabel('Emotion (x₀)')
    ax.set_ylabel('Memory (x₁)')
    ax.set_title('Basin Migration: Therapy-Induced Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_phase_diagram(model, n_trust=6, n_trauma=6, figsize=(16, 5)):
    """Plot phase diagram in trust vs trauma parameter space"""
    trust_vals = np.linspace(0.3, 0.85, n_trust)
    trauma_vals = np.linspace(1.5, 8.0, n_trauma)
    
    print(f"\nComputing phase diagram ({n_trust}×{n_trauma} grid)...")
    results = model.run_sweep(trust_vals, trauma_vals, runs_per_cell=10, steps=300)
    
    # Extract maps
    H_map = np.zeros((n_trauma, n_trust))
    R_map = np.zeros_like(H_map)
    T_map = np.zeros_like(H_map)
    
    for i, at in enumerate(trauma_vals):
        for j, t0 in enumerate(trust_vals):
            res = results[(t0, at)]
            H_map[i, j] = res['H']
            R_map[i, j] = res['R']
            T_map[i, j] = res['T']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'Phase Diagram: Parameter Space Exploration (n={model.n})', fontsize=14, fontweight='bold')
    
    # Healthy basin
    im0 = axes[0].contourf(trust_vals, trauma_vals, H_map, levels=10, cmap='Greens')
    axes[0].set_xlabel('Trust Base (τ)')
    axes[0].set_ylabel('Trauma Amplitude (w_T)')
    axes[0].set_title('Healthy Basin Probability')
    plt.colorbar(im0, ax=axes[0], label='P(H)')
    
    # Rigid basin
    im1 = axes[1].contourf(trust_vals, trauma_vals, R_map, levels=10, cmap='Reds')
    axes[1].set_xlabel('Trust Base (τ)')
    axes[1].set_ylabel('Trauma Amplitude (w_T)')
    axes[1].set_title('Rigid Basin Probability')
    plt.colorbar(im1, ax=axes[1], label='P(R)')
    
    # Trauma basin
    im2 = axes[2].contourf(trust_vals, trauma_vals, T_map, levels=10, cmap='Purples')
    axes[2].set_xlabel('Trust Base (τ)')
    axes[2].set_ylabel('Trauma Amplitude (w_T)')
    axes[2].set_title('Trauma Basin Probability')
    plt.colorbar(im2, ax=axes[2], label='P(T)')
    
    plt.tight_layout()
    return fig

def plot_metric_geometry(model, n_steps=500):
    """Plot metric properties: condition number, Christoffel norm (geodesic deviation)"""
    model.reset()
    x = np.random.uniform(-0.3, 0.3, model.n)
    
    cond_samples = []
    christoffel_norms = []
    
    for step_idx in range(n_steps):
        x, _, cond = model.step(x)
        cond_samples.append(cond)
        
        if step_idx % 50 == 0:  # Compute Christoffel norm periodically (expensive)
            try:
                Gamma = model.compute_christoffel(x, eps=1e-4)
                # ||Γ|| = sqrt(sum of all components squared) — measures manifold curvature
                christoffel_norm = np.sqrt(np.sum(Gamma**2))
                christoffel_norms.append(christoffel_norm)
            except:
                christoffel_norms.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Metric Geometry Properties (n={model.n})', fontsize=14, fontweight='bold')
    
    # Condition number trajectory
    ax = axes[0]
    ax.semilogy(cond_samples, 'b-', linewidth=1.5, label='cond(g) along trajectory')
    ax.fill_between(range(len(cond_samples)), cond_samples, alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Condition Number cond(g)')
    ax.set_title('Metric Conditioning: Anisotropy Growth')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Christoffel norm (geodesic deviation / manifold curvature)
    ax = axes[1]
    steps_gamma = [i*50 for i in range(len(christoffel_norms))]
    ax.plot(steps_gamma, christoffel_norms, 'mo-', markersize=6, linewidth=2, label='||Γ|| = geodesic deviation')
    ax.fill_between(steps_gamma, christoffel_norms, alpha=0.3, color='magenta')
    ax.set_xlabel('Step (sampled every 50)')
    ax.set_ylabel('Christoffel Norm ||Γ||')
    ax.set_title('Manifold Curvature: Quantified via Christoffel Symbols')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_lyapunov_stability(model, figsize=(12, 8)):
    """Plot Lyapunov eigenvalue distributions across basins and trust levels"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Lyapunov Stability (Discrete Map)', fontsize=14, fontweight='bold')
    
    basins = ['H', 'R']
    trusts = [0.4, 0.8]
    
    for idx, (basin, trust) in enumerate([(b, t) for b in basins for t in trusts]):
        ax = axes[idx // 2, idx % 2]
        
        lya = model.lyapunov_analysis(basin=basin, trust=trust, n_steps=200)
        eigs = lya['eigenvalues']
        max_eig = lya['max_abs_eigenvalue']
        
        # Plot eigenvalue magnitudes (display bar heights = |λ|; title shows spectral radius ρ(J))
        eig_mags = np.abs(eigs.real)
        eig_mags_sorted = np.sort(eig_mags)[::-1]

        colors_list = ['g' if e < 1 else 'r' for e in eig_mags_sorted]
        ax.bar(range(len(eig_mags_sorted)), eig_mags_sorted, color=colors_list, alpha=0.7, edgecolor='black')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Stability boundary (ρ(J)=1)')

        basin_name = {'H': 'Healthy', 'R': 'Rigid'}[basin]
        status = '✓ STABLE' if lya['stable'] else '✗ UNSTABLE'
        ax.set_title(f'{basin_name} Basin (τ={trust})\nρ(J)={max_eig:.4f} {status}')
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('|λ| (Magnitude)')
        ax.set_ylim([0, max(eig_mags_sorted)*1.2])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    plt.tight_layout()
    return fig

def main():
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("VISUALIZATION DASHBOARD: Geometry of Awareness v1.2")
    print("="*70)
    
    # Create output directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Initialize model
    model = GeometryOfAwareness(n=7, seed=42)
    
    # 1. Dynamics trajectory
    print("\n[1] Generating dynamics trajectory...")
    fig1 = plot_dynamics_trajectory(model, n_steps=1000)
    fig1.savefig('visualizations/01_dynamics_trajectory.png', dpi=150, bbox_inches='tight')
    print("    [OK] Saved: visualizations/01_dynamics_trajectory.png")
    
    # 2. Potential landscape
    print("\n[2] Generating potential landscape...")
    fig2 = plot_potential_landscape(model)
    fig2.savefig('visualizations/02_potential_landscape.png', dpi=150, bbox_inches='tight')
    print("    [OK] Saved: visualizations/02_potential_landscape.png")
    
    # 3. Therapy intervention
    print("\n[3] Generating therapy intervention...")
    fig3 = plot_therapy_intervention(model)
    fig3.savefig('visualizations/03_therapy_intervention.png', dpi=150, bbox_inches='tight')
    print("    [OK] Saved: visualizations/03_therapy_intervention.png")
    
    # 4. Phase diagram
    print("\n[4] Generating phase diagram...")
    fig4 = plot_phase_diagram(model, n_trust=6, n_trauma=6)
    fig4.savefig('visualizations/04_phase_diagram.png', dpi=150, bbox_inches='tight')
    print("    [OK] Saved: visualizations/04_phase_diagram.png")
    
    # 5. Metric geometry
    print("\n[5] Generating metric geometry properties...")
    fig5 = plot_metric_geometry(model, n_steps=500)
    fig5.savefig('visualizations/05_metric_geometry.png', dpi=150, bbox_inches='tight')
    print("    [OK] Saved: visualizations/05_metric_geometry.png")
    
    # 6. Lyapunov stability
    print("\n[6] Generating Lyapunov stability analysis...")
    fig6 = plot_lyapunov_stability(model)
    fig6.savefig('visualizations/06_lyapunov_stability.png', dpi=150, bbox_inches='tight')
    print("    [OK] Saved: visualizations/06_lyapunov_stability.png")
    
    print("\nTo view: Open PNG files in Windows Explorer or your image viewer")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
