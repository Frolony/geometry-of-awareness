"""
Visualization Dashboard for Geometry of Awareness v1.3

Generates comprehensive visualizations of:
- Dynamics (phase space, energy, salience, stability)
- Basin geometry (3D plots of potential landscape)
- Metric properties (condition number, eigenvalues, curvature)
- Lyapunov stability (eigenvalue distributions)
- Phase diagrams (trust vs trauma parameter space)
- Signed coherence analysis (negative couplings, inhibitory potentials, v1.3 metrics)
- Numerical summary dashboard (tabular data integration)
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

def plot_signed_coherence_analysis(model, n_steps=800, figsize=(16, 10)):
    """Plot v1.3 signed coherence metrics: negative couplings, inhibitory potential, repulsion dynamics"""
    model.reset()
    x = np.random.uniform(-0.3, 0.3, model.n)
    
    # Track signed metrics over time
    signed_fractions = []
    inhibitory_strengths = []
    
    for step_idx in range(n_steps):
        x, _, _ = model.step(x)
        signed_fractions.append(model.signed_fraction())
        inhibitory_strengths.append(model.inhibitory_strength())
    
    # Get final coherence matrix
    C_final = model.C.copy()
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Signed fraction evolution (% of negative couplings)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(signed_fractions, 'r-', linewidth=2, label='Signed Fractionᵥ1.3')
    ax1.fill_between(range(len(signed_fractions)), signed_fractions, alpha=0.3, color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Fraction Negative')
    ax1.set_title('Signed Fraction: % Inhibitory Couplings')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Inhibitory strength evolution (sum of |C_ij| for negatives)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(inhibitory_strengths, 'orange', linewidth=2, label='Inhibitory Strength')
    ax2.fill_between(range(len(inhibitory_strengths)), inhibitory_strengths, alpha=0.3, color='orange')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Sum of |Cᵢⱼ<0|')
    ax2.set_title('Inhibitory Strength: Total Repulsion Magnitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Potential comparison: V_total vs V_inhib
    ax3 = fig.add_subplot(gs[0, 2])
    V_total = np.array(model.history['V'])
    V_inhib = np.array(model.history['V_inhib'])
    ax3.plot(V_total, 'b-', linewidth=2, alpha=0.7, label='V_total')
    ax3.plot(V_inhib, 'r--', linewidth=2, alpha=0.7, label='V_inhib')
    ax3.fill_between(range(len(V_inhib)), V_inhib, alpha=0.2, color='red')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Potential Energy')
    ax3.set_title('Inhibitory Potential Contribution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Coherence matrix heatmap (initial)
    C_init = np.zeros_like(model.C)
    C_init[0, 2] = C_init[2, 0] = -0.45  # Demo seed
    np.fill_diagonal(C_init, 0.01)
    
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(C_init, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax4.set_title('Initial Coherence C(t=0)\nDemo Seed: C[0,2]=-0.45')
    ax4.set_xlabel('Dimension j')
    ax4.set_ylabel('Dimension i')
    plt.colorbar(im4, ax=ax4, label='C_ij')
    
    # 5. Coherence matrix heatmap (final)
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(C_final, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax5.set_title('Final Coherence C(t=final)\nAfter Learning')
    ax5.set_xlabel('Dimension j')
    ax5.set_ylabel('Dimension i')
    plt.colorbar(im5, ax=ax5, label='C_ij')
    
    # 6. Coherence change (final - initial)
    ax6 = fig.add_subplot(gs[1, 2])
    C_delta = C_final - C_init
    im6 = ax6.imshow(C_delta, cmap='coolwarm', aspect='auto')
    ax6.set_title('Coherence Change ΔC = C(final) - C(init)')
    ax6.set_xlabel('Dimension j')
    ax6.set_ylabel('Dimension i')
    plt.colorbar(im6, ax=ax6, label='ΔC_ij')
    
    # 7. Trajectory with emotion-narrative (repulsion wedge)
    ax7 = fig.add_subplot(gs[2, 0])
    traj = np.array(model.history['x'])
    ax7.plot(traj[:, 0], traj[:, 2], 'b-', alpha=0.6, linewidth=1, label='Trajectory')
    ax7.scatter(traj[0, 0], traj[0, 2], c='green', s=100, marker='o', zorder=5, label='Start')
    ax7.scatter(traj[-1, 0], traj[-1, 2], c='red', s=100, marker='x', zorder=5, label='End')
    ax7.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax7.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax7.set_xlabel('Emotion (x₀)')
    ax7.set_ylabel('Narrative (x₂)')
    ax7.set_title('Emotion-Narrative Coupling\n(C[0,2] Repulsion Wedge)')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # 8. Negative coupling count
    ax8 = fig.add_subplot(gs[2, 1])
    neg_counts = [np.sum(C < 0) for C in [model.history['C'][i] for i in range(0, len(model.history['C']), max(1, len(model.history['C'])//100))]]
    steps_sample = np.linspace(0, n_steps-1, len(neg_counts), dtype=int)
    ax8.plot(steps_sample, neg_counts, 'purple', marker='o', linewidth=2, markersize=4)
    ax8.fill_between(steps_sample, neg_counts, alpha=0.3, color='purple')
    ax8.set_xlabel('Step')
    ax8.set_ylabel('Count of Negative Couplings')
    ax8.set_title('Number of Inhibitory Links Over Time')
    ax8.grid(True, alpha=0.3)
    
    # 9. Statistics box: numerical summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    final_signed_frac = signed_fractions[-1]
    final_inhib_str = inhibitory_strengths[-1]
    final_V_inhib = V_inhib[-1]
    final_C_neg_count = np.sum(C_final < 0)
    final_C_neg_mean = np.mean(C_final[C_final < 0]) if final_C_neg_count > 0 else 0
    
    stats_text = f"""
    ╔═══════════════════════════════╗
    ║   V1.3 NUMERICAL SUMMARY      ║
    ╠═══════════════════════════════╣
    ║ Signed Fraction:    {final_signed_frac:.4f}    ║
    ║ Inhib. Strength:    {final_inhib_str:.4f}    ║
    ║ V_inhib (final):    {final_V_inhib:.4f}    ║
    ║ Neg. Links:         {final_C_neg_count:3d} / {model.n*(model.n-1)//2:3d}    ║
    ║ Mean C_neg:         {final_C_neg_mean:.4f}    ║
    ║ Steps Simulated:    {n_steps}      ║
    ╚═══════════════════════════════╝
    """
    ax9.text(0.5, 0.5, stats_text, fontfamily='monospace', fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    fig.suptitle(f'Signed Coherence Analysis: v1.3 Inhibitory Dynamics (n={model.n})', 
                fontsize=14, fontweight='bold')
    
    return fig

def plot_numerical_dashboard(model, n_steps=500, figsize=(16, 12)):
    """Comprehensive numerical dashboard with multiple metric summaries"""
    model.reset()
    x = np.random.uniform(-0.3, 0.3, model.n)
    
    # Run dynamics and collect comprehensive data
    data = {
        'steps': [],
        'V_total': [],
        'V_H': [], 'V_R': [], 'V_T': [], 'V_inhib': [],
        'lambda': [],
        'cond': [],
        'signed_frac': [],
        'inhib_strength': [],
        'x_norm': [],  # norm of x
        'grad_norm': []  # norm of gradient
    }
    
    for step_idx in range(n_steps):
        # Compute gradient
        grad = np.zeros(model.n)
        eps = 1e-5
        for i in range(model.n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            grad[i] = (model.potential(xp)[0] - model.potential(xm)[0]) / (2*eps)
        
        x, lam, cond = model.step(x)
        V_total, (V_H, V_R, V_T, V_inhib) = model.potential(x)
        
        data['steps'].append(step_idx)
        data['V_total'].append(V_total)
        data['V_H'].append(V_H)
        data['V_R'].append(V_R)
        data['V_T'].append(V_T)
        data['V_inhib'].append(V_inhib)
        data['lambda'].append(lam)
        data['cond'].append(cond)
        data['signed_frac'].append(model.signed_fraction())
        data['inhib_strength'].append(model.inhibitory_strength())
        data['x_norm'].append(np.linalg.norm(x))
        data['grad_norm'].append(np.linalg.norm(grad))
    
    # Convert to numpy arrays for easier handling
    for key in data:
        data[key] = np.array(data[key])
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Row 1: Energy and Forces
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['steps'], data['V_total'], 'k-', linewidth=2, label='V_total')
    ax1.plot(data['steps'], data['V_H'], 'g--', linewidth=1, alpha=0.7, label='V_H')
    ax1.plot(data['steps'], data['V_R'], 'r--', linewidth=1, alpha=0.7, label='V_R')
    ax1.set_ylabel('Potential Component')
    ax1.set_title('Basin Potentials')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data['steps'], data['V_inhib'], 'orange', linewidth=2, label='V_inhib (v1.3)')
    ax2.fill_between(data['steps'], data['V_inhib'], alpha=0.3, color='orange')
    ax2.set_ylabel('Inhibitory Potential')
    ax2.set_title('Repulsion Dynamics (v1.3)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(data['steps'], data['lambda'], 'purple', linewidth=1.5, label='Salience λ')
    ax3.set_ylabel('Salience Gate')
    ax3.set_ylim([0, 1.05])
    ax3.set_title('Learning Rate Modulation')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.semilogy(data['steps'], data['cond'], 'b-', linewidth=1.5, label='κ(g)')
    ax4.set_ylabel('Condition Number')
    ax4.set_title('Metric Anisotropy')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=8)
    
    # Row 2: v1.3 Specific Metrics
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(data['steps'], data['signed_frac'], 'r-', linewidth=2, label='Signed Fraction')
    ax5.fill_between(data['steps'], data['signed_frac'], alpha=0.3, color='red')
    ax5.set_ylabel('Fraction Negative')
    ax5.set_ylim([0, 1])
    ax5.set_title('Negative Coupling Fraction (v1.3)')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(data['steps'], data['inhib_strength'], 'darkred', linewidth=2, label='Inhib. Strength')
    ax6.fill_between(data['steps'], data['inhib_strength'], alpha=0.3, color='darkred')
    ax6.set_ylabel('Sum |C_ij<0|')
    ax6.set_title('Inhibitory Strength (v1.3)')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.plot(data['steps'], data['x_norm'], 'teal', linewidth=1.5, label='||x(t)||')
    ax7.fill_between(data['steps'], data['x_norm'], alpha=0.3, color='teal')
    ax7.set_ylabel('State Norm')
    ax7.set_title('State Space Magnitude')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=8)
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.plot(data['steps'], data['grad_norm'], 'brown', linewidth=1.5, label='||∇V||')
    ax8.fill_between(data['steps'], data['grad_norm'], alpha=0.3, color='brown')
    ax8.set_ylabel('Gradient Magnitude')
    ax8.set_title('Potential Landscape Steepness')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=8)
    
    # Row 3: Combined and Comparative
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9_twin = ax9.twinx()
    ax9.plot(data['steps'], data['V_total'], 'k-', linewidth=2.5, label='V_total', zorder=3)
    ax9_twin.plot(data['steps'], data['signed_frac'], 'r--', linewidth=2, alpha=0.7, label='Signed Frac', zorder=2)
    ax9.set_xlabel('Step')
    ax9.set_ylabel('Total Potential V(x)', color='k')
    ax9_twin.set_ylabel('Signed Fraction', color='r')
    ax9.set_title('Potential vs Inhibition Correlation')
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(axis='y', labelcolor='k')
    ax9_twin.tick_params(axis='y', labelcolor='r')
    lines1, labels1 = ax9.get_legend_handles_labels()
    lines2, labels2 = ax9_twin.get_legend_handles_labels()
    ax9.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper right')
    
    ax10 = fig.add_subplot(gs[2, 2:4])
    # Create numerical summary table
    ax10.axis('off')
    
    # Compute aggregate statistics
    summary_stats = {
        'Mean V_total': np.mean(data['V_total']),
        'Final Signed Frac': data['signed_frac'][-1],
        'Mean Inhib Strength': np.mean(data['inhib_strength']),
        'Max Salience': np.max(data['lambda']),
        'Mean κ(g)': np.mean(data['cond']),
        'Total V_inhib': np.sum(data['V_inhib']),
        'Mean ||x||': np.mean(data['x_norm']),
        'Mean ||∇V||': np.mean(data['grad_norm']),
        'Final ||x||': data['x_norm'][-1],
        'Basin Stability': 'Valid' if data['V_total'][-1] < np.mean(data['V_total']) else 'Equilibrating'
    }
    
    table_data = []
    for key, val in summary_stats.items():
        if isinstance(val, (int, np.integer)):
            table_data.append([key, f"{val}"])
        elif isinstance(val, str):
            table_data.append([key, val])
        else:
            table_data.append([key, f"{val:.6f}"])
    
    table = ax10.table(cellText=table_data, colLabels=['Metric', 'Value'],
                      cellLoc='left', loc='center', 
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    ax10.set_title('Numerical Summary Statistics', fontweight='bold', fontsize=11, pad=20)
    
    fig.suptitle(f'Numerical Dashboard: Integrated v1.3 Metrics (n={model.n}, {n_steps} steps)',
                fontsize=14, fontweight='bold')
    
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
    print("VISUALIZATION DASHBOARD: Geometry of Awareness v1.3")
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
    
    # 7. Signed coherence analysis (v1.3)
    print("\n[7] Generating signed coherence analysis (v1.3)...")
    fig7 = plot_signed_coherence_analysis(model, n_steps=800)
    fig7.savefig('visualizations/07_signed_coherence_v13.png', dpi=150, bbox_inches='tight')
    print("    [OK] Saved: visualizations/07_signed_coherence_v13.png")
    
    # 8. Numerical dashboard (v1.3)
    print("\n[8] Generating numerical dashboard (v1.3)...")
    fig8 = plot_numerical_dashboard(model, n_steps=500)
    fig8.savefig('visualizations/08_numerical_dashboard_v13.png', dpi=150, bbox_inches='tight')
    print("    [OK] Saved: visualizations/08_numerical_dashboard_v13.png")
    
    print("\nTo view: Open PNG files in Windows Explorer or your image viewer")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
