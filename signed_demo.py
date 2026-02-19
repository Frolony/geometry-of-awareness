"""
signed_demo.py — v1.3 Demo: Emotion-Narrative Repulsion

Demonstrates the v1.3 signed coherence framework:
- Initial negative coupling C[0,2] = -0.45 (Emotion ↔ Narrative)
- Tracks repulsion wedge in state space
- Shows metric, inhibitory strength, and basin transitions
"""

import numpy as np
import matplotlib.pyplot as plt
from geometry_of_awareness import GeometryOfAwareness

print("=" * 70)
print("v1.3 SIGNED COHERENCE DEMO: Emotion-Narrative Repulsion")
print("=" * 70)

# Initialize v1.3 system
g = GeometryOfAwareness(n=7, seed=123)

print(f"\nInitial state:")
print(f"  C[0,2] (Emotion ↔ Narrative): {g.C[0,2]:.4f}")
print(f"  Signed fraction: {g.signed_fraction():.4f}")
print(f"  Inhibitory strength: {g.inhibitory_strength():.4f}")

# Run 600 steps starting from high emotion + high narrative
x0 = g.mu_H.copy()
x0[0] += 0.3  # boost emotion
x0[2] += 0.2  # boost narrative

x = x0.copy()
n_steps = 600

print(f"\nRunning {n_steps} steps from elevated Emotion+Narrative state...")

for step_idx in range(n_steps):
    x, lambda_t, cond_g = g.step(x, trust=0.75)
    
    # Log every 60 steps
    if (step_idx + 1) % 60 == 0:
        V_total, (VH, VR, VT, V_inhib) = g.potential(x)
        sig_frac = g.signed_fraction()
        inhib_str = g.inhibitory_strength()
        print(f"Step {step_idx+1:3d}: x0={x[0]:+7.3f}, x2={x[2]:+7.3f}, "
              f"C[0,2]={g.C[0,2]:+7.4f}, sig_frac={sig_frac:.4f}, "
              f"inhib={inhib_str:.4f}, V={V_total:.3f}, V_inhib={V_inhib:.4f}")

print(f"\n{'='*70}")
print(f"FINAL STATE:")
print(f"  Position: x0={x[0]:+.4f} (Emotion), x2={x[2]:+.4f} (Narrative)")
V_final, (VH, VR, VT, V_inhib) = g.potential(x)
print(f"  Potential: V_total={V_final:.4f}, V_H={VH:.4f}, V_R={VR:.4f}, V_T={VT:.4f}, V_inhib={V_inhib:.4f}")
print(f"  Couplings: C[0,2]={g.C[0,2]:+.4f}, signed_frac={g.signed_fraction():.4f}, inhib_strength={g.inhibitory_strength():.4f}")
print(f"  Metric: condition(g)={np.linalg.cond(g.g):.2f}")

# Classification
if VH + 0.15 < min(VR, VT):
    outcome = "HEALTHY"
elif VR + 0.15 < min(VH, VT):
    outcome = "RIGID"
elif VT > max(VH, VR) + 0.5:
    outcome = "TRAUMA"
else:
    outcome = "LIMINAL"

print(f"  Basin: {outcome}")
print(f"{'='*70}")

# Plot trajectories
hist_x = np.array(g.history['x'])
hist_C02 = np.array([C[0, 2] for C in g.history['C']])
hist_V_inhib = np.array(g.history['V_inhib'])
hist_cond = np.array(g.history['cond_g'])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Emotion vs Narrative (repulsion wedge)
ax = axes[0, 0]
ax.plot(hist_x[:, 0], hist_x[:, 2], 'b-', alpha=0.6, linewidth=1.5, label='Trajectory')
ax.scatter([x0[0]], [x0[2]], c='green', s=100, marker='o', zorder=5, label='Start')
ax.scatter([x[0]], [x[2]], c='red', s=100, marker='x', zorder=5, linewidth=2, label='End')
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Emotion (x₀)')
ax.set_ylabel('Narrative (x₂)')
ax.set_title('Emotion-Narrative Repulsion Wedge')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: C[0,2] evolution (negative coupling)
ax = axes[0, 1]
ax.plot(hist_C02, 'r-', linewidth=2, label='C₀₂ (inhibitory)')
ax.axhline(-0.45, color='gray', linestyle='--', alpha=0.5, label='Init = -0.45')
ax.set_xlabel('Step')
ax.set_ylabel('C₀₂ value')
ax.set_title('Emotion-Narrative Coupling (Signed) Over Time')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Inhibitory potential contribution
ax = axes[1, 0]
ax.plot(hist_V_inhib, 'orange', linewidth=2, label='V_inhib')
ax.set_xlabel('Step')
ax.set_ylabel('V_inhib value')
ax.set_title('Inhibitory Potential: Impact of Negative Couplings')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Metric condition number
ax = axes[1, 1]
ax.plot(hist_cond, 'purple', linewidth=2, label='κ(g)')
ax.axhline(np.mean(hist_cond[-50:]), color='purple', linestyle='--', alpha=0.5, label='Mean (final 50)')
ax.set_xlabel('Step')
ax.set_ylabel('Condition Number κ(g)')
ax.set_title('Metric Anisotropy Over Time')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('signed_coherence_demo.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: signed_coherence_demo.png")

print("\nDEMO COMPLETE: v1.3 signed coherence and inhibitory potentials")
print(f"Key finding: Emotion-Narrative repulsion {g.C[0,2]:.4f} ")
print(f"  constrains co-activation, forcing dissociation.",)
