"""Basic simulation example: 1000-step trajectory through awareness manifold"""
import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
from geometry_of_awareness import GeometryOfAwareness

# Initialize and run simulation
model = GeometryOfAwareness(trust_base=0.65, seed=42)
x = np.random.uniform(-0.3, 0.3, model.n)

for step in range(1000):
    x, lam, cond = model.step(x)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Trajectory in (Emotion, Memory, Belief) space
traj = np.array(model.history['x'])
axes[0, 0].plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.6, linewidth=0.5)
axes[0, 0].scatter(*traj[[0, -1], :2].T, c=['g', 'r'], s=100, zorder=5, label=['Start', 'End'])
axes[0, 0].set_xlabel('Emotion'); axes[0, 0].set_ylabel('Memory')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Phase-space trajectory')

# Potential energy
axes[0, 1].plot(model.history['V'], 'k-', linewidth=1)
axes[0, 1].set_xlabel('Step'); axes[0, 1].set_ylabel('Total Potential V')
axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_title('Energy landscape')

# Salience (learning rate modulation)
axes[1, 0].plot(model.history['lambda'], 'purple', linewidth=1, alpha=0.7)
axes[1, 0].set_xlabel('Step'); axes[1, 0].set_ylabel('Salience λ')
axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_title('Salience gating')

# Metric condition number
axes[1, 1].semilogy(model.history['cond_g'], 'orange', linewidth=1)
axes[1, 1].set_xlabel('Step'); axes[1, 1].set_ylabel('cond(g)')
axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_title('Metric stability')

plt.tight_layout()
plt.savefig('basic_simulation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Final state: {x}")
print(f"Metric condition: {model.get_condition_number():.3f}")
