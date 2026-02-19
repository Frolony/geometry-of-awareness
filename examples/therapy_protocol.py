"""Therapy protocol example: pre/post intervention comparison"""
import numpy as np
import matplotlib.pyplot as plt
from geometry_of_awareness import GeometryOfAwareness

# Run therapy intervention
model = GeometryOfAwareness(trust_base=0.55, seed=42)
pre_cond, post_cond, x_final = model.run_therapy(pre_steps=400, therapy_steps=240, trust_lift=0.18)

# Extract metrics
V_history = np.array(model.history['V'])
lambda_history = np.array(model.history['lambda'])
cond_history = np.array(model.history['cond_g'])
trajectory = np.array(model.history['x'])

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Split pre/post
pre_end = 400
ax = axes[0, 0]
ax.plot(range(pre_end), V_history[:pre_end], 'b-', label='Pre-therapy', linewidth=1.5)
ax.plot(range(pre_end, len(V_history)), V_history[pre_end:], 'g-', label='Post-therapy', linewidth=1.5)
ax.axvline(pre_end, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Step'); ax.set_ylabel('Potential V')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('Therapy effect on potential')

# Salience modulation
ax = axes[0, 1]
ax.plot(range(pre_end), lambda_history[:pre_end], 'navy', label='Pre', linewidth=1.5)
ax.plot(range(pre_end, len(lambda_history)), lambda_history[pre_end:], 'lime', label='Post', linewidth=1.5)
ax.axvline(pre_end, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Step'); ax.set_ylabel('Salience λ')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('Salience during intervention')

# Metric condition number comparison
ax = axes[1, 0]
pre_cond_vals = cond_history[:pre_end]
post_cond_vals = cond_history[pre_end:]
ax.semilogy(range(pre_end), pre_cond_vals, 'b.', alpha=0.5, label=f'Pre (mean={pre_cond:.2f})')
ax.semilogy(range(pre_end, len(cond_history)), post_cond_vals, 'g.', alpha=0.5, label=f'Post (mean={post_cond:.2f})')
ax.axvline(pre_end, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Step'); ax.set_ylabel('cond(g)')
ax.legend(); ax.grid(True, alpha=0.3, which='both')
ax.set_title('Metric stability improvement')

# 3D trajectory (E, M, B)
ax = axes[1, 1]
ax.plot(trajectory[:pre_end, 0], trajectory[:pre_end, 1], 'b-', alpha=0.6, linewidth=0.8, label='Pre')
ax.plot(trajectory[pre_end:, 0], trajectory[pre_end:, 1], 'g-', alpha=0.6, linewidth=0.8, label='Post')
ax.scatter(*trajectory[[0, -1], :2].T, c=['blue', 'lime'], s=100, zorder=5, marker='o')
ax.set_xlabel('Emotion'); ax.set_ylabel('Memory')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('State space shift')

plt.tight_layout()
plt.savefig('therapy_protocol.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Therapy effect: cond(g) improved from {pre_cond:.4f} → {post_cond:.4f} ({100*(pre_cond-post_cond)/pre_cond:.1f}%)")
print(f"Final state after therapy: {x_final}")
