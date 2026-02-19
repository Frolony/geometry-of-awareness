"""Phase diagram sweep: trust vs trauma parameter space"""
import numpy as np
import matplotlib.pyplot as plt
from geometry_of_awareness import GeometryOfAwareness

# Sweep over trust and trauma amplitude parameters
trust_vals = np.linspace(0.3, 0.85, 6)
trauma_vals = np.linspace(1.5, 8.0, 6)

model = GeometryOfAwareness()
results = model.run_sweep(trust_vals, trauma_vals, runs_per_cell=15, steps=800)

# Extract phase diagram
H_map = np.zeros((len(trauma_vals), len(trust_vals)))
R_map = np.zeros_like(H_map)
T_map = np.zeros_like(H_map)
cond_map = np.zeros_like(H_map)
core_C_map = np.zeros_like(H_map)

for i, at in enumerate(trauma_vals):
    for j, t0 in enumerate(trust_vals):
        res = results[(t0, at)]
        H_map[i, j] = res['H']
        R_map[i, j] = res['R']
        T_map[i, j] = res['T']
        cond_map[i, j] = res['cond_mean']
        core_C_map[i, j] = res['core_C']

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Healthy basin
im0 = axes[0].contourf(trust_vals, trauma_vals, H_map, levels=10, cmap='Greens')
axes[0].set_xlabel('Trust base'); axes[0].set_ylabel('Trauma amplitude')
axes[0].set_title('Healthy basin probability')
plt.colorbar(im0, ax=axes[0])

# Rigid basin
im1 = axes[1].contourf(trust_vals, trauma_vals, R_map, levels=10, cmap='Reds')
axes[1].set_xlabel('Trust base'); axes[1].set_ylabel('Trauma amplitude')
axes[1].set_title('Rigid basin probability')
plt.colorbar(im1, ax=axes[1])

# Metric condition number
im2 = axes[2].contourf(trust_vals, trauma_vals, cond_map, levels=10, cmap='plasma')
axes[2].set_xlabel('Trust base'); axes[2].set_ylabel('Trauma amplitude')
axes[2].set_title('Mean metric condition cond(g)')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('phase_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Phase diagram computed: {len(trust_vals)}×{len(trauma_vals)} grid, 15 runs per cell")
print(f"Healthy basin range: {H_map.min():.2f}–{H_map.max():.2f}")
print(f"Metric cond(g) range: {cond_map.min():.2f}–{cond_map.max():.2f}")
