"""Test Jacobian stability around equilibrium basins"""
import numpy as np
from geometry_of_awareness import GeometryOfAwareness

# Test
model = GeometryOfAwareness(seed=42)
model.trust_base = 0.8  # high trust

# Around Healthy
J_H_high = model.compute_jacobian(model.mu_H, trust=0.8)
eigs_H_high = np.linalg.eigvals(J_H_high)
max_abs_H_high = np.max(np.abs(eigs_H_high))

print("Healthy basin, high trust:")
print("Max |eigenvalue|:", max_abs_H_high)
print("All inside unit circle?", max_abs_H_high < 1)

# Low trust
model.trust_base = 0.4
J_H_low = model.compute_jacobian(model.mu_H, trust=0.4)
eigs_H_low = np.linalg.eigvals(J_H_low)
max_abs_H_low = np.max(np.abs(eigs_H_low))

print("\nHealthy basin, low trust:")
print("Max |eigenvalue|:", max_abs_H_low)

# Rigid
model.trust_base = 0.4
J_R_low = model.compute_jacobian(model.mu_R, trust=0.4)
eigs_R_low = np.linalg.eigvals(J_R_low)
max_abs_R_low = np.max(np.abs(eigs_R_low))

print("\nRigid basin, low trust:")
print("Max |eigenvalue|:", max_abs_R_low)
print("All inside unit circle?", max_abs_R_low < 1)
