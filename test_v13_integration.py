"""
Integration test for v1.3: Verify all major features
"""
import numpy as np
from geometry_of_awareness import GeometryOfAwareness

print('='*70)
print('V1.3 INTEGRATION TEST')
print('='*70)

# Test 1: n=7
g7 = GeometryOfAwareness(n=7)
assert g7.n == 7, 'n=7 failed'
assert g7.C[0,2] == -0.45, 'Demo seed failed'
x = g7.mu_H + 0.01*np.random.randn(7)
for _ in range(10):
    x, l, c = g7.step(x)
assert np.isfinite(x).all(), 'Step x not finite'
assert np.isfinite(g7.potential(x)[0]), 'Potential not finite'
sig_frac = g7.signed_fraction()
inhib = g7.inhibitory_strength()
assert 0 <= sig_frac <= 1, 'signed_frac out of range'
assert inhib >= 0, 'inhib_strength negative'
print(f'✓ n=7: signed_frac={sig_frac:.4f}, inhib_strength={inhib:.4f}')

# Test 2: n=15
g15 = GeometryOfAwareness(n=15)
assert g15.n == 15, 'n=15 failed'
x = g15.mu_H + 0.01*np.random.randn(15)
for _ in range(10):
    x, l, c = g15.step(x)
assert np.isfinite(x).all(), 'n=15 step x not finite'
print(f'✓ n=15: step OK')

# Test 3: Jacobian computation
J = g7.compute_jacobian(g7.mu_H)
rho_J = np.max(np.abs(np.linalg.eigvals(J)))
assert rho_J < 2.0, f'Jacobian eigenvalue too large: {rho_J}'
print(f'✓ Jacobian: rho(J)={rho_J:.4f} (should be < 2.0)')

# Test 4: Lyapunov analysis
result = g7.lyapunov_analysis(basin='H', trust=0.7)
assert 'eigenvalues' in result, 'Lyapunov missing eigenvalues'
assert result['stable'] or result['max_abs_eigenvalue'] < 1.05, 'Basin unstable'
print(f'✓ Lyapunov: rho(J)={result["max_abs_eigenvalue"]:.4f} at H basin')

# Test 5: Metric SPD
eigs_g = np.linalg.eigvalsh(g7.g)
assert np.all(eigs_g > 0), 'Metric not SPD'
print(f'✓ Metric SPD: lambda_min={np.min(eigs_g):.4f}, lambda_max={np.max(eigs_g):.4f}')

# Test 6: Christoffel symbols
gamma = g7.compute_christoffel(g7.mu_H)
assert np.isfinite(gamma).all(), 'Christoffel not finite'
christoffel_norm = np.sum(gamma**2)
print(f'✓ Christoffel: shape {gamma.shape}, ||Gamma||^2={christoffel_norm:.6f}')

# Test 7: Therapy protocol
pre_cond, post_cond, x_final = g7.run_therapy(pre_steps=50, therapy_steps=50)
print(f'✓ Therapy: pre_cond={pre_cond:.4f}, post_cond={post_cond:.4f}')

# Test 8: Reset and reuse
g7.reset(trust_base=0.8, w_T=5.0)
assert g7.C[0,2] == -0.45, 'Reset did not restore demo seed'
assert g7.trust_base == 0.8, 'Reset trust failed'
assert g7.w_T == 5.0, 'Reset w_T failed'
print(f'✓ Reset: trust_base={g7.trust_base}, w_T={g7.w_T}')

print('='*70)
print('ALL INTEGRATION TESTS PASSED')
print('='*70)
