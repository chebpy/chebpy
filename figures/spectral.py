import sys
sys.path.insert(0, '../src')
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.op_discretization import OpDiscretization
from chebpy.utilities import Domain, Interval
from chebpy.chebtech import Chebtech
from chebpy.bndfun import Bndfun

DOMAIN = [-10, 5]
Ai_l, Ai_r = airy(-10)[0], airy(5)[0]

# Uses linop to avoid adaptive solver
def solve(N):
    L = LinOp(coeffs=[chebfun(lambda x: -x, DOMAIN), None, chebfun(lambda x: 1+0*x, DOMAIN)], domain=Domain(DOMAIN), diff_order=2)
    L.lbc, L.rbc = Ai_l, Ai_r
    L.prepare_domain()
    A, b = L.assemble_system(OpDiscretization.build_discretization(L, n=N, bc_enforcement='append'))
    u, *_ = np.linalg.lstsq(A.toarray(), np.array(b).flatten(), rcond=None)
    return Bndfun(Chebtech.initvalues(u), Interval(*DOMAIN))

x_fine, x_test = np.linspace(*DOMAIN, 500), np.linspace(*DOMAIN, 2000)
exact = lambda x: airy(x)[0]
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# (a) Solution
ax[0,0].plot(x_fine, exact(x_fine), 'k-', lw=2, label='Exact Ai(x)')
for N, c, s in [(20,'blue','--'), (32,'red','-.'), (50,'green',':')]:
    ax[0,0].plot(x_fine, solve(N)(x_fine), color=c, ls=s, lw=1.5, label=f'N={N}')

ax[0,0].set(xlabel='x', ylabel='u(x)', title=r"(a) Solution of $u'' = xu$")
ax[0,0].legend()
ax[0,0].grid(alpha=0.3)

# (b) Convergence
Ns = np.array([16,20,24,28,32,36,40,45,50,55,60])
errs = []
for N in Ns:
    errs.append(max(np.max(np.abs(exact(x_test) - solve(N)(x_test))), 1e-16))
errs = np.array(errs)
fit = errs > 1e-12
c = -np.polyfit(Ns[fit], np.log(errs[fit]), 1)[0]
ax[0,1].semilogy(Ns, errs, 'bo-', lw=2, ms=6, label='Spectral error')
ax[0,1].semilogy(Ns, errs[0]*np.exp(-c*(Ns-Ns[0])), 'r--', lw=1.5, alpha=0.7, label=rf'$\propto e^{{-{c:.2f}N}}$')
ax[0,1].axhline(2.2e-16, color='gray', ls=':', alpha=0.7, label=r'Machine $\epsilon$')
ax[0,1].set(xlabel='N', ylabel='Max error', title='(b) Exponential Convergence', ylim=[1e-16,1e1])
ax[0,1].legend()
ax[0,1].grid(alpha=0.3, which='both')

# (c) Coefficients
_, u_vals = None, solve(80)
coeffs = np.fft.ifft(np.concatenate([u_vals(np.cos(np.pi*np.arange(81)/80)*7.5-2.5),
                                      u_vals(np.cos(np.pi*np.arange(81)/80)*7.5-2.5)[-2:0:-1]])).real[:81]
coeffs[1:-1] *= 2
idx = np.arange(len(coeffs))
valid = np.abs(coeffs) > 1e-17
ax[1,0].semilogy(idx[valid], np.abs(coeffs)[valid], 'bo-', ms=4)
ax[1,0].axhline(2.2e-16, color='gray', ls=':', alpha=0.7, label=r'Machine $\epsilon$')
ax[1,0].set(xlabel='Coefficient index $k$', ylabel='$|a_k|$', title='(c) Chebyshev Coefficient Decay')
ax[1,0].legend()
ax[1,0].grid(alpha=0.3, which='both')

# (d) Spectral vs FD
ax[1,1].loglog(Ns, errs, 'bo-', lw=2, ms=6, label='Spectral')
ax[1,1].loglog(Ns, errs[0]*(Ns[0]/Ns)**2, 'r--', lw=1.5, alpha=0.7, label=r'$O(N^{-2})$ FD')
ax[1,1].axhline(2.2e-16, color='gray', ls=':', alpha=0.7, label=r'Machine $\epsilon$')
ax[1,1].set(xlabel='N', ylabel='Max error', title='(d) Spectral vs Finite Difference', ylim=[1e-16,1e1])
ax[1,1].legend()
ax[1,1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('spectral.png')
