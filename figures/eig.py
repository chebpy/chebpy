import sys
sys.path.insert(0, '../src')
import numpy as np, matplotlib.pyplot as plt
from chebpy import chebop

def qho():
    L = chebop([-6, 6])
    L.op = lambda x, u: -u.diff(2) + x**2 * u
    L.lbc = 0
    L.rbc = 0
    L.n_current = 80
    return L

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# (a) Eigenfunctions
eigs, efuns = qho().eigs(k=5)
x = np.linspace(-6, 6, 500)
for i, (v, ef) in enumerate(zip(eigs, efuns)):
    if ef: 
        y = ef(x)
        y = y/np.max(np.abs(y))
        y = -y if y[len(y)//2] < 0 else y
    ax[0].plot(x, y, lw=1.8, label=f'$\\psi_{i}$: $\\lambda_{i}$={v:.4f} (theory: {2*i+1})')
ax[0].set(xlabel='$x$', ylabel=r'$\psi_k(x)$', title='(a) Hermite function eigenfunctions', xlim=[-6,6])
ax[0].legend(fontsize=9)
ax[0].grid(alpha=0.3)
ax[0].axhline(0, color='k', lw=0.5)
ax[0].text(0.02, 0.98, r"$-\psi'' + x^2\psi = \lambda\psi$", transform=ax[0].transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (b) Eigenvalue accuracy
eigs10, _ = qho().eigs(k=10)
theory = 2*np.arange(10)+1
ax[1].scatter(theory, eigs10, s=80, alpha=0.8, c='blue', edgecolors='k', lw=1, label=r'Computed $\lambda_k$', zorder=3)
ax[1].plot([0, 21], [0, 21], 'r--', lw=1.5, label='$y=x$', zorder=2)
ax[1].set(xlabel=r'Theoretical $\lambda_k = 2k+1$', ylabel='Computed', title='(b) Eigenvalue accuracy', aspect='equal', xlim=[0,21], ylim=[0,21])
ax[1].legend()
ax[1].grid(alpha=0.3)
ax[1].text(0.05, 0.95, f'Max error: {np.max(np.abs(eigs10-theory)):.2e}\n$n=80$', transform=ax[1].transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('eig.png')
