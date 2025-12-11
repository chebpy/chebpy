import sys
sys.path.insert(0, '../src')
import numpy as np
import matplotlib.pyplot as plt
from chebpy import chebfun, chebop
from chebpy.linop import LinOp
from chebpy.utilities import Domain
from chebpy.op_discretization import OpDiscretization

def spy(ax, A, title):
    M = A.toarray() if hasattr(A, 'toarray') else A
    ax.spy(M, markersize=2, color='black')
    m, n = M.shape
    ax.set(title=title, xlabel='Column', ylabel='Row')
    ax.text(0.95, 0.05, f'{m}x{n}\n{np.count_nonzero(np.abs(M)>1e-14)/M.size*100:.0f}% dense', transform=ax.transAxes, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)

fig, ax = plt.subplots(1, 3, figsize=(10, 4.5))
n = 31
domain = Domain([0, 1])

# (a) u'' with Dirichlet BCs
N1 = chebop(op=lambda u: u.diff(2), domain=domain)
N1.lbc = N1.rbc = 0
L1 = N1.to_linop()
L1.prepare_domain()
A1, _ = L1.assemble_system(OpDiscretization.build_discretization(L1, n, bc_enforcement='append'))
spy(ax[0], A1, r"(a) $u''$, $u(0)=u(1)=0$")

# (b) u'''' with clamped BCs
N2 = chebop(op=lambda u: u.diff(4), domain=domain)
N2.lbc = N2.rbc = [0, 0]
L2 = N2.to_linop()
L2.prepare_domain()
A2, _ = L2.assemble_system(OpDiscretization.build_discretization(L2, n, bc_enforcement='append'))
spy(ax[1], A2, r"(b) $u''''$, $u(0)=u'(0)=u(1)=u'(1)=0$")

# (c) Condition numbers by order
orders, conds = [1, 2, 3, 4], []
for k in orders:
    coeffs = [None] * (k+1)
    coeffs[k] = chebfun(lambda x: 1+0*x, Domain([-1,1]))
    L = LinOp(coeffs=coeffs, domain=Domain([-1,1]), diff_order=k)
    L.lbc = 0 if k==1 else ([0,0] if k>=3 else 0)
    L.rbc = 0 if k<=2 else ([0,0] if k==4 else 0)
    L.prepare_domain()
    A, _ = L.assemble_system(OpDiscretization.build_discretization(L, n, bc_enforcement='append'))
    conds.append(np.linalg.cond(A.toarray()))

ax[2].semilogy(orders, conds, 'o-', lw=2, ms=8, color='#d62728')
ax[2].set(xlabel='Derivative order $k$', ylabel=r'$\kappa(A)$', title=r'(c) Conditioning ($n=32$)', xticks=orders)
ax[2].grid(alpha=0.3, ls='--')

plt.tight_layout()
plt.savefig('disc_mat.png')