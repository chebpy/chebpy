"""Marimo notebook: Variance Swap Replication with Calls and Puts.

Demonstrates how ChebPy Chebfuns and quasimatrices can replicate
the log-contract payoff underlying a variance swap using European call
and put options, following the Carr-Madan static replication approach.
"""

# /// script
# dependencies = ["marimo==0.18.4", "chebfun", "seaborn"]
# requires-python = ">=3.13"
#
# [tool.uv.sources.chebfun]
# path = "../../.."
# editable = true
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    mpl.rc("figure", figsize=(9, 5), dpi=100)


@app.cell(hide_code=True)
def _():
    from chebpy import Quasimatrix, chebfun

    return Quasimatrix, chebfun


@app.cell(hide_code=True)
def _():
    from math import erfc as _erfc

    _verfc = np.vectorize(_erfc)

    def norm_cdf(x):
        """Standard normal CDF (no scipy needed)."""
        return 0.5 * _verfc(-np.asarray(x, dtype=float) / np.sqrt(2))

    return (norm_cdf,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Variance Swap Replication

    A **variance swap** pays the difference between realised variance
    and a fixed strike $K_{\text{var}}$.  A celebrated result of Carr
    and Madan (1998) shows that the fair strike can be computed from a
    static portfolio of out-of-the-money European calls and puts:

    $$
    K_{\text{var}} = \frac{2}{T}\!\left[\int_0^{F} \frac{P(K)}{K^2}\,dK + \int_F^{\infty} \frac{C(K)}{K^2}\,dK\right]
    $$

    where $F$ is the forward price, $T$ the maturity, and $C(K)$, $P(K)$
    are European call and put prices as functions of strike.

    The key identity underlying this formula is a **payoff decomposition**:

    $$
    -\log\!\frac{S_T}{F} = -\frac{S_T - F}{F} + \int_0^{F} \frac{(K - S_T)^+}{K^2}\,dK + \int_F^{\infty} \frac{(S_T - K)^+}{K^2}\,dK
    $$

    This notebook uses ChebPy **Chebfuns** and **quasimatrices** to
    turn these integrals into exact arithmetic, drawing on the same
    hat-function fitting strategy demonstrated in the quasimatrix
    notebook.

    > P. Carr & D. Madan, "Towards a theory of volatility trading",
    > *Volatility: New Estimation Techniques for Pricing Derivatives*,
    > Risk Books, 1998.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Black–Scholes option prices as Chebfuns

    We begin by constructing the Black-Scholes call and put prices as
    Chebfun objects — smooth functions of the strike $K$ on a truncated
    domain $[K_{\min}, K_{\max}]$.

    | Parameter | Value |
    |-----------|-------|
    | Spot $S_0$ | 100 |
    | Forward $F$ | 100 (zero rates) |
    | Volatility $\sigma$ | 20 % |
    | Maturity $T$ | 1 year |
    """)
    return


@app.cell(hide_code=True)
def _():
    S0 = 100.0
    F = 100.0
    sigma = 0.20
    T = 1.0
    return F, S0, T, sigma


@app.cell(hide_code=True)
def _(F, S0, T, chebfun, norm_cdf, sigma):
    K_lo, K_hi = 20.0, 300.0

    def _bs_call(K):
        K = np.asarray(K, dtype=float)
        d1 = (np.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm_cdf(d1) - K * norm_cdf(d2)

    def _bs_put(K):
        K = np.asarray(K, dtype=float)
        d1 = (np.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * norm_cdf(-d2) - S0 * norm_cdf(-d1)

    C = chebfun(_bs_call, [F, K_hi])
    P = chebfun(_bs_put, [K_lo, F])

    print(f"Call C(K) on [{F}, {K_hi}]  — length {len(C)}")
    print(f"Put  P(K) on [{K_lo}, {F}] — length {len(P)}")
    return C, K_hi, K_lo, P


@app.cell(hide_code=True)
def _(C, F, K_hi, K_lo, P):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _kp = np.linspace(K_lo, F, 300)
    _ax1.plot(_kp, P(_kp), linewidth=2, color="C3")
    _ax1.set_title("OTM Put Price $P(K)$")
    _ax1.set_xlabel("Strike $K$")
    _ax1.set_ylabel("Price")
    _ax1.grid(True)

    _kc = np.linspace(F, K_hi, 300)
    _ax2.plot(_kc, C(_kc), linewidth=2, color="C0")
    _ax2.set_title("OTM Call Price $C(K)$")
    _ax2.set_xlabel("Strike $K$")
    _ax2.grid(True)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. The Carr–Madan integral

    The fair variance strike is

    $$
    K_{\text{var}} = \frac{2}{T}\!\left[\int_{K_{\min}}^{F} \frac{P(K)}{K^2}\,dK + \int_F^{K_{\max}} \frac{C(K)}{K^2}\,dK\right]
    $$

    Since $C(K)$ and $P(K)$ are Chebfuns, dividing by the identity
    function $K^2$ and calling `.sum()` evaluates each integral to
    machine precision.
    """)
    return


@app.cell(hide_code=True)
def _(C, F, K_hi, K_lo, P, T, chebfun, sigma):
    K_put = chebfun("x", [K_lo, F])
    K_call = chebfun("x", [F, K_hi])

    put_integrand = P / K_put**2
    call_integrand = C / K_call**2

    I_put = put_integrand.sum()
    I_call = call_integrand.sum()

    K_var = (2.0 / T) * (I_put + I_call)
    exact = sigma**2

    print(f"Put  integral : {float(I_put):.10f}")
    print(f"Call integral : {float(I_call):.10f}")
    print(f"K_var (ChebPy): {float(K_var):.10f}")
    print(f"σ² (exact)    : {exact:.10f}")
    print(f"Relative error: {abs(float(K_var) - exact) / exact:.2e}")
    return call_integrand, put_integrand


@app.cell(hide_code=True)
def _(F, K_hi, K_lo, call_integrand, put_integrand):
    _fig, _ax = plt.subplots(figsize=(10, 4))

    _kp = np.linspace(K_lo + 1, F, 300)
    _kc = np.linspace(F, K_hi, 300)
    _ax.fill_between(_kp, put_integrand(_kp), alpha=0.3, color="C3", label="$P(K)/K^2$  (puts)")
    _ax.plot(_kp, put_integrand(_kp), color="C3", linewidth=2)
    _ax.fill_between(_kc, call_integrand(_kc), alpha=0.3, color="C0", label="$C(K)/K^2$  (calls)")
    _ax.plot(_kc, call_integrand(_kc), color="C0", linewidth=2)
    _ax.axvline(F, color="k", linestyle="--", linewidth=1, label="$F$")
    _ax.set_xlabel("Strike $K$")
    _ax.set_ylabel("Integrand")
    _ax.set_title("Carr–Madan integrand: option price / $K^2$")
    _ax.legend()
    _ax.grid(True)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Payoff replication — the hat-function connection

    The Carr–Madan identity works at the **payoff** level, not just
    through prices.  For any realisation $S_T$, the log payoff decomposes
    into a forward piece plus option payoffs integrated over strikes:

    $$
    -\log\!\frac{S_T}{F} = -\frac{S_T - F}{F} + \int_0^{F} \frac{(K - S_T)^+}{K^2}\,dK + \int_F^{\infty} \frac{(S_T - K)^+}{K^2}\,dK
    $$

    In practice we discretise the strike axis.  Each option payoff is a
    **piecewise-linear ramp** in $S_T$. Note that a linear change of
    basis produces the **triangular hat function** from the quasimatrix
    notebook. To option traders this basis has the interpretation of a
    portfolio of butterflies.
    """)
    return


@app.cell(hide_code=True)
def _(F, chebfun):
    S_lo, S_hi = 30.0, 250.0

    log_payoff = chebfun(lambda s: -np.log(s / F), [S_lo, S_hi])
    fwd_payoff = chebfun(lambda s: -(s - F) / F, [S_lo, S_hi])
    residual = log_payoff - fwd_payoff

    print(f"log payoff length: {len(log_payoff)}")
    residual
    return S_hi, S_lo, fwd_payoff, log_payoff, residual


@app.cell(hide_code=True)
def _(F, S_hi, S_lo, fwd_payoff, log_payoff, residual):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _ss = np.linspace(S_lo, S_hi, 500)
    _ax1.plot(_ss, log_payoff(_ss), linewidth=2, label=r"$-\log(S_T/F)$")
    _ax1.plot(_ss, fwd_payoff(_ss), "--", linewidth=1.5, label=r"$-(S_T - F)/F$")
    _ax1.axvline(F, color="k", linestyle=":", linewidth=0.8)
    _ax1.legend()
    _ax1.set_xlabel("$S_T$")
    _ax1.set_title("Log payoff and forward component")
    _ax1.grid(True)

    _ax2.plot(_ss, residual(_ss), linewidth=2, color="C2")
    _ax2.axvline(F, color="k", linestyle=":", linewidth=0.8)
    _ax2.set_xlabel("$S_T$")
    _ax2.set_title("Residual to replicate with options")
    _ax2.grid(True)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Building the option payoff quasimatrix

    We will work with the classic put and call basis directly. To do this
    we assemble a quasimatrix of OTM option payoffs and replicate the
    log contract.

    We pick $n$ equally spaced strikes spanning $[S_{\min}, S_{\max}]$.
    For each strike $K_i$:

    - If $K_i \le F$: include the put payoff $(K_i - S_T)^+$
    - If $K_i > F$: include the call payoff $(S_T - K_i)^+$

    Stacked into a quasimatrix, the columns are piecewise-linear ramps —
    the continuous analogues of the hat-function basis from the
    quasimatrix notebook.
    """)
    return


@app.cell(hide_code=True)
def _(F, Quasimatrix, S_hi, S_lo, chebfun):
    n_strikes = 16
    strikes = np.linspace(50.0, 200.0, n_strikes)
    all_bkpts = sorted({S_lo, *list(strikes), S_hi})

    _cols = []
    for _Ki in strikes:
        if _Ki <= F:
            _cols.append(chebfun(lambda s, _K=_Ki: np.maximum(_K - s, 0.0), all_bkpts))
        else:
            _cols.append(chebfun(lambda s, _K=_Ki: np.maximum(s - _K, 0.0), all_bkpts))

    Q_pay = Quasimatrix(_cols)
    print(f"Payoff quasimatrix shape: {Q_pay.shape}")
    return Q_pay, n_strikes, strikes


@app.cell(hide_code=True)
def _(F, Q_pay):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    Q_pay.plot(ax=_ax)
    _ax.axvline(F, color="k", linestyle="--", linewidth=1, label="$F$  (forward)")
    _ax.set_xlabel("$S_T$")
    _ax.set_title("OTM option payoffs — columns of the quasimatrix")
    _ax.legend()
    _ax.grid(True)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Least-squares weights via `Quasimatrix.solve()`

    The Carr–Madan formula tells us to weight each option by
    $\Delta K / K_i^2$.  Multiplying the quasimatrix by this weight
    vector gives an approximation to the residual log payoff.  At 16
    strikes the Carr–Madan replication is somewhat close but a long
    way from perfect.  The problem is domain truncation error —
    reflecting the fact that we cannot use strikes all the way down
    to zero and all the way up to infinity in practice — which
    distorts the ability of the $1/K^2$ weights to replicate
    accurately.

    Instead of the theoretical Carr–Madan weights, we can let ChebPy
    find the **optimal** weights that minimise
    $\| Q\,\mathbf{w} - g \|_2$, exactly as in the hat-function
    least-squares fit from the quasimatrix notebook.  This does much
    better under realistic strike-range truncation.
    """)
    return


@app.cell(hide_code=True)
def _(F, Q_pay, S_hi, S_lo, n_strikes, residual, strikes):
    _dK = strikes[1] - strikes[0]
    w_cm = _dK / strikes**2
    rep_cm = Q_pay @ w_cm

    w_ls = Q_pay.solve(residual)
    rep_ls = Q_pay @ w_ls

    _err_cm = float((residual - rep_cm).norm(2))
    _err_ls = float((residual - rep_ls).norm(2))

    _ss = np.linspace(S_lo, S_hi, 500)

    _fig, (_ax1, _ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        height_ratios=(2, 1),
    )

    # Top: replication comparison (three lines)
    _ax1.plot(_ss, residual(_ss), linewidth=2, label="Approximation Target")
    _ax1.plot(
        _ss,
        rep_cm(_ss),
        "--",
        linewidth=3,
        color="C3",
        label=rf"Carr–Madan ($\propto 1/K^2$) [$L^2$ err = {_err_cm:.3f}]",
    )
    _ax1.plot(
        _ss, rep_ls(_ss), "--", linewidth=3, color="C2", label=f"Chebfun least-squares  [$L^2$ err = {_err_ls:.3f}]"
    )
    _ax1.axvline(F, color="k", linestyle=":", linewidth=0.8)
    _ax1.legend()
    _ax1.set_xlabel("$S_T$")
    _ax1.set_title(f"Discrete replication ({n_strikes} strikes)")
    _ax1.grid(True)

    # Bottom: grouped bar chart of weights
    _bar_w = 0.3 * _dK
    _ax2.bar(
        strikes - _bar_w / 2,
        w_cm,
        width=_bar_w,
        color="C3",
        alpha=0.7,
        edgecolor="C3",
        label=r"Carr–Madan ($\propto 1/K^2$)",
    )
    _ax2.bar(
        strikes + _bar_w / 2, w_ls, width=_bar_w, color="C2", alpha=0.7, edgecolor="C2", label="Chebfun least-squares"
    )
    _ax2.set_xlabel("Strike $K$")
    _ax2.set_ylabel("Weight")
    _ax2.set_title("Portfolio weights")
    _ax2.legend()
    _ax2.grid(True, axis="y")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
