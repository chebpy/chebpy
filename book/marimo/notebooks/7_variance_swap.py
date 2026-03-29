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
    # Variance Swap Replication with Calls and Puts

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

    We begin by constructing the Black\u2013Scholes call and put prices as
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
    return K_var, call_integrand, put_integrand


@app.cell(hide_code=True)
def _(F, K_hi, K_lo, call_integrand, put_integrand):
    _fig, _ax = plt.subplots()

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
    **piecewise-linear ramp** in $S_T$, and a butterfly spread at
    adjacent strikes produces a **triangular hat function** — exactly the
    same basis used in the quasimatrix notebook.

    We now build a quasimatrix of OTM option payoffs and replicate the
    log contract.
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
    n_strikes = 21
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
    _fig, _ax = plt.subplots()
    Q_pay.plot(ax=_ax)
    _ax.axvline(F, color="k", linestyle="--", linewidth=1, label="$F$")
    _ax.set_xlabel("$S_T$")
    _ax.set_title("OTM option payoffs — columns of the quasimatrix")
    _ax.legend()
    _ax.grid(True)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Weighting by $\Delta K / K^2$

    The Carr–Madan formula tells us to weight each option by
    $\Delta K / K_i^2$.  Multiplying the quasimatrix by this weight
    vector gives an approximation to the residual log payoff.
    """)
    return


@app.cell(hide_code=True)
def _(F, Q_pay, S_hi, S_lo, n_strikes, residual, strikes):
    _dK = strikes[1] - strikes[0]
    w_cm = _dK / strikes**2
    rep_cm = Q_pay @ w_cm

    _fig, _ax = plt.subplots()
    _ss = np.linspace(S_lo, S_hi, 500)
    _ax.plot(_ss, residual(_ss), linewidth=2, label="Exact residual")
    _ax.plot(
        _ss,
        rep_cm(_ss),
        "--",
        linewidth=2,
        color="C3",
        label=f"$\\Delta K / K^2$ weights ({n_strikes} strikes)",
    )
    _ax.axvline(F, color="k", linestyle=":", linewidth=0.8)
    _ax.legend()
    _ax.set_xlabel("$S_T$")
    _ax.set_title("Discrete replication of the log payoff residual")
    _ax.grid(True)

    _err = (residual - rep_cm).norm(2)
    print(f"L² replication error ({n_strikes} strikes, 1/K² weights): {float(_err):.6f}")

    plt.tight_layout()
    plt.show()
    return rep_cm, w_cm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Least-squares weights via `Quasimatrix.solve()`

    Instead of the theoretical $1/K^2$ weights, we can let ChebPy find
    the **optimal** weights that minimise
    $\| Q\,\mathbf{w} - g \|_2$, exactly as in the hat-function
    least-squares fit from the quasimatrix notebook.
    """)
    return


@app.cell(hide_code=True)
def _(F, Q_pay, S_hi, S_lo, rep_cm, residual, strikes, w_cm):
    w_ls = Q_pay.solve(residual)
    rep_ls = Q_pay @ w_ls

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _ss = np.linspace(S_lo, S_hi, 500)
    _ax1.plot(_ss, residual(_ss), linewidth=2, label="Exact residual")
    _ax1.plot(
        _ss,
        rep_ls(_ss),
        "--",
        linewidth=2,
        color="C1",
        label="Least-squares fit",
    )
    _ax1.axvline(F, color="k", linestyle=":", linewidth=0.8)
    _ax1.legend()
    _ax1.set_xlabel("$S_T$")
    _ax1.set_title("Least-squares replication")
    _ax1.grid(True)

    _ax2.plot(strikes, w_cm, "o-", color="C3", label="$\\Delta K / K^2$")
    _ax2.plot(strikes, w_ls, "s--", color="C1", label="Least-squares")
    _ax2.legend()
    _ax2.set_xlabel("Strike $K$")
    _ax2.set_ylabel("Weight")
    _ax2.set_title("Portfolio weights comparison")
    _ax2.grid(True)

    _err_ls = float((residual - rep_ls).norm(2))
    _err_cm = float((residual - rep_cm).norm(2))
    print(f"L² error (1/K² weights):  {_err_cm:.6f}")
    print(f"L² error (least-squares): {_err_ls:.6f}")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Convergence with the number of strikes

    As we increase the number of strikes in the discrete portfolio,
    the replication error shrinks.  ChebPy's $L^2$ norm makes the
    convergence easy to track.
    """)
    return


@app.cell(hide_code=True)
def _(F, Quasimatrix, S_hi, S_lo, chebfun, residual):
    n_vals = [5, 11, 21, 41, 81]
    errs_cm = []
    errs_ls = []

    for _n in n_vals:
        _strikes = np.linspace(50.0, 200.0, _n)
        _bkpts = sorted({S_lo, *list(_strikes), S_hi})
        _dK = _strikes[1] - _strikes[0]

        _cols = []
        for _Ki in _strikes:
            if _Ki <= F:
                _cols.append(chebfun(lambda s, _K=_Ki: np.maximum(_K - s, 0.0), _bkpts))
            else:
                _cols.append(chebfun(lambda s, _K=_Ki: np.maximum(s - _K, 0.0), _bkpts))

        _Q = Quasimatrix(_cols)
        _w_cm = _dK / _strikes**2
        errs_cm.append(float((residual - _Q @ _w_cm).norm(2)))

        _w_ls = _Q.solve(residual)
        errs_ls.append(float((residual - _Q @ _w_ls).norm(2)))

    _fig, _ax = plt.subplots()
    _ax.loglog(n_vals, errs_cm, "o-", linewidth=2, label="$\\Delta K / K^2$ weights")
    _ax.loglog(n_vals, errs_ls, "s--", linewidth=2, label="Least-squares weights")
    _ax.set_xlabel("Number of strikes")
    _ax.set_ylabel("$L^2$ replication error")
    _ax.set_title("Convergence of discrete variance swap replication")
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.5)

    plt.tight_layout()
    plt.show()
    return errs_cm, errs_ls, n_vals


@app.cell(hide_code=True)
def _(K_var, errs_cm, errs_ls, n_vals, sigma):
    _rows = "\n    ".join(f"| {n} | {e1:.6f} | {e2:.6f} |" for n, e1, e2 in zip(n_vals, errs_cm, errs_ls, strict=False))
    mo.md(rf"""
    ## Summary

    The Carr–Madan integral, evaluated exactly via ChebPy, recovers the
    Black–Scholes fair variance to high accuracy:

    | Method | Value |
    |---|---|
    | $K_{{\text{{var}}}}$ (ChebPy integral) | {float(K_var):.10f} |
    | $\sigma^2$ (exact) | {sigma**2:.10f} |

    The discrete payoff replication converges as more strikes are added:

    | Strikes | $L^2$ error ($1/K^2$) | $L^2$ error (least-sq) |
    |---|---|---|
    {_rows}

    The least-squares weights, found via `Quasimatrix.solve()`,
    consistently outperform the theoretical $1/K^2$ weights on the
    truncated domain — the same continuous least-squares machinery
    that fitted hat functions to $e^x\sin 6x$ in the quasimatrix
    notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
