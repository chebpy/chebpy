import marimo

__generated_with = "0.14.16"
app = marimo.App()

with app.setup:
    from pathlib import Path
    import marimo as mo

    path = Path(__file__).parent


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Implementation Notes

    The internals of ChebPy have been designed to resemble the design structure of MATLAB Chebfun. The Chebfun v5 class diagram thus provides a useful map for understanding how the various pieces of ChebPy fit together (diagram courtesy of the Chebfun team, available [here](https://github.com/chebfun/chebfun/wiki/Class-diagram)):
    """
    )
    return


@app.cell(hide_code=True)
def _():
    file = path / "chebfun-v5-class-diag.png"
    assert file.exists()
    mo.image(file.read_bytes())
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    At this stage, only a relatively small subset of MATLAB Chebfun has been implemented in ChebPy. In the class diagram above, this consists of all the classes traced by the path going up from from `Chebtech` (green box in the bottom right), to `Chebfun` (blue box near the top-left). More explicitly, the following classes currently exist in ChebPy:

    - `Chebfun` ([core/chebfun.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/chebfun.py))
    - `Fun` ([core/fun.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/fun.py))
    - `Classicfun` ([core/classicfun.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/classicfun.py))
    - `Bndfun` ([core/bndfun.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/bndfun.py))
    - `Onefun` ([core/onefun.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/onefun.py))
    - `Smoothfun` ([core/smoothfun.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/smoothfun.py))
    - `Chebtech` ([core/chebtech.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/chebtech.py))
    - `Chebtech` ([core/chebtech.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/chebtech.py))

    ChebPy additionally provides the following classes which do not appear in their present form in MATLAB Chebfun:

    - `Interval` ([core/utilities.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/utilities.py))
    - `Domain` ([core/utilities.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/utilities.py))

    ChebPy allows the user to override some default preferences, similar to what is available through `chebfunpref` in `Chebfun` (bottom right). Not all options are the same, and chebpy allows additional customisation not found in `Chebfun`.

    - `UserPrefs` ([core/settings.py](https://github.com/chebpy/chebpy/blob/master/chebpy/core/settings.py))

    The general rule is that each ChebPy class lives in its own python file.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""One can explore the organisation of the library in practice as follows. We describe the three core components with reference to the colours in the above class diagram.""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return np, plt


@app.cell
def _():
    # optional plot settings
    import matplotlib
    import seaborn as sns

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    matplotlib.rc("figure", figsize=(9, 5), dpi=100)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Chebfun (blue)

    We'll start by creating an identity chebfun on an arbitrary non-unit interval:
    """
    )
    return


@app.cell
def _():
    from chebpy import chebfun

    x = chebfun("x", [-2, 3])
    x
    return chebfun, x


@app.cell(hide_code=True)
def _():
    mo.md(r"""This variable is an object of class `Chebfun`:""")
    return


@app.cell
def _(x):
    type(x)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Fun (yellow)

    Conceptually, `Chebfun` objects are defined as a collection (numpy array) of Fun objects. One can access these via the `.funs` attribute, and in this example, since our function is globally smooth, our chebfun is composed of single Fun:
    """
    )
    return


@app.cell
def _(x):
    x.funs
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Fun is an abstract class, so we don't actually see Fun objects per-se, but rather objects further down the inheritance tree. Specifically, we see objects of type `BndFun`, denoting a function defined on a bounded interval.

    Here's a piecewise smooth function created by inducing a discontinuity via use of the maximum operator.
    """
    )
    return


@app.cell
def _(np, x):
    f = np.sin(x).maximum(-np.sin(x))
    f
    return (f,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""Here the representation consists of two `Fun` objects:""")
    return


@app.cell
def _(f):
    f.funs
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""The core `Chebfun` class constructor accepts iterable collections of `Fun` objects, and the above chebfun can be recreated as follows:""")
    return


@app.cell
def _(f):
    from chebpy.core.chebfun import Chebfun

    Chebfun(f.funs)
    return (Chebfun,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The Fun objects defining a chebfun must provide a non-overlapping and complete partition of the global approximation interval. ChebPy-specific exceptions will be raised if the user/developer does not properly account for this.

    To see this, let's break the second Fun into two sub-pieces (using `fun.restrict`) and attempt various reconstruction permutations.
    """
    )
    return


@app.cell
def _(f):
    f.breakpoints
    return


@app.cell
def _(f, plt):
    from chebpy.core.utilities import Interval

    _, a, c = f.breakpoints
    b = 1
    fun0 = f.funs[0]
    fun1 = f.funs[1].restrict(Interval(a, b))
    fun2 = f.funs[1].restrict(Interval(b, c))
    fun0.plot(linewidth=3)
    fun1.plot(linewidth=3)
    fun2.plot(linewidth=3)
    plt.show()
    return fun0, fun1, fun2


@app.cell(hide_code=True)
def _():
    mo.md(r"""So, the following works:""")
    return


@app.cell
def _(Chebfun, fun0, fun1, fun2):
    Chebfun([fun0, fun1, fun2])
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    However, the following will raise an exception:

    ```
    >> Chebfun([fun0, fun2])
    IntervalGap: The supplied Interval objects do not form a complete partition of the approximation interval
    ```

    As will:

    ```
    >> Chebfun([fun0, f.funs[1], fun1])
    IntervalOverlap: The supplied Interval objects overlap
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Onefun (green)

    A Fun object is defined by the composition of `Onefun` object and an Interval object. A `Onefun` object in ChebPy, as with MATLAB Chebfun, define a set of core approximation behaviour on the unit interval [-1,1]. The computational mechanics of mapping these operations to arbitrary intervals [a,b] is managed, in part, by the a corresponding Interval object.

    To illustrate, let's take the first component `Fun` from earlier (which was specifically a `Bndfun`):
    """
    )
    return


@app.cell
def _(f):
    f.funs[0]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""The Onefun and Interval object components are accessed as follows:""")
    return


@app.cell
def _(f):
    f.funs[0].onefun, f.funs[0].interval
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""`Onefun` is an abstract class, so what we actually see instantiated is an object of type `Chebtech`. To see that the `Onefun` object is what is claims to be (a representation defined on the unit-interval), we can plot it (users will rarely do this in practice, but this can nevertheless be a useful feature for developers):""")
    return


@app.cell
def _(f, plt):
    f.funs[0].onefun.plot(linewidth=3)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""So while the `Onefun` attribute determines approximation behaviour, the interval attribute manages the mapping to and from the approximation interval to [-1,1]. So for instance, one can evaluate the interval object at some set of points in [-1,1] and obtain these values mapped to [a,b]:""")
    return


@app.cell
def _(f, np):
    f.funs[0].interval(np.linspace(-1, 1, 11))
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## User preferences (`UserPrefs`)

    The user may want to specify different tolerances, for example if speed is important or the function under consideration is particularly difficult. It is also possible to change default behaviour like plotting.
    """
    )
    return


@app.cell
def _(chebfun, plt):
    from chebpy import UserPreferences
    from chebpy.core.settings import DefaultPreferences

    user_prefs = UserPreferences()

    with user_prefs as local_prefs:
        local_prefs.eps = 1e-10  # lower the tolerance in chebpy
        cheb = chebfun(lambda x: x**2)
    assert user_prefs.eps == DefaultPreferences.eps  # value did reset!

    with user_prefs:  # we don't have to assign a new name
        user_prefs.N_plot = 21  # use fewer points for plotting
        cheb.plot(marker="x", label="N_plot = 21")
        user_prefs.reset("N_plot")  # restore default
        cheb.plot(label="default N_plot")


    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
