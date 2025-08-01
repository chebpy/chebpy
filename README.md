# ChebPy - A Python implementation of Chebfun

[![CI](https://github.com/chebpy/chebpy/actions/workflows/ci.yml/badge.svg)](https://github.com/chebpy/chebpy/actions/workflows/ci.yml)
[![Coverage](https://coveralls.io/repos/github/chebpy/chebpy/badge.svg?branch=master)](https://coveralls.io/github/chebpy/chebpy?branch=master)
[![Python](https://img.shields.io/badge/python-%203.10_--%203.13-blue.svg?)](https://github.com/chebpy/chebpy/actions/workflows/unittest.yml)

[![Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/chebpy/chebpy)

Numerical computing with Chebyshev series approximations in Python.

![ChebPy Example](docs/chebpy-readme-image1.png)

ChebPy is a Python implementation of [Chebfun](http://www.chebfun.org/).

- The software is licensed under a 3-Clause BSD License, see [LICENSE.rst](LICENSE.rst).
- For installation details, see [INSTALL.rst](INSTALL.rst).
- The code is documented in various files in the [docs](docs/) folder.

The figure above was generated with the following simple ChebPy code:

```python
>>> # Import required libraries
>>> import numpy as np
>>> from chebpy import chebfun

>>> # Create first chebfun representing a sum of sine functions on interval [0, 10]
>>> f = chebfun(lambda x: np.sin(x**2) + np.sin(x)**2, [0, 10])
>>> # Create second chebfun representing a Gaussian function centered at x=5
>>> g = chebfun(lambda x: np.exp(-(x-5)**2/10), [0, 10])

>>> # Find the roots (zeros) of the difference between f and g
>>> # These are the points where the two functions intersect
>>> r = (f-g).roots()

>>> # Plot the first function
>>> ax = f.plot()
>>> # Add the second function to the same plot
>>> ax = g.plot(ax=ax)
>>> # Mark the intersection points with circles
>>> # The underscore (_) is used to suppress the output of the plot command
>>> _ = ax.plot(r, f(r), 'o')
```
