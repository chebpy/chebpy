# 📊 ChebPy - A Python implementation of Chebfun

[![CI](https://github.com/chebpy/chebpy/actions/workflows/ci.yml/badge.svg)](https://github.com/chebpy/chebpy/actions/workflows/ci.yml)
[![Coverage](https://coveralls.io/repos/github/chebpy/chebpy/badge.svg?branch=master)](https://coveralls.io/github/chebpy/chebpy?branch=master)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://github.com/chebpy/chebpy/actions/workflows/unittest.yml)

[![Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/chebpy/chebpy)

**🔬 Numerical computing with Chebyshev series approximations in Python.**

ChebPy is a Python implementation of [Chebfun](http://www.chebfun.org/), enabling symbolic-numeric computation with functions, not just numbers.

<div align="center">
  <img src="docs/chebpy-readme-image1.png" alt="ChebPy Example" width="80%">
</div>

---

## 📥 Installation

To install ChebPy, simply run:

```bash
pip install chebpy
```

To install the latest version from source:

```bash
git clone https://github.com/chebpy/chebpy.git
cd chebpy
pip install .
```

## 🛠️ Development

Rather than installing Chebpy into your existing project
you may want to work with the repository directly.
Chebpy uses modern Python development tools:

```bash
# Install development dependencies
make install

# Run tests
make test

# Format code
make fmt
make lint

# Start interactive notebooks
make marimo
```

## 🔧 Quick Start

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

## 📄 License

Chebpy is licensed under the 3-Clause BSD License. 
See the full license in the [LICENSE.rst](LICENSE.rst) file.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For more information, see [CONTRIBUTING.md](.github/CONTRIBUTING.md).
