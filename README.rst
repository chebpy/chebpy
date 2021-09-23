================================================
ChebPy - A Python implementation of Chebfun
================================================

.. image:: https://travis-ci.org/chebpy/chebpy.svg?branch=master
    :target: https://travis-ci.org/chebpy/chebpy
.. image:: https://coveralls.io/repos/github/chebpy/chebpy/badge.svg?branch=master
    :target: https://coveralls.io/github/chebpy/chebpy?branch=master
.. image:: https://img.shields.io/badge/python-%203.6,%203.7,%203.8-blue.svg?
    :target: https://travis-ci.org/chebpy/chebpy
.. image:: https://img.shields.io/conda/dn/conda-forge/chebfun?label=conda%20downloads
    :target: https://anaconda.org/conda-forge/chebfun

|


Numerical computing with Chebyshev series approximations in Python.


.. image:: docs/chebpy-readme-image1.png


ChebPy is a Python implementation of `Chebfun <http://www.chebfun.org/>`_.

- The software is licensed under a 3-Clause BSD License, see `LICENSE.rst <LICENSE.rst>`_.
- For installation details, see `INSTALL.rst <INSTALL.rst>`_.
- The code is documented in various files in the `docs <docs/>`_ folder.


The figure above was generated with the following simple ChebPy code:

.. code:: python

	f = chebfun(lambda x: np.sin(x**2) + np.sin(x)**2, [0, 10])
	g = chebfun(lambda x: np.exp(-(x-5)**2/10), [0, 10])
	r = (f-g).roots()
	ax = f.plot(); g.plot()
	ax.plot(r, f(r), 'o')
