Latest stable release via Conda
-------------------------------

If you're working with Conda you can run::

    $ conda install -c conda-forge chebfun

Directly from GitHub
--------------------

If you're not working with conda (or would otherwise like to obtain up-to-date master branch of ChebPy), run::

    $ pip install git+https://github.com/chebpy/chebpy.git
    
To update, run::

    $ pip install git+https://github.com/chebpy/chebpy.git -U

The above has been tested from both Windows and Linux (and requires `git <https://git-scm.com>`_). 

If pip fails with "error: unknown file type '.pyx'", then you may need to first run

    $ pip install cython

Contributors
------------

Find a suitable location on your machine and run::

    $ git clone https://github.com/chebpy/chebpy.git

One way to proceed is then to add the outer chebpy/ folder to a PYTHONPATH environment variable.
