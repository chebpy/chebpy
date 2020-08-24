Default installation
--------------------

Perhaps the most convenient way to get going is to run::

    $ pip install git+https://github.com/chebpy/chebpy.git
    
If you are running Anaconda this can also be run from the Anaconda prompt. The above has been tested from both Windows and Linux and requires `git <https://git-scm.com>`_ as a prerequisite. 

To update chebpy from source, run::

    $ pip install git+https://github.com/chebpy/chebpy.git -U

Alternative installation
------------------------

To install manually, first install the `fftw` library::

    # On Linux
    $ sudo apt-get install libfftw3-dev

    # Via Homebrew on Mac
    $ brew install fftw

And then proceed to install chebpy with::

    $ python setup.py install
