# -*- coding: utf-8 -*-

# import fftw via pyfftw if the user has it installed, otherwise default to 
# fftpack via numpy

try:
    from pyfftw.interfaces.numpy_fft import fft
    from pyfftw.interfaces.numpy_fft import ifft
except:
    from numpy.fft import fft
    from numpy.fft import ifft
