try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='chebpy',
    version='0.4.0',
    description='A Python implementation of Chebfun',
    long_description=open('README.rst',"rt").read(),
    author='Mark Richardson',
    author_email='mrichardson82@gmail.com',
    url='https://github.com/chebpy/chebpy',
    license=open('LICENSE.rst',"rt").read(),
    packages= ['chebpy', 'chebpy.core'],
    install_requires=[
        'numpy>=1.16',
        'pyfftw>=0.11',
    ],
    test_suite="tests",
)
