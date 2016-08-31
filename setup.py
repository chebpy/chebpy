try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='chebpy',
    version='0.0.1',
    description='A Python implementation of Chebfun',
    long_description=open('README.rst',"rt").read(),
    # author='',
    author_email='',
    url='https://github.com/chebpy/chebpy',
    # license='',
    packages= ['chebpy'],
    install_requires=[
        "numpy >= 1.11.0",
        "matplotlib < 2.0.0", # mpl > 2.0 only works on py3
        "pyFFTW >= 0.8.1",
    ],
    test_suite="tests",
)
