# Introduction

ChebPy is a Python implementation of [Chebfun](http://www.chebfun.org/), a system
for numerical computing with functions. Rather than working with discrete vectors of
numbers, ChebPy lets you work with *functions* as first-class objects.

## How It Works

Under the hood, each function is represented by a piecewise Chebyshev polynomial
expansion. ChebPy automatically determines the polynomial degree required to
approximate a given function to machine precision — you simply provide the function
and the domain.

The key idea comes from *Approximation Theory and Approximation Practice* by
Lloyd N. Trefethen: any smooth function on a bounded interval can be approximated
to essentially machine precision by a Chebyshev interpolant of moderate degree.

## Architecture

ChebPy uses a layered class hierarchy:

| Layer | Class | Purpose |
|-------|-------|---------|
| Top | `Chebfun` | Piecewise function on arbitrary intervals |
| Middle | `Bndfun` / `Classicfun` | Function on a single bounded interval $[a, b]$ |
| Base | `Chebtech` | Chebyshev expansion on the canonical interval $[-1, 1]$ |

A `Chebfun` consists of one or more `Bndfun` pieces, each of which maps its
interval to $[-1, 1]$ and delegates to a `Chebtech` for all the numerical work.

## Core Concepts

### Adaptive Construction

When constructing a Chebfun, ChebPy samples the function on progressively finer
Chebyshev grids until the coefficients decay below a tolerance (machine epsilon
by default). This means:

- Smooth functions are represented compactly (low degree)
- Functions with localised features may require breakpoint detection
- The resulting representation is accurate to roughly 15 digits

### Operations

Once a function is represented as a Chebfun, operations such as differentiation,
integration, and root-finding reduce to operations on the Chebyshev coefficients
and are performed in $O(n)$ or $O(n \log n)$ time, where $n$ is the polynomial degree.

## References

- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM, 2013.
- T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.), *Chebfun Guide*, Pafnuty Publications, 2014.
- [chebfun.org](http://www.chebfun.org/)
