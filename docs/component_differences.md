# Differences Between Chebtech, Bndfun, and Chebfun

This document explains the key differences and relationships between the three main components of the ChebPy library: Chebtech, Bndfun, and Chebfun.

## Class Hierarchy and Relationships

The ChebPy library implements a hierarchical structure for representing and manipulating functions:

```
                  Fun (ABC)
                    |
                    |
        +-----------+-----------+
        |                       |
        v                       v
   Smoothfun (ABC)         Classicfun (ABC)
        |                       |
        |                       |
        v                       v
   Chebtech (ABC)            Bndfun
        |                       |
        |                       |
        v                       |
    Chebtech2 <-----------------+
                                |
                                v
                             Chebfun
```

## Chebtech

**Purpose**: Chebtech is an abstract base class that represents functions on the standard domain [-1, 1] using Chebyshev polynomials.

**Key characteristics**:
- Works with Chebyshev coefficients to represent functions
- Provides core operational functionality for function manipulation
- Implements methods for evaluation, algebraic operations, calculus operations, and plotting
- Serves as the foundation for representing smooth functions
- Operates exclusively on the standard domain [-1, 1]

**Usage context**:
- Used internally as the underlying representation for functions
- Users rarely work with this class directly

## Bndfun

**Purpose**: Bndfun (Bounded Function) represents functions on arbitrary bounded intervals [a, b].

**Key characteristics**:
- Extends Classicfun, which is an abstract base class for functions on arbitrary intervals
- Maps functions from arbitrary intervals to the standard domain [-1, 1]
- Delegates the actual function representation to an underlying Chebtech object
- Handles the mapping between the arbitrary interval and the standard domain
- Provides methods for evaluation, algebraic operations, calculus operations, and plotting

**Usage context**:
- Used as building blocks for Chebfun objects
- Represents a single continuous piece of a piecewise function

## Chebfun

**Purpose**: Chebfun is the main class for representing and manipulating functions in ChebPy.

**Key characteristics**:
- Represents functions using piecewise polynomial approximations on arbitrary intervals
- Composed of an array of Bndfun objects, each representing a piece of the function
- Provides a comprehensive set of operations for working with function representations
- Supports both adaptive and fixed-length approximations
- Handles domain breaking operations (like max/min) that may introduce discontinuities
- Manages breakpoints between different pieces of the function

**Usage context**:
- Main interface for users to create and manipulate function representations
- Provides a high-level, user-friendly API for working with functions

## Summary of Differences

1. **Level of abstraction**:
   - Chebtech: Low-level representation on standard domain [-1, 1]
   - Bndfun: Mid-level representation on arbitrary intervals [a, b]
   - Chebfun: High-level representation of piecewise functions on arbitrary domains

2. **Domain handling**:
   - Chebtech: Fixed to standard domain [-1, 1]
   - Bndfun: Single arbitrary interval [a, b]
   - Chebfun: Multiple intervals with breakpoints

3. **Composition**:
   - Chebtech: Standalone representation using Chebyshev coefficients
   - Bndfun: Contains a Chebtech object and handles mapping to/from standard domain
   - Chebfun: Contains an array of Bndfun objects

4. **Functionality**:
   - Chebtech: Core mathematical operations on standard domain
   - Bndfun: Mathematical operations on arbitrary intervals
   - Chebfun: Comprehensive operations on piecewise functions, including domain breaking

5. **User interaction**:
   - Chebtech: Rarely used directly by users
   - Bndfun: Sometimes used directly, but often through Chebfun
   - Chebfun: Main interface for users

This hierarchical design allows ChebPy to efficiently represent and manipulate a wide variety of functions, from simple polynomials to complex piecewise functions with discontinuities.
