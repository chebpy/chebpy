# About ChebPy and Chebfun

ChebPy is a Python adaptation of [Chebfun](https://www.chebfun.org/), the
open-source numerical computing system for functions originally developed
in **Oxford's Numerical Analysis Group** at the Mathematical Institute,
University of Oxford. ChebPy reimplements a portion of the Chebfun design
in Python; the mathematics, algorithms, and overall architecture are
Chebfun's. The notes below give a brief account of where those ideas come
from, and which members of the Chebfun community to credit. They are
deliberately a summary — for a full account, see Chebfun's own
[History](https://www.chebfun.org/about/history.html) and
[People](https://www.chebfun.org/about/people.html) pages.

## Timeline

### Origins (2002–2005)

The project began as an idea of **Nick Trefethen** in late 2001: that one
could overload MATLAB vectors so they behave like the continuous functions
they sample. **Zachary Battles** worked out the consequences as his DPhil
thesis at Oxford under Trefethen, producing **Chebfun Version 1** for
smooth functions on $[-1, 1]$ (Battles & Trefethen, *SIAM J. Sci. Comp.*,
2004).

### Version 2 (2006–2008)

Version 2 grew out of EPSRC-funded work from 2006 onwards. **Ricardo
Pachón** extended the representation to piecewise-continuous functions on
arbitrary intervals; **Rodrigo Platte** added automatic subdivision and
edge detection; and **Toby Driscoll**, joining from the University of
Delaware in 2008, led the introduction of linear operators, ODEs,
integral operators, eigenvalue problems, and operator exponentials, in
collaboration with **Folkmar Bornemann**. Version 2 shipped in June 2008.

Key references for this era:

- Pachón, Platte & Trefethen, *Piecewise smooth chebfuns*, IMA J. Numer.
  Anal. (2010).
- Driscoll, Bornemann & Trefethen, *The chebop system for automatic
  solution of differential equations*, BIT Numer. Math. (2008).

### Version 3 (2008–2009)

**Nick Hale** joined as Trefethen's DPhil student and then postdoc, and
contributed Gauss-quadrature machinery for very large node counts and the
`pde15s` time-stepper for nonlinear PDEs. **Ásgeir Birkisson** introduced
automatic differentiation and nonlinear BVPs. **Mark Richardson** added
support for poles and endpoint singularities — the part of Chebfun whose
direct descendant is ChebPy's `Singfun`. Further contributions came
from Pedro Gonnet, Sheehan Olver, Joris Van Deun, and Alex Townsend.
Version 3 was released in December 2009, coordinated by Platte and Hale.

### Open source and the v5 rewrite (2010–2014)

Chebfun was relicensed under a BSD licence in 2010 and moved to GitHub.
The next four years saw the differential-equation infrastructure of
Driscoll and Hale, the Chebgui graphical interface (Birkisson and Hale),
and a curated Examples collection led by Trefethen. A 2012 ERC Advanced
Grant brought in a new generation of DPhil students and postdocs —
Anthony Austin, Mohsin Javed, Hadrien Montanelli, Hrothgar, Kuan Xu,
Behnam Hashemi, Jared Aurentz, Olivier Sète, and others. The September
2012 *Chebfun and Beyond* workshop set the direction for a ground-up
rewrite, and the modular, GitHub-hosted **Chebfun Version 5** was
released in June 2014, led by Hale.

In parallel, Trefethen's textbook
**[*Approximation Theory and Approximation Practice*](https://www.chebfun.org/ATAP/)**
was published by SIAM in 2013 (extended edition 2019) and remains the
mathematical foundation of the project.

### 2D, 3D and periodic extensions (2013–2016)

**Alex Townsend** built **Chebfun2** for two-dimensional functions on
rectangles using low-rank representations; **Behnam Hashemi** later
followed with **Chebfun3** (2016).

During a 2014 sabbatical at Oxford, **Grady Wright** (Boise State)
introduced a `'trig'` option for trigonometric Fourier-based chebfuns —
the "trigfun" capability that is the direct ancestor of ChebPy's
`Trigtech` module. Wright, Townsend, and **Heather Wilber** then built
**Spherefun** (May 2016); Wilber went on to lead **Diskfun**.

References:

- Townsend & Trefethen, *An extension of Chebfun to two dimensions*,
  SIAM J. Sci. Comp. (2013).
- Hashemi & Trefethen, *Chebfun in three dimensions*, SIAM J. Sci. Comp.
  (2017).
- Wright, Javed, Montanelli & Trefethen, *Extension of Chebfun to
  periodic functions*, SIAM J. Sci. Comp. (2015).
- Townsend, Wilber & Wright, *Computing with functions in spherical and
  polar geometries I & II*, SIAM J. Sci. Comp. (2016).

### Later work (2014–)

Subsequent additions to Chebfun include unification of IVPs and BVPs
under `u = N\f` (Birkisson), the `spin`/`spin2`/`spin3`/`spinsphere`
exponential-integrator solvers for stiff PDEs (Montanelli), and the AAA
algorithm and `minimax` for rational approximation (**Yuji Nakatsukasa**,
Olivier Sète, Trefethen, and **Silviu Filip**). Birkisson's IVP/BVP work
informs the book *Exploring ODEs* by Trefethen, Birkisson, and Driscoll.

- Nakatsukasa, Sète & Trefethen, *The AAA algorithm for rational
  approximation*, SIAM J. Sci. Comp. (2018).

## Key contributors

The Chebfun project is the work of many people; the names below are
those whose contributions most directly shape what ChebPy adapts. The
Chebfun team maintains a comprehensive list at
[chebfun.org/about/people.html](https://www.chebfun.org/about/people.html);
that page is the canonical reference.

The **founders and Version 1–2 leads** that ChebPy most directly inherits
from are:

- **Nick Trefethen** — invented Chebfun (2002) and led the project
  throughout; author of *ATAP* and co-author of *Exploring ODEs*.
- **Zachary Battles** — Chebfun Version 1; quasimatrix `qr`, `svd`, `\`,
  `roots`.
- **Ricardo Pachón** — piecewise polynomials; `remez`, `chebpade`,
  `lebesgue`.
- **Rodrigo Platte** — extension to infinite intervals; edge detection
  and `splitting on`.
- **Toby Driscoll** — ODE/integral side from 2008; `solvebvp`, `eigs`,
  `expm`, `volt`, `fred`; rectangular and block spectral
  discretisations.
- **Nick Hale** — Chebfun project director 2010–2014; led the v5
  rewrite; quadrature, `pde15s`, `conv`, `legpts`, `cheb2leg`.

Other contributors whose work appears in features ChebPy adapts (or
intends to adapt) include:

- **Ásgeir Birkisson** — automatic differentiation; nonlinear BVPs/IVPs;
  Chebgui.
- **Alex Townsend** — Chebfun2; `legpts`, `sum`, `conv`, `cheb2leg`,
  `nufft`; co-author of Spherefun.
- **Mark Richardson** — `blowup on`; poles and singularities (the basis
  of ChebPy's `Singfun`).
- **Folkmar Bornemann** — lazy-evaluation idea enabling operators.
- **Pedro Gonnet** — `ratinterp`, `padeapprox`.
- **Stefan Güttel** — rational functions; `chebsnake`.
- **Anthony Austin** — preferences architecture; technical leadership.
- **Mohsin Javed** — `dirac`, `trigremez`; coding-style guide.
- **Georges Klein** — `chebfun(...,'equi')` for equispaced data.
- **Hrothgar** — v5 website; block operators and adjoints.
- **Kuan Xu** — rectangular differentiation matrices; v5 unbounded
  intervals.
- **Hadrien Montanelli** — `spin`, `spin2`, `spin3`, `spinsphere`,
  `cheb.choreo`.
- **Behnam Hashemi** — Chebfun3.
- **Jared Aurentz** — `standardChop`, `adjoint`; continuous Krylov
  methods.
- **Olivier Sète** — co-author of `aaa`.
- **Yuji Nakatsukasa** — `aaa`, `minimax`; fast linear algebra for
  `spinsphere`.
- **Silviu Filip** — `minimax` rational best approximation.
- **Grady Wright** — the `'trig'` (trigfun) capability that ChebPy's
  `Trigtech` is descended from; led Spherefun.
- **Heather Wilber** — co-author of Spherefun; led Diskfun.
- **Sheehan Olver** — ultraspherical spectral methods (with Townsend).
- **Vanni Noferini** — fast rootfinding for Chebfun2.
- **Marcus Webb** — barycentric formulas; analytic continuation;
  improvements to `bary`.
- **Joris Van Deun** — `cf` for Carathéodory–Féjer approximation.
- **Richard Mikael Slevinsky** — quadrature, ultraspherical methods,
  ApproxFun link.
- **Jean-Paul Berrut** — championed the barycentric formulas that were
  the original spark for Chebfun.
- **Lourenço Peixoto** — `ultrapts`, improvements to `jacpts`.

## Foundational publications

The Chebfun project maintains a comprehensive
[publications list](https://www.chebfun.org/publications/). The papers
below are the most directly relevant to the algorithms and design adapted
in ChebPy.

### Books

- L. N. Trefethen,
  [*Approximation Theory and Approximation Practice*](https://www.chebfun.org/ATAP/),
  SIAM, 2013 (extended edition 2019).
- T. A. Driscoll, N. Hale, and L. N. Trefethen (eds.),
  [*Chebfun Guide*](https://www.chebfun.org/docs/guide/), Pafnuty
  Publications, Oxford, 2014. **(Recommended Chebfun citation.)**
- L. N. Trefethen, Á. Birkisson, and T. A. Driscoll,
  [*Exploring ODEs*](https://people.maths.ox.ac.uk/trefethen/ExplODE/),
  SIAM, 2018.

### Chebfun fundamentals

- Z. Battles and L. N. Trefethen,
  [*An extension of MATLAB to continuous functions and operators*](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2004_107.pdf),
  SIAM J. Sci. Comp., 2004.
- R. Pachón, R. B. Platte and L. N. Trefethen,
  [*Piecewise smooth chebfuns*](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2010_134.pdf),
  IMA J. Numer. Anal., 2010.
- L. N. Trefethen,
  [*Computing numerically with functions instead of numbers*](https://people.maths.ox.ac.uk/trefethen/cacm.pdf),
  Comm. ACM, 2015.
- J. L. Aurentz and L. N. Trefethen,
  [*Chopping a Chebyshev series*](https://people.maths.ox.ac.uk/trefethen/aurentz_trefethen_revised.pdf),
  ACM Trans. Math. Softw., 2017.

### Periodic functions (the Trigtech ancestor)

- G. B. Wright, M. Javed, H. Montanelli and L. N. Trefethen,
  [*Extension of Chebfun to periodic functions*](https://people.maths.ox.ac.uk/trefethen/trigpaper.pdf),
  SIAM J. Sci. Comp., 2015.

### Quadrature and convolution (used by ChebPy's `conv`)

- N. Hale and A. Townsend,
  [*Fast and accurate computation of Gauss–Legendre and Gauss–Jacobi quadrature nodes and weights*](https://www.chebfun.org/publications/HaleTownsend2013a.pdf),
  SIAM J. Sci. Comp., 2013.
- N. Hale and A. Townsend,
  [*An algorithm for the convolution of Legendre series*](https://www.chebfun.org/publications/HaleTownsend2014_PREPRINT.pdf),
  SIAM J. Sci. Comp., 2014.
- N. Hale and A. Townsend,
  [*A fast, simple, and stable Chebyshev–Legendre transform using an asymptotic formula*](https://www.chebfun.org/publications/HaleTownsend2013b_PREPRINT.pdf),
  SIAM J. Sci. Comp., 2014.

### Quasimatrix algebra (used by ChebPy's quasimatrix module)

- L. N. Trefethen,
  [*Householder triangularization of a quasimatrix*](https://www.chebfun.org/publications/trefethen_householder.pdf),
  IMA J. Numer. Anal., 2010.
- A. Townsend and L. N. Trefethen,
  [*Continuous analogues of matrix factorizations*](https://www.chebfun.org/publications/townsend_trefethen2014.pdf),
  Proc. Royal Soc. A, 2015.

### Rational approximation

- Y. Nakatsukasa, O. Sète and L. N. Trefethen,
  [*The AAA algorithm for rational approximation*](https://arxiv.org/abs/1612.00337),
  SIAM J. Sci. Comp., 2018.

## Sponsors of the Chebfun project

Chebfun has been supported over the years by **EPSRC** (UK Engineering and
Physical Sciences Research Council), the **European Research Council**
(Trefethen Advanced Grant, 2012–2017), **MathWorks**, and the **Oxford
Centre for Collaborative Applied Mathematics (OCCAM)**. See the
[Chebfun sponsors](https://www.chebfun.org/about/sponsors.html) page.

## Citing ChebPy and Chebfun

If you use ChebPy in published work, please also cite the original Chebfun
project:

> T. A. Driscoll, N. Hale, and L. N. Trefethen, editors,
> *Chebfun Guide*, Pafnuty Publications, Oxford, 2014.

Where the topic is closer to a specific Chebfun paper above (for instance
periodic representations or Hale–Townsend convolution), please cite that
paper in addition.

---

*This page summarises material from the Chebfun project's
[history](https://www.chebfun.org/about/history.html) and
[people](https://www.chebfun.org/about/people.html) pages, © Copyright the
University of Oxford and the Chebfun Developers, used here for
attribution.*
