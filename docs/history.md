# History of the Chebfun Project

ChebPy is a Python adaptation of [Chebfun](https://www.chebfun.org/), an
open-source software system for numerical computing with functions originating
in **Oxford's Numerical Analysis Group** at the Mathematical Institute,
University of Oxford. This page summarises the history and people behind the
original Chebfun project, on whose mathematical and algorithmic foundations
ChebPy is built. The narrative below is condensed from the Chebfun project's
own [History](https://www.chebfun.org/about/history.html) and
[People](https://www.chebfun.org/about/people.html) pages.

## Timeline

### The beginning (2002–2005)

Chebfun began during 2002–2005 as the DPhil research of **Zachary Battles**, a
Rhodes Scholar from the USA, supervised by **Nick Trefethen**. The idea of
overloading MATLAB vectors as functions was first written down in an email
from Trefethen to Battles on 8 December 2001. This led to **Chebfun
Version 1**, for smooth functions on $[-1, 1]$, described in:

- Battles & Trefethen, *An extension of MATLAB to continuous functions and
  operators*, **SIAM J. Sci. Comp.** (2004).

### Version 2 (2006–2008)

The second phase began in autumn 2006 with EPSRC research funding.
**Ricardo Pachón** (Colombia) extended Chebfun to **piecewise continuous
functions and arbitrary intervals**. Automatic subdivision and edge detection
were added with **Rodrigo Platte** (Brazil, postdoc from 2007). From January
2008, **Toby Driscoll** (University of Delaware) led the addition of linear
operators, ODEs, integral operators, eigenvalue problems, and operator
exponentials, in collaboration with **Folkmar Bornemann** (TU Munich).
Version 2 was released in **June 2008**.

- Pachón, Platte & Trefethen, *Piecewise smooth chebfuns*, **IMA J. Numer.
  Anal.** (2010).
- Platte & Trefethen, *Chebfun: a new kind of numerical computing*, **ECMI
  2008 Proceedings** (2010).
- Driscoll, Bornemann & Trefethen, *The chebop system for automatic solution
  of differential equations*, **BIT Numer. Math.** (2008).

### Version 3 (2008–2009)

**Nick Hale** joined as Trefethen's DPhil student and then postdoc, adding
Gauss quadrature for millions of points and `pde15s` for nonlinear PDEs.
**Mark Richardson** added support for functions with poles and singularities;
**Ásgeir Birkisson** added automatic differentiation and nonlinear BVPs with
Driscoll. **Pedro Gonnet**, **Sheehan Olver**, **Joris Van Deun** and
**Alex Townsend** also contributed. Version 3 was released in
**December 2009**, coordinated by Platte and Hale.

### Version 4 (2010–2014)

Chebfun became **open source under a BSD licence** in 2010. Major new
features were Driscoll and Hale's differential-equation infrastructure,
the **Chebgui** graphical interface (Birkisson & Hale), and a curated
**Examples** collection led by Trefethen.

### Version 5 and "Chebfun and Beyond" (2012–2014)

A five-year **ERC Advanced Grant** to Trefethen in 2012 brought a new
generation of contributors: **Anthony Austin**, **Mohsin Javed**,
**Hadrien Montanelli**, **Hrothgar** (DPhil students), and **Kuan Xu**,
**Behnam Hashemi**, **Jared Aurentz**, **Olivier Sète** (postdocs).
The September 2012 *Chebfun and Beyond* workshop set the direction for a
ground-up rewrite. **Chebfun Version 5** — modular, GitHub-hosted, with a
new website by Hrothgar — was released on **21 June 2014**, led by
Nick Hale.

### Approximation Theory and Approximation Practice (2011–2013)

Trefethen's textbook **[*Approximation Theory and Approximation Practice*](https://www.chebfun.org/ATAP/)**
was published by SIAM in 2013 (extended edition 2019) and remains the
mathematical foundation of the project.

### Chebfun2, Chebfun3 (2013–2016)

**Alex Townsend** created **Chebfun2** for 2D functions on rectangles
using low-rank representations. **Behnam Hashemi** later created
**Chebfun3** (2016) for 3D functions.

- Townsend & Trefethen, *An extension of Chebfun to two dimensions*,
  **SIAM J. Sci. Comp.** (2013).
- Hashemi & Trefethen, *Chebfun in three dimensions*,
  **SIAM J. Sci. Comp.** (2017).

### Periodic functions, Spherefun, and Diskfun (2014–)

During a 2014 sabbatical at Oxford, **Grady Wright** (Boise State)
introduced a `'trig'` option for trigonometric (Fourier-based) Chebfuns —
the "trigfun" capability, which is the direct ancestor of ChebPy's
`Trigtech` module. Wright, Townsend, and **Heather Wilber** then built
**Spherefun** (released with v5.4 in May 2016); Wilber went on to lead
**Diskfun**.

- Wright, Javed, Montanelli & Trefethen, *Extension of Chebfun to periodic
  functions*, **SIAM J. Sci. Comp.** (2015).
- Townsend, Wilber & Wright, *Computing with functions in spherical and
  polar geometries I & II*, **SIAM J. Sci. Comp.** (2016).

### Unifying ODE IVPs and BVPs (2014–2015)

**Ásgeir Birkisson** unified the IVP/BVP user syntax so both can be solved
via `u = N\f`, despite using very different underlying algorithms. This
work informs the book **Exploring ODEs** by Trefethen, Birkisson, and
Driscoll.

### Time-dependent PDEs and `spin`

**Hadrien Montanelli** introduced `spin`, `spin2`, `spin3` and
`spinsphere` for stiff reaction–diffusion-type PDEs, based on exponential
integrators (ETDRK4 of Cox & Matthews).

### Rational approximation: AAA and minimax

**Yuji Nakatsukasa**, **Olivier Sète**, and Trefethen developed the
**AAA algorithm** for rational approximation; **Silviu Filip** led
`minimax` for rational best approximation.

- Nakatsukasa, Sète & Trefethen, *The AAA algorithm for rational
  approximation*, **SIAM J. Sci. Comp.** (2018).

## Key contributors

The list below summarises the people whose contributions shape the
mathematical and software design that ChebPy adapts. Each name links to
the contributor's current academic homepage where available, falling back
to a Google Scholar search otherwise. The Chebfun team maintains a more
complete list at
[chebfun.org/about/people.html](https://www.chebfun.org/about/people.html).

| Person | Role / contribution |
| --- | --- |
| [**Nick Trefethen**](https://people.maths.ox.ac.uk/trefethen/) | Invented Chebfun (2002); project leader throughout; author of *ATAP* and co-author of *Exploring ODEs* |
| [**Zachary Battles**](https://scholar.google.com/scholar?q=Zachary+Battles+chebfun) | DPhil 2002–2005; wrote Chebfun v1; quasimatrix `qr`, `svd`, `\`, `roots` |
| [**Ricardo Pachón**](https://scholar.google.com/citations?user=BJ7HWzkAAAAJ&hl=en) | DPhil from 2006; piecewise polynomials; `remez`, `chebpade`, `lebesgue` |
| [**Rodrigo Platte**](https://scholar.google.com/citations?user=sujKeUUAAAAJ&hl=en) | Postdoc 2007–2009; extension to infinite intervals; edge-detection / `splitting on` |
| [**Toby Driscoll**](https://tobydriscoll.net/) | Led ODE/integral side from 2008; `solvebvp`, `eigs`, `expm`, `volt`, `fred`; rectangular & block spectral discretisations |
| [**Folkmar Bornemann**](https://www.professoren.tum.de/en/bornemann-folkmar) | Lazy-evaluation idea enabling operators |
| [**Nick Hale**](https://scholar.google.com/scholar?q=Nick+Hale+chebfun) | DPhil 2006–2009, postdoc to 2014; project director 2010–2014; led v5 rewrite; quadrature, `pde15s`, `conv`, `legpts`, `cheb2leg` |
| [**Mark Richardson**](https://scholar.google.com/citations?user=mxOjLP8AAAAJ&hl=en) |MSc / DPhil 2008–2013; `blowup on`, poles and singularities |
| [**Ásgeir Birkisson**](https://scholar.google.com/citations?user=rVsf9VoAAAAJ&hl=en) | MSc / DPhil / postdoc 2008–2015; automatic differentiation; nonlinear BVPs/IVPs; `Chebgui` |
| [**Pedro Gonnet**](https://scholar.google.com/citations?user=IqOpdUkAAAAJ&hl=en) | Postdoc 2009–2012; `ratinterp`, `padeapprox` |
| [**Stefan Güttel**](https://guettel.com/) | Postdoc 2011–2012; rational functions; `chebsnake` |
| [**Alex Townsend**](https://alextownsend.net/) | DPhil 2010–2014; created Chebfun2; co-author of Spherefun; `legpts`, `sum`, `conv`, `cheb2leg`, `nufft` |
| [**Anthony Austin**](https://scholar.google.com/citations?user=J8pe_aYAAAAJ&hl=en) | DPhil from 2012; preferences architecture; technical leadership |
| [**Mohsin Javed**](https://scholar.google.com/citations?user=boh_ussAAAAJ&hl=en) | MSc / DPhil 2011–2017; `dirac`, `trigremez`; coding-style guide |
| [**Georges Klein**](https://scholar.google.com/scholar?q=Georges+Klein+barycentric+interpolation) | Postdoc 2012–2013; `chebfun(...,'equi')` for equispaced data |
| [**Hrothgar**](https://scholar.google.com/scholar?q=Hrothgar+chebfun) | DPhil 2013–2015; v5 website; block operators / adjoints |
| [**Kuan Xu**](https://scholar.google.com/scholar?q=Kuan+Xu+chebfun) | Postdoc 2012–2015; rectangular differentiation matrices; v5 unbounded-interval code |
| [**Hadrien Montanelli**](https://scholar.google.com/citations?user=Bjmkfe8AAAAJ&hl=en) | DPhil 2013–2017; `spin`, `spin2`, `spin3`, `spinsphere`, `cheb.choreo` |
| [**Behnam Hashemi**](https://scholar.google.com/scholar?q=Behnam+Hashemi+chebfun3) | Postdoc 2014–2017; created Chebfun3 |
| [**Jared Aurentz**](https://scholar.google.com/citations?user=HrT33IsAAAAJ&hl=en) | Postdoc 2014–2016; `standardChop`, `adjoint`; continuous Krylov methods |
| [**Olivier Sète**](https://scholar.google.com/citations?user=zuvLfoQAAAAJ&hl=en) | Postdoc 2015–2017; co-author of `aaa` |
| [**Yuji Nakatsukasa**](https://people.maths.ox.ac.uk/nakatsukasa/) | Visiting researcher; `aaa`, `minimax`; fast linear algebra for `spinsphere` |
| [**Silviu Filip**](https://scholar.google.com/citations?user=IbrAWwgAAAAJ&hl=en) | Postdoc from 2016; `minimax` rational best approximation |
| [**Grady Wright**](https://scholar.google.com/citations?user=-wgTqlMAAAAJ&hl=en) | 2014 visit; `'trig'` (trigfun) capability; led Spherefun |
| [**Heather Wilber**](https://scholar.google.com/citations?user=1Nw_is4AAAAJ&hl=en) | MSc 2015–2016 (Boise State); co-author of Spherefun; led Diskfun |
| [**Sheehan Olver**](https://www.ma.imperial.ac.uk/~solver/) | JRF Oxford 2007–2011; ultraspherical spectral methods (with Townsend) |
| [**Vanni Noferini**](https://scholar.google.com/scholar?q=Vanni+Noferini+chebfun) | Postdoc Manchester; fast rootfinding for Chebfun2 |
| [**Marcus Webb**](https://personalpages.manchester.ac.uk/staff/marcus.webb/) | Barycentric formulas; analytic continuation; improvements to `bary` |
| [**Joris Van Deun**](https://scholar.google.com/scholar?q=Joris+Van+Deun+CF+approximation) | `cf` for Carathéodory–Féjer approximation |
| [**Richard Mikael Slevinsky**](https://scholar.google.com/scholar?q=Richard+Mikael+Slevinsky) | Postdoc 2014–2016; quadrature, ultraspherical methods, ApproxFun link |
| [**Jean-Paul Berrut**](https://scholar.google.com/scholar?q=Jean-Paul+Berrut+barycentric) | Champion of barycentric formulas — the original spark for Chebfun |
| [**Lourenço Peixoto**](https://scholar.google.com/citations?user=1x1cmO4AAAAJ&hl=en) | `ultrapts`, improvements to `jacpts` |

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
