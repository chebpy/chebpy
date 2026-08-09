# Paper

**ChebPy — Computing with Functions via Chebyshev Approximation in Python**

A single self-contained document describing what ChebPy is, the mathematics it
rests on, the numerical kernels that implement it, and the object model that
organises them. It is the reference to read end-to-end; the
[User Guide](../user/intro.md) and [API Reference](../api.md) are the reference
to look things up in.

[Download the PDF](chebpy.pdf){ .md-button .md-button--primary download="chebpy.pdf" }
[Read the LaTeX source](https://github.com/chebpy/chebpy/blob/master/docs/paper/chebpy.tex){ .md-button }

## Abstract

ChebPy is a Python library for numerical computing with *functions* rather than
numbers. A smooth function is replaced by a polynomial interpolant through
Chebyshev points, accurate to close to machine precision, and thereafter
differentiation, integration, rootfinding, convolution and ordinary arithmetic
are carried out on that surrogate. The result is a system in which `f + g`,
`f.diff()` and `f.roots()` mean what a mathematician expects them to mean. The
paper describes the mathematical foundation, the numerical kernels that
implement it, the two-hierarchy object model that organises them, the extensions
that carry the idea beyond smooth functions on bounded intervals — Fourier
technology for periodic functions, support truncation for infinite intervals and
endpoint-clustering maps for branch singularities — and the higher-level
applications built on top. ChebPy is a Python descendant of the MATLAB package
Chebfun, and follows its algorithmic choices closely while departing from them
where a different design suits the host language better.

## What is inside

| Section | Subject | Related pages |
| --- | --- | --- |
| Chebyshev approximation | Chebyshev series, coefficient decay, interpolation | [Approximation](../user/features/approximation.md) |
| The numerical kernels | DCT, `standard_chop`, adaptive construction, Clenshaw evaluation, calculus, rootfinding, Legendre conversion | [Calculus](../user/features/calculus.md), [Root-Finding](../user/features/rootfinding.md), [Fast Convolution](../user/features/convolution.md) |
| Architecture | The two runtime hierarchies, import layering, the piecewise container, the public surface | [Architecture](../development/architecture.md) |
| Beyond smooth functions | `Trigtech`, `CompactFun`, `Singfun` | [Periodic Functions](../user/features/periodic.md), [Infinite Intervals](../user/features/infinite-intervals.md), [Endpoint Singularities](../user/features/singularities.md) |
| Applications | Quasimatrices, Gaussian process regression | [Quasimatrix Algebra](../user/features/quasimatrix.md), [Gaussian Processes](../user/features/gpr.md) |

## Read it here

<!-- markdownlint-disable MD033 -->
<object data="chebpy.pdf" type="application/pdf" width="100%" height="900" title="ChebPy paper">
  <p>Your browser will not display the PDF inline —
  <a href="chebpy.pdf" download>download it instead</a>.</p>
</object>
<!-- markdownlint-enable MD033 -->

## Building it yourself

The PDF committed alongside this page is the artifact the book ships, because
the book build has no LaTeX toolchain. To rebuild it from
[`docs/paper/chebpy.tex`](https://github.com/chebpy/chebpy/blob/master/docs/paper/chebpy.tex):

```bash
make paper        # latexmk -pdf -bibtex; writes docs/paper/chebpy.pdf
make paper-clean  # drop latexmk build artifacts
```

`make paper` needs a LaTeX distribution (TeX Live, MacTeX) on `PATH`. Commit the
regenerated `docs/paper/chebpy.pdf` whenever the source changes, so the
published book and the source stay in step. The
[`(RHIZA) PAPER`](https://github.com/chebpy/chebpy/actions/workflows/rhiza_paper.yml)
workflow also compiles it on every change under `docs/paper/` and uploads the
result as a downloadable run artifact.
