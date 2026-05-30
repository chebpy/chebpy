"""Visualise issue 45: tangential roots in pointwise maximum.

Run from the repository root with:

    uv run python docs/examples/issue45_tangential_maximum.py
"""

import numpy as np
import matplotlib.pyplot as plt

from chebpy import chebfun


def main() -> None:
    x = chebfun("x", [-2, 3])
    f1 = np.sin(3 * x)
    f2 = -np.sin(x)
    diff = f1 - f2
    roots = diff.roots()

    xx = np.linspace(-2, 3, 3000)
    f1_values = np.sin(3 * xx)
    f2_values = -np.sin(xx)

    # Recreate the old failure mode for illustration: split at every raw
    # root and alternate branches blindly.
    breakpoints = np.r_[-2, roots, 3]
    old_values = np.empty_like(xx)
    midpoint = 0.5 * (breakpoints[0] + breakpoints[1])
    use_f1 = f1(float(midpoint)) >= f2(float(midpoint))
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=False):
        mask = (left <= xx) & (xx <= right)
        old_values[mask] = f1_values[mask] if use_f1 else f2_values[mask]
        use_f1 = not use_f1

    fixed = f1.maximum(f2)

    print("Raw roots of sin(3*x) + sin(x):")
    print(roots)
    print()
    print("Fixed maximum domain:")
    print(fixed.domain)
    print()
    print(fixed)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(xx, f1_values, label="sin(3x)")
    axes[0].plot(xx, f2_values, label="-sin(x)")
    axes[0].plot(xx, diff(xx), color="0.3", linewidth=1, label="difference")
    axes[0].axhline(0, color="0.7", linewidth=1)
    axes[0].plot(roots, np.zeros_like(roots), "rx", label="raw roots")
    axes[0].set_title("Raw roots include near-duplicate tangential contacts")
    axes[0].legend()

    axes[1].plot(xx, np.maximum(f1_values, f2_values), "k", linewidth=3, label="true max")
    axes[1].plot(xx, old_values, "--", label="old alternating switch behavior")
    axes[1].plot(xx, fixed(xx), ":", linewidth=3, label="fixed maximum()")
    for root in roots:
        axes[1].axvline(root, color="r", alpha=0.2)
    axes[1].set_title("Only sign-changing roots become branch switches")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
