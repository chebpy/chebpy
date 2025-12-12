"""Diagnostic tools for LinOp operator analysis.

This module provides functions to detect and warn about numerical issues
in differential operator coefficients that may cause solver failures.

Issues detected:
- Vanishing highest-order coefficients (singular operators)
- Highly oscillatory coefficients (requiring high resolution)
- Near-singularities and ill-conditioning
"""

import warnings

import numpy as np

from .chebfun import Chebfun


def check_coefficient_singularities(
    coeffs: list[Chebfun], diff_order: int, domain, tol: float = 1e-8
) -> tuple[bool, list[str]]:
    """Check for singularities in operator coefficients.

    Detects:
    1. Vanishing highest-order coefficient (creates singular operator)
    2. Near-zero highest-order coefficient (ill-conditioned)
    3. Sign changes in highest-order coefficient (type-changing PDE)

    Args:
        coeffs: List of coefficient functions [a_0(x), ..., a_z(x)]
        diff_order: Highest derivative order
        domain: Domain object
        tol: Tolerance for detecting near-zero values

    Returns:
        (has_issues, warnings_list): Boolean and list of warning messages
    """
    if not coeffs or diff_order < 1:
        return False, []

    warnings_list = []
    has_issues = False

    # Get highest-order coefficient
    if diff_order >= len(coeffs):
        # No coefficient for highest derivative (implicitly 1)
        return False, []

    a_top = coeffs[diff_order]
    if a_top is None:
        return False, []

    # Evaluate at fine grid to check for issues
    a, b = domain.support
    x_test = np.linspace(a, b, 200)

    try:
        a_vals = a_top(x_test)
    except Exception:
        # Can't evaluate - skip checks
        return False, []

    # Check for vanishing coefficient
    min_abs = np.min(np.abs(a_vals))
    max_abs = np.max(np.abs(a_vals))

    if max_abs < tol:
        # Coefficient is essentially zero everywhere
        warnings_list.append(
            f"Highest-order coefficient a_{diff_order}(x) is essentially zero "
            f"(max |a_{diff_order}| = {max_abs:.2e}). This creates a singular operator "
            f"of lower order than specified."
        )
        has_issues = True
        return has_issues, warnings_list

    if min_abs < tol:
        # Coefficient vanishes somewhere
        # Find location(s) where it vanishes
        zero_locs = x_test[np.abs(a_vals) < tol]
        if len(zero_locs) > 0:
            zero_loc = zero_locs[0]
            warnings_list.append(
                f"Highest-order coefficient a_{diff_order}(x) vanishes at x ≈ {zero_loc:.3f} "
                f"(and possibly other locations). This creates a SINGULAR OPERATOR that "
                f"changes type across the domain. The spectral collocation method may fail "
                f"to properly enforce boundary conditions.\n"
                f"Suggestions:\n"
                f"  1. Split the domain at singularity locations: domain.breakpoints = [{zero_loc:.3f}]\n"
                f"  2. Reformulate as a first-order system\n"
                f"  3. Use regularization: add small epsilon to coefficient\n"
                f"  4. Check if problem is well-posed (may need different BCs on each side)"
            )
            has_issues = True

    # Check for near-singularities (very small coefficient)
    if min_abs / max_abs < 1e-3 and min_abs > tol:
        warnings_list.append(
            f"Highest-order coefficient a_{diff_order}(x) varies widely: "
            f"min|a_{diff_order}| = {min_abs:.2e}, max|a_{diff_order}| = {max_abs:.2e}. "
            f"This may cause numerical difficulties. Consider rescaling or using "
            f"adaptive domain splitting."
        )
        has_issues = True

    # Check for sign changes
    if np.any(a_vals > tol) and np.any(a_vals < -tol):
        # Coefficient changes sign - check where
        sign_changes = []
        for i in range(len(a_vals) - 1):
            if a_vals[i] * a_vals[i + 1] < 0:
                sign_changes.append(x_test[i])

        if sign_changes:
            warnings_list.append(
                f"Highest-order coefficient a_{diff_order}(x) changes sign at "
                f"x ≈ {sign_changes[0]:.3f} (and possibly other locations). "
                f"This creates a TYPE-CHANGING PDE (e.g., elliptic ↔ hyperbolic). "
                f"The problem may be ill-posed or require different boundary conditions "
                f"on each side of the transition."
            )
            has_issues = True

    return has_issues, warnings_list


def check_coefficient_oscillation(
    coeffs: list[Chebfun], diff_order: int, domain, min_points_per_wavelength: int = 10
) -> tuple[bool, list[str], int | None]:
    """Check if coefficients are highly oscillatory and estimate required resolution.

    Highly oscillatory coefficients need fine discretization grids to be
    accurately represented and integrated.

    Args:
        coeffs: List of coefficient functions [a_0(x), ..., a_z(x)]
        diff_order: Highest derivative order
        domain: Domain object
        min_points_per_wavelength: Minimum collocation points per oscillation

    Returns:
        (is_oscillatory, warnings_list, suggested_n): Boolean, warnings, and suggested grid size
    """
    if not coeffs or diff_order < 1:
        return False, [], None

    warnings_list = []
    max_n_required = 16  # Start with minimum

    # Check all coefficients for oscillation
    for k, coeff in enumerate(coeffs):
        if coeff is None:
            continue

        # Check resolution of coefficient representation
        # If coefficient needs many points, it's oscillatory
        coeff_size = 0
        if hasattr(coeff, "funs"):
            for fun in coeff.funs:
                coeff_size = max(coeff_size, fun.size)

        if coeff_size > 50:
            # Highly oscillatory coefficient
            # Rough estimate: n Chebyshev points can resolve ~n/2 oscillations
            num_oscillations = coeff_size / 2

            # For spectral accuracy, need at least min_points_per_wavelength per oscillation
            suggested_n = int(num_oscillations * min_points_per_wavelength)

            # For higher derivative orders, need even more resolution
            # because differentiation amplifies high-frequency errors
            suggested_n *= 1 + diff_order // 2

            max_n_required = max(max_n_required, suggested_n)

            warnings_list.append(
                f"Coefficient a_{k}(x) is highly oscillatory (needs {coeff_size} points "
                f"to represent). This requires fine discretization for accurate solution.\n"
                f"Estimated {num_oscillations:.0f} oscillations over domain.\n"
                f"Suggested minimum n = {suggested_n} for this coefficient."
            )

    is_oscillatory = max_n_required > 64

    if is_oscillatory:
        warnings_list.append(
            f"\nRECOMMENDATION: Set max_n >= {max_n_required} for this problem.\nExample: L.max_n = {max_n_required}"
        )

    return is_oscillatory, warnings_list, max_n_required if is_oscillatory else None


def check_operator_wellposedness(
    coeffs: list[Chebfun], diff_order: int, lbc, rbc, domain, bc=None
) -> tuple[bool, list[str]]:
    """Check if operator with given BCs is well-posed.

    Well-posedness requires:
    1. Correct number of boundary conditions (= diff_order)
    2. Non-degenerate highest-order coefficient
    3. Boundary conditions that properly constrain the problem

    Args:
        coeffs: List of coefficient functions
        diff_order: Highest derivative order
        lbc: Left boundary condition(s)
        rbc: Right boundary condition(s)
        domain: Domain object
        bc: String BC specification (e.g., 'periodic') or None

    Returns:
        (is_wellposed, warnings_list): Boolean and list of warning messages
    """
    warnings_list = []

    # Check for periodic BCs - these provide diff_order constraints
    is_periodic = isinstance(bc, str) and bc.lower() == "periodic"
    if is_periodic:
        # Periodic BCs provide diff_order constraints automatically
        # No additional checks needed - periodic is always well-posed for the given order
        return True, warnings_list

    # Count boundary conditions
    num_lbc = 0
    if lbc is not None:
        if isinstance(lbc, (list, tuple)):
            # Count non-None entries
            num_lbc = sum(1 for bc_item in lbc if bc_item is not None)
        else:
            num_lbc = 1

    num_rbc = 0
    if rbc is not None:
        if isinstance(rbc, (list, tuple)):
            num_rbc = sum(1 for bc_item in rbc if bc_item is not None)
        else:
            num_rbc = 1

    total_bcs = num_lbc + num_rbc

    if total_bcs < diff_order:
        warnings_list.append(
            f"UNDERDETERMINED SYSTEM: Differential order is {diff_order} but only "
            f"{total_bcs} boundary conditions provided ({num_lbc} left, {num_rbc} right). "
            f"Need {diff_order} boundary conditions for well-posedness. "
            f"Solution will not be unique."
        )
        return False, warnings_list

    if total_bcs > diff_order:
        warnings_list.append(
            f"OVERDETERMINED SYSTEM: Differential order is {diff_order} but "
            f"{total_bcs} boundary conditions provided ({num_lbc} left, {num_rbc} right). "
            f"System may be inconsistent. Typically need exactly {diff_order} BCs."
        )
        # Not necessarily ill-posed, but warn

    return True, warnings_list


def check_periodic_compatibility(linop) -> tuple[bool, list[str]]:
    """Check compatibility conditions for periodic BCs.

    For differential equations with periodic BCs, certain compatibility
    conditions must be satisfied for a solution to exist.

    For u^(n) + ... = f with periodic BCs:
    - Integrating n times shows that ∫f must satisfy constraints
    - For u'' = f: Need ∫f dx = 0 over the period

    Args:
        linop: LinOp object with periodic BCs

    Returns:
        (is_compatible, warnings_list): Boolean and list of warning messages
    """
    warnings_list = []

    # Only check if periodic BCs
    is_periodic = isinstance(linop.bc, str) and linop.bc.lower() == "periodic"
    if not is_periodic:
        return True, warnings_list

    # Only check for second-order operators (most common case)
    if linop.blocks is None or not linop.blocks:
        return True, warnings_list

    block = linop.blocks[0]
    diff_order = block["diff_order"]

    if diff_order != 2:
        # For other orders, compatibility is more complex - skip for now
        return True, warnings_list

    # Check if RHS is provided
    if linop.rhs is None:
        return True, warnings_list

    # For u'' = f with periodic BCs, need ∫f dx = 0
    try:
        rhs_integral = linop.rhs.sum()
        tol = 1e-8  # Reasonable tolerance

        if abs(rhs_integral) > tol:
            warnings_list.append(
                f"PERIODIC COMPATIBILITY ERROR: For u'' = f with periodic BCs, "
                f"the RHS must satisfy ∫f dx = 0 over the period. "
                f"Found ∫f dx = {rhs_integral:.6e}, which violates this condition. "
                f"The problem is mathematically ILL-POSED and has no periodic solution. "
                f"Either: (1) use non-periodic BCs, or (2) modify RHS to have zero integral."
            )
            return False, warnings_list

    except Exception:
        # If we can't compute integral, skip check
        pass

    return True, warnings_list


def diagnose_linop(linop, verbose: bool = True) -> bool:
    """Run all diagnostic checks on a LinOp and emit warnings.

    This is the main entry point for operator diagnostics. It checks:
    1. Coefficient singularities and vanishing
    2. Coefficient oscillation and resolution requirements
    3. Well-posedness of boundary value problem
    4. Periodic compatibility conditions

    Args:
        linop: LinOp object to diagnose
        verbose: If True, print warnings immediately; if False, only return status

    Returns:
        has_issues: True if any issues were detected
    """
    if linop.blocks is None:
        linop.prepare_domain()

    if not linop.blocks:
        return False

    # Get coefficients from first block (typically single-interval problems)
    block = linop.blocks[0]
    coeffs = block["coeffs"]
    diff_order = block["diff_order"]

    has_any_issues = False

    # Check 1: Singularities
    has_sing, sing_warnings = check_coefficient_singularities(coeffs, diff_order, linop.domain)
    if has_sing:
        has_any_issues = True
        if verbose:
            for msg in sing_warnings:
                warnings.warn(msg, UserWarning, stacklevel=2)

    # Check 2: Oscillation
    is_osc, osc_warnings, suggested_n = check_coefficient_oscillation(coeffs, diff_order, linop.domain)
    if is_osc:
        has_any_issues = True
        if verbose:
            for msg in osc_warnings:
                warnings.warn(msg, UserWarning, stacklevel=2)

        # Update max_n if not already set high enough
        if suggested_n and linop.max_n < suggested_n:
            if verbose:
                warnings.warn(
                    f"Automatically increasing max_n from {linop.max_n} to {suggested_n} "
                    f"to handle oscillatory coefficients.",
                    UserWarning,
                    stacklevel=2,
                )
            linop.max_n = suggested_n

    # Check 3: Well-posedness
    is_wellposed, wellposed_warnings = check_operator_wellposedness(
        coeffs, diff_order, linop.lbc, linop.rbc, linop.domain, linop.bc
    )
    if not is_wellposed:
        has_any_issues = True
        if verbose:
            for msg in wellposed_warnings:
                warnings.warn(msg, UserWarning, stacklevel=2)

    # Check 4: Periodic compatibility
    is_compatible, compat_warnings = check_periodic_compatibility(linop)
    if not is_compatible:
        has_any_issues = True
        if verbose:
            for msg in compat_warnings:
                warnings.warn(msg, UserWarning, stacklevel=2)
        # For compatibility errors, raise an exception to fail fast
        if compat_warnings:
            raise ValueError(compat_warnings[0])

    return has_any_issues
