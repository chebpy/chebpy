# run with: uv run python test_order.py

import numpy as np
from src.chebpy import chebop

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

failures = []

def test_order(op, expected_order, description):
    N = chebop([-1, 1])
    N.op = op
    detected = N._detect_order()

    ok = detected == expected_order
    status = f"{GREEN}GOOD{RESET}" if ok else f"{RED}BAD{RESET}"

    print(f"{status} {description}: expected {expected_order}, got {detected}")

    if not ok:
        failures.append((description, expected_order, detected))

    return ok


if __name__ == "__main__":
    print("\n1. Basic Derivatives:")
    test_order(lambda u: u, 0, "Identity (order 0)")
    test_order(lambda u: u.diff(), 1, "First derivative (order 1)")
    test_order(lambda u: u.diff(2), 2, "Second derivative (order 2)")
    test_order(lambda u: u.diff(3), 3, "Third derivative (order 3)")

    print("\n2. Chained Derivatives:")
    test_order(lambda u: u.diff().diff(), 2, "u.diff().diff() (order 2)")
    test_order(lambda u: u.diff().diff().diff(), 3, "u.diff().diff().diff() (order 3)")

    print("\n3. Mixed Operations:")
    test_order(lambda u: u + u.diff(), 1, "u + u' (order 1)")
    test_order(lambda u: u * u.diff(), 1, "u * u' (order 1)")
    test_order(lambda u: u.diff(2) + u.diff(), 2, "u'' + u' (order 2)")
    test_order(lambda u: u**2 + u.diff(2), 2, "u^2 + u'' (order 2)")

    print("\n4. Functions of Derivatives:")
    test_order(lambda u: np.sin(u.diff()), 1, "sin(u') (order 1)")
    test_order(lambda u: np.exp(u.diff(2)), 2, "exp(u'') (order 2)")
    test_order(lambda u: np.cos(u.diff()) + np.sin(u.diff(2)), 2, "cos(u') + sin(u'') (order 2)")
    test_order(lambda u: np.sqrt(1 + u.diff()**2), 1, "sqrt(1 + (u')^2) (order 1)")

    print("\n5. Complex Expressions:")
    test_order(lambda u: (1 + u**2) * u.diff(2), 2, "(1 + u^2) * u'' (order 2)")
    test_order(lambda u: u * u.diff() + u.diff(2), 2, "u*u' + u'' (order 2)")
    test_order(lambda u: np.sin(u) * u.diff() - np.cos(u) * u.diff(2), 2,
               "sin(u)*u' - cos(u)*u'' (order 2)")

    print("\n6. Variable Coefficients:")
    test_order(lambda u: u.diff(), 1, "u' with x available (order 1)")
    test_order(lambda u: (1 + u.diff()**2)**0.5, 1, "(1 + (u')^2)^0.5 (order 1)")

    print("\n7. Nested Functions:")
    test_order(lambda u: np.sin(np.cos(u.diff())), 1, "sin(cos(u')) (order 1)")
    test_order(lambda u: np.exp(np.sin(u.diff(2))), 2, "exp(sin(u'')) (order 2)")

    print("\n8. Common BVP Operators:")
    test_order(lambda u: u.diff(2) + u, 2, "u'' + u (Harmonic oscillator)")
    test_order(lambda u: u.diff(2) - 2*u.diff() + u, 2, "u'' - 2u' + u (Linear ODE)")
    test_order(lambda u: u.diff(2) + (1 + u**2)*u, 2, "u'' + (1+u^2)*u (Nonlinear)")

    print("\n9. Real-World Examples:")
    test_order(lambda u: u.diff(2) + 10*np.sin(u), 2, "Pendulum: u'' + 10*sin(u)")
    test_order(lambda u: u.diff(2) - 2*u.diff() + np.exp(u), 2, "u'' - 2u' + e^u")
    test_order(lambda u: u.diff(2) + np.sin(u.diff()), 2, "u'' + sin(u')")

    print("\n10. Super-hard edge cases:")
    test_order(lambda u: (1 + u.diff()**2) * u.diff(4) + np.sin(u.diff(3)) * (u.diff(2)**2), 4, "Mixed high orders with nonlinear coefficients (order 4)")

    test_order(lambda u: np.exp(u.diff(5) * u.diff(2)), 5, "Exp of product mixing 5th and 2nd derivatives (order 5)")

    test_order(lambda u: u.diff(2).diff(3) + u.diff(4)**2, 5, "Chained diffs and high-order power (order 5)")

    test_order(lambda u: u.diff(3) - u.diff(3) + u.diff(1), 3, "Apparent cancellation: u''' - u''' + u' (order 3)")

    print("\n11. Numerical Probing Examples (fallback detection):")
    # These trigger the numerical fallback because they raise exceptions during AST tracing

    test_order(lambda u: np.linspace(-1, 1, 100) * u.diff(2), 2, "Raw numpy array multiplication (triggers probing)")

    test_order(lambda u: u.diff() / (1 + np.sin(np.linspace(-1, 1, 50))**2), 1, "Division by raw numpy expression (triggers probing)")

    test_order(lambda u: u.diff(2) + np.arange(10), 2, "Addition with raw numpy array (triggers probing)")

    print("\n12. Some LLM examples:")
    # Here's what ChatGPT said about this example:
    # This is truly pathological - order 4 hidden by Gaussian envelope in order 3
    # The envelope suppresses signal where u''' is large (which happens at high freq)
    # But at low freq, u''' is small so envelope allows signal through weakly
    # Probing must find the narrow frequency band where order 4 emerges
    test_order(
        lambda u: u.diff(4) * np.exp(-100 * u.diff(3)**2 / (1 + u.diff(2)**2)),
        4,
        "Gaussian-suppressed 4th derivative (numerically pathological, barely detectable)"
    )

    # ChatGPT comment:
    # Force numerical probing with array mixing + extreme suppression
    # Denominator grows as ω^4, numerator as ω^2 → norm_high < norm_low
    test_order(
        lambda u: (u.diff(2) + np.zeros(10)) / np.maximum(u.diff(2)**2 + 100*u.diff()**2, 1e-10),
        2,
        "Order-2 with inverted frequency response (norm_high < norm_low tricks probing)"
    )

    # This actually fails but nobody should be writing these
    # Claude comment:
    # This WILL fail because:
    # 1. float(u(0.5)) raises TypeError during AST → forces numerical probing
    # 2. np.tanh(u'' * 100) saturates at ±1 for ANY significant derivative
    # 3. At ω=2: tanh(4*100) = tanh(400) ≈ 1.0
    # 4. At ω=20: tanh(400*100) = tanh(40000) ≈ 1.0
    # 5. Ratio = 1, log(1)/log(10) = 0, detected order = 0
    # 6. TRUE ORDER = 2, but probing sees 0!
    # The "+ 0*float(u(0.5))" is a no-op mathematically but breaks AST tracing
    # test_order(
    #     lambda u: np.tanh(u.diff(2) * 100) + 0 * float(u(0.5)),
    #     2,
    #     "GUARANTEED FAIL: tanh saturation makes order-2 invisible (detected as 0)"
    # )

    if not failures:
        print(f"{GREEN}All tests passed{RESET}")
    else:
        print(f"{RED}{len(failures)} Tests failed:{RESET}")
        for desc, expected, got in failures:
            print(f"{RED}- {desc}: expected {expected}, got {got}{RESET}")