# Unit Testing Agent

## Purpose
This agent prescribes the methodology and standards for writing unit tests in the chebpy project. The goal is to achieve and maintain high code coverage while following a clear, consistent test structure.

**Important:** This agent writes and modifies TEST CODE ONLY. Any changes to source code in `src/chebpy/` MUST be explicitly requested and approved by the user. Never modify production code without explicit permission.

## Permissions and Scope

### What This Agent Can Do ✅
- Write new test files in `tests/`
- Modify existing test files in `tests/`
- Refactor and improve test organization
- Add fixtures, test classes, and test methods
- Update test documentation
- Run tests and analyze coverage reports

### What This Agent Cannot Do Without Explicit Permission ❌
- Modify any file in `src/chebpy/`
- Change production code to make tests pass
- Add new source modules
- Refactor production code
- Change function signatures or APIs

### When Source Changes Are Needed
If tests reveal that source code needs modification:
1. Report the issue to the user
2. Explain what source change would be needed
3. Wait for explicit approval
4. Only make the change after user grants permission

## Test Structure Principles

### One-to-One Module Correspondence
Tests MUST have a one-to-one correspondence with source modules. Each source
file maps to exactly one flat test file in `tests/`:

```
src/chebpy/chebfun.py         → tests/test_chebfun.py
src/chebpy/bndfun.py          → tests/test_bndfun.py
src/chebpy/chebtech.py        → tests/test_chebtech.py
src/chebpy/chebyshev.py       → tests/test_chebyshev.py
src/chebpy/algorithms.py      → tests/test_algorithms.py
src/chebpy/api.py             → tests/test_api.py
src/chebpy/decorators.py      → tests/test_decorators.py
src/chebpy/plotting.py        → tests/test_plotting.py
src/chebpy/quasimatrix.py     → tests/test_quasimatrix.py
src/chebpy/utilities.py       → tests/test_utilities.py
src/chebpy/settings.py        → tests/test_settings.py
src/chebpy/exceptions.py      → tests/test_exceptions.py
src/chebpy/fun.py             → tests/test_fun.py
src/chebpy/onefun.py          → tests/test_onefun.py
src/chebpy/smoothfun.py       → tests/test_smoothfun.py
src/chebpy/classicfun.py      → tests/test_classicfun.py
```

Convolution tests live inside `tests/test_algorithms.py` (covering
Hale–Townsend convolution algorithms alongside the other algorithm tests).

Abstract base classes (`fun.py`, `onefun.py`, `smoothfun.py`, `classicfun.py`)
have minimal stub test files verifying they cannot be instantiated directly,
and are tested thoroughly through their concrete subclasses.

**Important:** When adding a new module to `src/chebpy/`, you MUST create a corresponding test file:
- New module: `src/chebpy/new_module.py`
- New test file: `tests/test_new_module.py`

This maintains the one-to-one structure and ensures all code has dedicated test coverage.

### Shared Test Fixtures
- **`tests/conftest.py`** - Shared test fixtures and utilities for all tests
- **`tests/utilities.py`** - Helper functions for test data generation and validation

### Generic (Reusable) Test Suites

The `tests/generic/` directory contains reusable test functions that run
across multiple concrete implementations (Bndfun, Chebfun, Chebtech).
Each flat test file selectively imports the specific generic functions it needs
via `from tests.generic.<module> import <function>`.
The `conftest.py` fixtures (`emptyfun`, `constfun`, `complexfun`) auto-detect
the correct type based on the requesting test module name.

**Name-collision rule (`# noqa: F811`):** When a flat test file imports a
generic test function (e.g. `test_roots` from `tests.generic.complex`) *and*
defines a class method with the same name, ruff raises F811 (redefinition of
unused name). Suppress this on the **method definition** with
`# noqa: F811` — the import is still collected by pytest at module level while
the class method runs in its own scope:

```python
from tests.generic.complex import test_roots  # noqa: F401

class TestRoots:
    @pytest.mark.parametrize(("f", "roots"), rootstestfuns)
    def test_roots(self, f, roots):  # noqa: F811
        ...
```

### Test Organization Within Files

Each test file should be organized with:

1. **Module docstring** describing what is being tested
2. **Imports** - all at the top of the file
3. **Fixtures** - pytest fixtures for reusable test data (if needed)
4. **Test classes** - organized by logical grouping
5. **Test methods** - descriptive names starting with `test_`

Example structure:
```python
"""Tests for the module_name module.

This module tests:
- Feature 1
- Feature 2
- Edge cases and error handling
"""

import pytest
import numpy as np
from chebpy import Chebfun

# Fixtures (if needed)
@pytest.fixture
def example_chebfun():
    return Chebfun.initfun_adaptive(lambda x: np.sin(x), [-1, 1])

# Test classes organized by feature/component
class TestFeatureName:
    """Test FeatureName functionality."""
    
    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        f = Chebfun.initfun_adaptive(lambda x: x**2)
        
        # Act
        result = f(0.5)
        
        # Assert
        assert result == pytest.approx(0.25)
```

## Coverage Requirements

### Target: 100% Code Coverage
Every line of production code must be covered by at least one test.

### Running Coverage
```bash
# Run full test suite with coverage
make test

# Run specific module coverage
uvx pytest --cov=src/chebpy/module_name --cov-report=term-missing tests/test_module_name.py

# Check overall coverage
uvx pytest --cov=src/chebpy --cov-report=term-missing tests/
```

### Coverage Verification
After writing tests:
1. Run `make test` to verify all tests pass
2. Check coverage report shows 100% for the modified module
3. Ensure no regressions in other modules

## Test Writing Guidelines

### Test Naming
- Test methods: `test_<what_is_being_tested>`
- Test classes: `Test<ComponentName>`
- Be descriptive: `test_quoted_line_validates_quarter_point_precision` is better than `test_validation`

### Test Structure (AAA Pattern)
```python
def test_something(self):
    """Test description."""
    # Arrange - set up test data
    f = Chebfun.initfun_adaptive(np.sin)
    
    # Act - execute the code being tested
    result = f(0.0)
    
    # Assert - verify the results
    assert result == pytest.approx(0.0, abs=1e-14)
```

### What to Test

#### Happy Path
- Normal operation with valid inputs
- Expected return values and side effects

#### Edge Cases
- Boundary values (0, empty, max values)
- Special cases specific to domain

#### Error Handling
- Invalid inputs raise appropriate exceptions
- Error messages are clear and helpful

#### Integration Points
- Cross-module interactions
- Data flow between components

### Fixtures vs Direct Instantiation
Use fixtures when:
- Test data is reused across multiple tests
- Setup is complex or expensive
- Teardown is needed

Use direct instantiation when:
- Test is simple and isolated
- Data is specific to one test

## Testing Best Practices

### Do's ✅
- Write tests before or alongside production code (TDD when appropriate)
- Test one thing per test method
- Use descriptive assertion messages when helpful
- Use pytest.approx() for floating point comparisons
- Parametrize tests for multiple similar cases
- Keep tests independent (no shared state between tests)
- Test both success and failure paths

### Don'ts ❌
- Don't test implementation details, test behavior
- Don't write tests that depend on execution order
- Don't use time.sleep() - use proper mocking/fixtures
- Don't test external dependencies directly - mock them
- Don't duplicate test code - use fixtures or helper functions
- Don't leave commented-out test code

### Floating Point Comparisons

**Important:** When testing Chebfun operations, always use `pytest.approx()` for floating point comparisons to account for numerical precision:

```python
def test_evaluation_precision(self):
    """Test evaluation with proper floating point tolerance."""
    f = Chebfun.initfun_adaptive(np.sin, [-np.pi, np.pi])
    
    # Use pytest.approx() for floating point values
    assert f(0.0) == pytest.approx(0.0, abs=1e-14)
    assert f(np.pi/2) == pytest.approx(1.0, abs=1e-14)
```

This applies to:
- All numerical comparisons in tests
- Assertions verifying function evaluation
- Comparison of Chebfun coefficients or properties

## Common Testing Patterns

### Testing Abstract Classes
```python
def test_abstract_class_cannot_be_instantiated(self):
    """Test that abstract base class raises TypeError."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractClass()
```

### Testing Exceptions
```python
def test_invalid_input_raises_value_error(self):
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="expected pattern"):
        function_that_should_fail(invalid_input)
```

### Testing Numpy Arrays
```python
def test_array_output(self):
    """Test that function returns correct array."""
    f = Chebfun.initfun_adaptive(np.cos, [-np.pi, np.pi])
    x = np.linspace(-np.pi, np.pi, 50)
    result = f(x)
    
    # Check shape
    assert result.shape == (50,)
    
    # Check specific values with tolerance
    assert result[0] == pytest.approx(np.cos(-np.pi))
    
    # Compare entire arrays
    np.testing.assert_allclose(result, np.cos(x), atol=1e-12)
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input_value,expected", [
    (0, 0),
    (1, 2),
    (2, 4),
    (-1, -2),
])
def test_multiple_cases(input_value, expected):
    """Test function with multiple input/output pairs."""
    assert function(input_value) == expected
```

## Module-Specific Guidelines

### tests/test_chebfun.py
Tests for the Chebfun class and variants:
- Construction from functions and values
- Arithmetic operations (add, subtract, multiply, divide)
- Calculus operations (differentiation, integration)
- Evaluation and composition
- Roots and extrema finding
- Plotting and visualization
- Complex-valued functions
- Domain breaking and piecewise definitions
- Private methods and additional coverage

### tests/test_bndfun.py
Tests for boundary-value functions:
- Construction and initialization
- Algebra and calculus operations
- Evaluation and properties
- Plotting
- Complex functions
- Roots and extrema
- Universal functions (ufuncs)

### tests/test_chebtech.py
Tests for low-level Chebyshev technology:
- Chebyshev point generation
- Point selection algorithms
- Coefficient-to-value conversions
- Barycentric interpolation
- Recursive construction

### tests/test_chebyshev.py
Tests for the ChebyshevPolynomial class:
- Class properties and usage (equality, evaluation, degree, copy)
- Arithmetic operations (add, sub, mul, scalar ops)
- Construction via factory functions (from_coefficients, from_values, from_roots, from_constant)
- Complex coefficient handling
- Calculus (differentiation and integration)
- Root-finding
- Plotting

### test_algorithms.py
Tests for core algorithms:
- Barycentric interpolation and Clenshaw evaluation
- Coefficient manipulation (coeffmult, vals/coeffs conversions)
- Adaptive construction
- Convergence and accuracy
- Chebyshev-to-Legendre coefficient conversion (cheb2leg)
- Legendre-to-Chebyshev coefficient conversion (leg2cheb)
- Low-level Legendre convolution (_conv_legendre)
- User-facing `Chebfun.conv()` method

### test_api.py
Tests for user-facing factory functions:
- `chebfun()` constructor with various input types (callable, string, float)
- `pwc()` piecewise constant constructor
- Domain handling and serialization round-trips
- Version information

### test_decorators.py
Tests for decorator utilities:
- `cache` method output caching
- `self_empty` empty-object handling
- `preandpostprocess` edge-case handling for bary/clenshaw
- `float_argument` scalar/array consistency
- `cast_arg_to_chebfun` and `cast_other` type conversion

### test_plotting.py
Tests for plotting module functions:
- `plotfun` real and complex function plots
- `plotfuncoeffs` semilogy coefficient plots
- Axes creation and parameter handling

### test_quasimatrix.py
Tests for quasimatrix operations:
- Construction and manipulation
- Matrix operations (QR, SVD)
- Integration, differentiation, and least-squares (polyfit)

### test_utilities.py
Tests for utility functions:
- `Interval` and `Domain` classes
- `infnorm`, `check_funs`, `compute_breakdata`
- Domain composition and arithmetic

### test_settings.py
Tests for configuration:
- `UserPreferences` access and reset
- Context manager for temporary settings

### test_exceptions.py
Tests for exception classes:
- Custom exception types
- Exception inheritance
- Error message validation

### test_fun.py, test_onefun.py, test_smoothfun.py, test_classicfun.py
Minimal tests for abstract base classes:
- Verify that each ABC cannot be instantiated directly
- Concrete behavior tested indirectly through subclass test files

## Continuous Improvement

### Adding New Features
When adding tests for new production code:
1. **If creating a new module:** Confirm with user, then create the corresponding test file (e.g., `src/chebpy/new_module.py` → `tests/test_new_module.py`)
2. Write tests in the corresponding test file
3. Ensure high code coverage of new code
4. Run full test suite to check for regressions
5. Update this agent guide if new patterns emerge

**Note:** Only create new source modules (`src/chebpy/*.py`) if explicitly requested by the user. This agent's primary role is test creation, not production code.

### Refactoring Tests
If test structure needs improvement:
1. Maintain one-to-one module correspondence
2. Keep all tests passing during refactoring
3. Improve organization and clarity
4. Update documentation

### Reviewing Tests
When reviewing test PRs, check for:
- Correct test file (matches module)
- 100% coverage of changes
- Clear, descriptive test names
- Proper use of fixtures
- No test interdependencies
- Follows AAA pattern

## Troubleshooting

### Tests Failing After Changes
1. Run single test: `pytest tests/test_file.py::TestClass::test_method -v`
2. Check error message carefully
3. Verify test data matches new behavior
4. Update tests if behavior intentionally changed

### Coverage Not 100%
1. Run with missing lines: `pytest --cov=src/chebpy --cov-report=term-missing`
2. Identify uncovered lines in report
3. Write tests specifically for those lines
4. Consider if code is dead code that should be removed

### Tests Too Slow
1. Identify slow tests: `uvx pytest --durations=10`
2. Consider using lower polynomial degrees or smaller domains
3. Mock expensive operations like adaptive construction when testing logic
4. Use fixtures to avoid repeated Chebfun construction

## Summary

The key principles for unit testing in chebpy:
1. **High code coverage** - comprehensive testing of production code - 100% coverage is the goal for all modules
2. **One-to-one structure** - each source module has corresponding test file(s)
3. **Clear organization** - tests grouped logically by feature/component (by class, by operation type)
4. **Descriptive names** - tests clearly indicate what they verify
5. **Independent tests** - no shared state or execution order dependencies
6. **Numerical rigor** - proper use of floating point tolerances with `pytest.approx()` and `np.testing`
7. **Quality over quantity** - meaningful tests that verify behavior, not implementation

By following these guidelines, we maintain a robust, maintainable test suite that gives us confidence in our code.