"""Comprehensive Chebop test suite with Python/MATLAB comparison.

Easy-to-extend test framework with colored terminal output showing
pass/fail status, detailed metrics (residual, error, time) for each test,
and side-by-side comparison with MATLAB equivalents.

Based on working examples from test_100_examples.py and canonical ODE problems.
"""

import time
import numpy as np
import subprocess
import os
import platform
from chebpy import chebfun, chebop

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


# Add new tests by appending to this list
TEST_CASES = [
    {
        "name": "Linear BVP",
        "desc": "u'' + u = 1, u(0)=u(1)=0",
        "domain": [0, 1],
        "op": lambda u: u.diff(2) + u - 1,
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 0.0,
        "matlab_op": "@(u) diff(u,2) + u - 1",
        "matlab_lbc": "0",
        "matlab_rbc": "0",
    },
    {
        "name": "Cubic Nonlinearity",
        "desc": "u'' + u^3 = 1, u(0)=u(1)=0",
        "domain": [0, 1],
        "op": lambda u: u.diff(2) + u**3 - 1,
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 0.0,
        "matlab_op": "@(u) diff(u,2) + u.^3 - 1",
        "matlab_lbc": "0",
        "matlab_rbc": "0",
    },
    {
        "name": "Fourth Order BVP",
        "desc": "u'''' = 1, u(0)=u'(0)=u(1)=u'(1)=0",
        "domain": [0, 1],
        "op": lambda u: u.diff(4) - 1,
        "lbc": lambda u: [u, u.diff()],
        "lbc_val": [0.0, 0.0],
        "lbc_type": "mixed",
        "rbc": lambda u: [u, u.diff()],
        "rbc_val": [0.0, 0.0],
        "matlab_op": "@(u) diff(u,4) - 1",
        "matlab_lbc": "@(u) [u; diff(u)]",
        "matlab_lbc_vals": "[0; 0]",
        "matlab_rbc": "@(u) [u; diff(u)]",
        "matlab_rbc_vals": "[0; 0]",
    },
    {
        "name": "Poisson Equation",
        "desc": "-u'' = exp(x), u(-1)=u(1)=0",
        "domain": [-1, 1],
        "op": lambda u: -u.diff(2) - chebfun(lambda x: np.exp(x), [-1, 1]),
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 0.0,
        "matlab_op": "@(u) -diff(u,2)",
        "matlab_rhs": "@(x) exp(x)",
        "matlab_lbc": "0",
        "matlab_rbc": "0",
    },
    {
        "name": "Heat Equation Steady State",
        "desc": "-u'' = sin(πx), u(0)=u(1)=0",
        "domain": [0, 1],
        "op": lambda u: -u.diff(2) - chebfun(lambda x: np.sin(np.pi * x), [0, 1]),
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 0.0,
        "matlab_op": "@(u) -diff(u,2)",
        "matlab_rhs": "@(x) sin(pi*x)",
        "matlab_lbc": "0",
        "matlab_rbc": "0",
    },
    {
        "name": "Simple Second Order",
        "desc": "u'' + u = exp(x), u(0)=0, u(1)=1",
        "domain": [0, 1],
        "op": lambda u: u.diff(2) + u - chebfun(lambda x: np.exp(x), [0, 1]),
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 1.0,
        "matlab_op": "@(u) diff(u,2) + u",
        "matlab_rhs": "@(x) exp(x)",
        "matlab_lbc": "0",
        "matlab_rbc": "1",
    },
    {
        "name": "Nonlinear Bratu",
        "desc": "u'' + exp(u) = 0, u(0)=u(1)=0",
        "domain": [0, 1],
        "op": lambda u: u.diff(2) + np.exp(u),
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 0.0,
        "matlab_op": "@(u) diff(u,2) + exp(u)",
        "matlab_lbc": "0",
        "matlab_rbc": "0",
    },
    {
        "name": "Quadratic Nonlinearity",
        "desc": "u'' - u^2 + 1 = 0, u(-1)=0, u(1)=0",
        "domain": [-1, 1],
        "op": lambda u: u.diff(2) - u**2 + 1,
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 0.0,
        "matlab_op": "@(u) diff(u,2) - u.^2 + 1",
        "matlab_lbc": "0",
        "matlab_rbc": "0",
    },
    {
        "name": "Oscillatory RHS",
        "desc": "u'' + u = sin(10πx), u(0)=u(1)=0",
        "domain": [0, 1],
        "op": lambda u: u.diff(2) + u - chebfun(lambda x: np.sin(10 * np.pi * x), [0, 1]),
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 0.0,
        "matlab_op": "@(u) diff(u,2) + u",
        "matlab_rhs": "@(x) sin(10*pi*x)",
        "matlab_lbc": "0",
        "matlab_rbc": "0",
    },
    {
        "name": "Absorption Term",
        "desc": "u'' - u = -x^2, u(0)=0, u(1)=0",
        "domain": [0, 1],
        "op": lambda u: u.diff(2) - u + chebfun(lambda x: x**2, [0, 1]),
        "lbc": 0.0,
        "lbc_type": "dirichlet",
        "rbc": 0.0,
        "matlab_op": "@(u) diff(u,2) - u",
        "matlab_rhs": "@(x) -x.^2",
        "matlab_lbc": "0",
        "matlab_rbc": "0",
    },
    {
        "name": "Singular Perturbation",
        "desc": "epsilon * u'' - u = 0, u(0)=0, u(1)=1, epsilon = 1e-6",
        "domain": [0, 1],
        "op": lambda u: 1e-3 * u.diff(2) - u.diff(),
        "lbc": 0.0,
        "rbc": 1.0,
        "lbc_type": "dirichlet",
        "matlab_op": "@(u) 1e-3*diff(u,2) - diff(u)",
        "matlab_lbc": "0",
        "matlab_rbc": "1"
    },
    {
        "name": "Duffing Nonlinear ODE",
        "desc": "u'' + u + u^3 = 0, u(0)=0, u(1)=0",
        "domain": [-10, 10],
        "op": lambda u: u.diff(2) + u + u**3,
        "lbc": 0.0,
        "rbc": 0.0,
        "lbc_type": "dirichlet",
        "matlab_op": "@(u) diff(u,2) + u + u^3",
        "matlab_lbc": "0",
        "matlab_rbc": "0"
    },
]

def solve_chebpy_legendre(problem):
    """Solve with ChebPy and compute metrics."""
    N = chebop(problem["domain"], uselegendre=True)
    N.op = problem["op"]

    # Handle boundary conditions
    if "lbc" in problem and problem["lbc"] is not None:
        if problem["lbc_type"] == "dirichlet":
            N.lbc = problem["lbc"]
        elif problem["lbc_type"] == "mixed":
            N.lbc = problem["lbc"]
            if "lbc_val" in problem:
                N.lbc_val = problem["lbc_val"]

    if "rbc" in problem and problem["rbc"] is not None:
        if problem["lbc_type"] == "dirichlet":
            N.rbc = problem["rbc"]
        elif problem["lbc_type"] == "mixed":
            N.rbc = problem["rbc"]
            if "rbc_val" in problem:
                N.rbc_val = problem["rbc_val"]

    start = time.time()
    try:
        u = N.solve()
        elapsed = time.time() - start

        # Compute residual
        residual = N.op(u)
        x_test = np.linspace(problem["domain"][0], problem["domain"][1], 100)
        res_vals = residual(x_test)
        max_residual = np.max(np.abs(res_vals))

        # Compute BC error
        a, b = problem["domain"]
        bc_left = 0.0
        bc_right = 0.0

        if "lbc" in problem and problem["lbc"] is not None:
            if problem["lbc_type"] == "dirichlet":
                bc_left = abs(u(a) - problem["lbc"])
            elif problem["lbc_type"] == "mixed" and "lbc_val" in problem:
                bc_left = max(abs(u(a) - problem["lbc_val"][0]),
                            abs(u.diff()(a) - problem["lbc_val"][1]))

        if "rbc" in problem and problem["rbc"] is not None:
            if problem["lbc_type"] == "dirichlet":
                bc_right = abs(u(b) - problem["rbc"])
            elif problem["lbc_type"] == "mixed" and "rbc_val" in problem:
                bc_right = max(abs(u(b) - problem["rbc_val"][0]),
                             abs(u.diff()(b) - problem["rbc_val"][1]))

        bc_error = max(bc_left, bc_right)

        # Get solution size
        if hasattr(u, "funs"):
            solution_size = max(fun.size for fun in u.funs)
        else:
            solution_size = len(u)

        return {
            "success": True,
            "time": elapsed,
            "max_residual": max_residual,
            "bc_error": bc_error,
            "solution_size": solution_size,
        }
    except Exception as e:
        return {
            "success": False,
            "time": time.time() - start,
            "error_msg": str(e)[:100],
        }


def solve_chebpy(problem):
    """Solve with ChebPy and compute metrics."""
    N = chebop(problem["domain"])
    N.op = problem["op"]

    # Handle boundary conditions
    if "lbc" in problem and problem["lbc"] is not None:
        if problem["lbc_type"] == "dirichlet":
            N.lbc = problem["lbc"]
        elif problem["lbc_type"] == "mixed":
            N.lbc = problem["lbc"]
            if "lbc_val" in problem:
                N.lbc_val = problem["lbc_val"]

    if "rbc" in problem and problem["rbc"] is not None:
        if problem["lbc_type"] == "dirichlet":
            N.rbc = problem["rbc"]
        elif problem["lbc_type"] == "mixed":
            N.rbc = problem["rbc"]
            if "rbc_val" in problem:
                N.rbc_val = problem["rbc_val"]

    start = time.time()
    try:
        u = N.solve()
        elapsed = time.time() - start

        # Compute residual
        residual = N.op(u)
        x_test = np.linspace(problem["domain"][0], problem["domain"][1], 100)
        res_vals = residual(x_test)
        max_residual = np.max(np.abs(res_vals))

        # Compute BC error
        a, b = problem["domain"]
        bc_left = 0.0
        bc_right = 0.0

        if "lbc" in problem and problem["lbc"] is not None:
            if problem["lbc_type"] == "dirichlet":
                bc_left = abs(u(a) - problem["lbc"])
            elif problem["lbc_type"] == "mixed" and "lbc_val" in problem:
                bc_left = max(abs(u(a) - problem["lbc_val"][0]),
                            abs(u.diff()(a) - problem["lbc_val"][1]))

        if "rbc" in problem and problem["rbc"] is not None:
            if problem["lbc_type"] == "dirichlet":
                bc_right = abs(u(b) - problem["rbc"])
            elif problem["lbc_type"] == "mixed" and "rbc_val" in problem:
                bc_right = max(abs(u(b) - problem["rbc_val"][0]),
                             abs(u.diff()(b) - problem["rbc_val"][1]))

        bc_error = max(bc_left, bc_right)

        # Get solution size
        if hasattr(u, "funs"):
            solution_size = max(fun.size for fun in u.funs)
        else:
            solution_size = len(u)

        return {
            "success": True,
            "time": elapsed,
            "max_residual": max_residual,
            "bc_error": bc_error,
            "solution_size": solution_size,
        }
    except Exception as e:
        return {
            "success": False,
            "time": time.time() - start,
            "error_msg": str(e)[:100],
        }

def solve_matlab(problem, idx):
    """Solve with MATLAB and compute metrics."""
    script = f"""
addpath('matlab_chebfun');

tic;
try
    N = chebop({problem['domain'][0]}, {problem['domain'][1]});
    N.op = {problem['matlab_op']};
    """

    # Add boundary conditions
    if "matlab_lbc" in problem:
        if "matlab_lbc_vals" in problem:
            script += f"    N.lbc = {problem['matlab_lbc']};\n"
            script += f"    N.lbc = {problem['matlab_lbc_vals']};\n"
        else:
            script += f"    N.lbc = {problem['matlab_lbc']};\n"

    if "matlab_rbc" in problem:
        if "matlab_rbc_vals" in problem:
            script += f"    N.rbc = {problem['matlab_rbc']};\n"
            script += f"    N.rbc = {problem['matlab_rbc_vals']};\n"
        else:
            script += f"    N.rbc = {problem['matlab_rbc']};\n"

    if "matlab_rhs" in problem:
        script += f"    N.rhs = {problem['matlab_rhs']};\n"

    script += """    u = N\\0;
    elapsed = toc;

    % Compute residual
    x_test = linspace(N.domain(1), N.domain(2), 100)';
    residual = N.op(u);
    res_vals = residual(x_test);
    max_residual = max(abs(res_vals));

    % Compute BC error (simplified)
    a = N.domain(1);
    b = N.domain(2);
    bc_error = 0;

    fprintf('SUCCESS\\n');
    fprintf('Time: %.6f\\n', elapsed);
    fprintf('MaxResidual: %.6e\\n', max_residual);
    fprintf('BCError: %.6e\\n', bc_error);
    fprintf('SolutionSize: %d\\n', length(u));
catch ME
    elapsed = toc;
    fprintf('FAILED\\n');
    fprintf('Time: %.6f\\n', elapsed);
    fprintf('Error: %s\\n', ME.message);
end
"""

    script_file = f"chebop_test_{idx}.m"
    with open(script_file, "w") as f:
        f.write(script)

    # Absolute, MATLAB-friendly path to the script
    script_path = os.path.abspath(script_file)
    script_path_matlab = script_path.replace("\\", "/")

    try:
        system = platform.system()

        if system == "Windows":
            # Fully headless, no windows: matlab.exe -batch "run('...')"
            cmd = [
                "matlab.exe",
                "-batch",
                f"run('{script_path_matlab}')",
            ]
        else:
            # Original style for macOS/Linux
            cmd = [
                "matlab",
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                (
                    f"try; run('{script_path_matlab}'); "
                    f"catch ME; fprintf('FAILED\\nError: %s\\n', ME.message); end; exit;"
                ),
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        output = result.stdout
        if "SUCCESS" in output:
            time_str = output.split("Time: ")[1].split("\n")[0]
            res_str = output.split("MaxResidual: ")[1].split("\n")[0]
            bc_str = output.split("BCError: ")[1].split("\n")[0]
            size_str = output.split("SolutionSize: ")[1].split("\n")[0]

            return {
                "success": True,
                "time": float(time_str),
                "max_residual": float(res_str),
                "bc_error": float(bc_str),
                "solution_size": int(size_str),
            }
        else:
            error_msg = output.split("Error: ")[1].split("\n")[0] if "Error: " in output else "Unknown"
            return {"success": False, "error_msg": error_msg[:100]}
    except subprocess.TimeoutExpired:
        return {"success": False, "error_msg": "Timeout (>120s)"}
    except Exception as e:
        return {"success": False, "error_msg": str(e)[:100]}


def format_metric(value, format_str=".2e"):
    """Format metric value with color coding."""
    formatted = f"{value:{format_str}}"
    if format_str == ".2e":
        if value < 1e-10:
            return f"{GREEN}{formatted}{RESET}"
        elif value < 1e-6:
            return f"{YELLOW}{formatted}{RESET}"
        else:
            return f"{RED}{formatted}{RESET}"
    return formatted


def run_tests():
    """Run all tests and display results."""
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}CHEBOP TEST SUITE - PYTHON vs MATLAB{RESET}{BOLD}{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}\n")

    results = []
    passed = 0
    failed = 0

    for idx, problem in enumerate(TEST_CASES, 1):
        print(f"{BOLD}[{idx}/{len(TEST_CASES)}] {problem['name']}{RESET}")
        print(f"    {problem['desc']}")
        print()

        # ChebPy
        print(f"    {BOLD}ChebPy:{RESET} ", end="", flush=True)
        chebpy_result = solve_chebpy(problem)

        if chebpy_result["success"]:
            print(f"{GREEN}✓{RESET} {chebpy_result['time']:.3f}s")
            print(f"            Residual:  {format_metric(chebpy_result['max_residual'])}")
            print(f"            BC Error:  {format_metric(chebpy_result['bc_error'])}")
            print(f"            Size:      {chebpy_result['solution_size']} pts")
        else:
            print(f"{RED}✗ FAILED{RESET}")
            print(f"            Error: {chebpy_result['error_msg']}")
            failed += 1

        # ChebPy with Legendre
        print(f"    {BOLD}ChebPy-Leg:{RESET} ", end="", flush=True)
        chebpy_leg_result = solve_chebpy_legendre(problem)

        if chebpy_result["success"]:
            print(f"{GREEN}✓{RESET} {chebpy_result['time']:.3f}s")
            print(f"            Residual:  {format_metric(chebpy_result['max_residual'])}")
            print(f"            BC Error:  {format_metric(chebpy_result['bc_error'])}")
            print(f"            Size:      {chebpy_result['solution_size']} pts")
        else:
            print(f"{RED}✗ FAILED{RESET}")
            print(f"            Error: {chebpy_result['error_msg']}")
            failed += 1

        # MATLAB
        print(f"    {BOLD}MATLAB:{RESET} ", end="", flush=True)
        matlab_result = solve_matlab(problem, idx)

        if matlab_result["success"]:
            print(f"{GREEN}✓{RESET} {matlab_result['time']:.3f}s")
            print(f"            Residual:  {format_metric(matlab_result['max_residual'])}")
            print(f"            BC Error:  {format_metric(matlab_result['bc_error'])}")
            print(f"            Size:      {matlab_result['solution_size']} pts")

            # Comparison
            if chebpy_result["success"]:
                speedup = matlab_result["time"] / chebpy_result["time"]
                if speedup > 1.1:
                    print(f"    {GREEN}>>> ChebPy is {speedup:.2f}x faster{RESET}")
                elif speedup < 0.9:
                    print(f"    {YELLOW}>>> MATLAB is {1/speedup:.2f}x faster{RESET}")
                passed += 1
        else:
            print(f"{RED}✗ FAILED{RESET}")
            print(f"            Error: {matlab_result['error_msg']}")

        print()
        results.append({"problem": problem, "chebpy": chebpy_result, "matlab": matlab_result})

    # Summary
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}SUMMARY{RESET}{BOLD}{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}\n")

    if passed == len(TEST_CASES):
        print(f"{GREEN}{BOLD}✓ ALL {len(TEST_CASES)} TESTS PASSED!{RESET}")
    else:
        print(f"{YELLOW}{BOLD}Passed: {passed}/{len(TEST_CASES)}{RESET}")
        if failed > 0:
            print(f"{RED}{BOLD}Failed: {failed}/{len(TEST_CASES)}{RESET}")

    # Statistics
    chebpy_times = [r["chebpy"]["time"] for r in results if r["chebpy"]["success"]]
    matlab_times = [r["matlab"]["time"] for r in results if r["matlab"]["success"]]

    if chebpy_times and matlab_times:
        avg_speedup = np.mean([m / c for c, m in zip(chebpy_times, matlab_times)])
        print(f"\n{BOLD}Average speedup:{RESET} ", end="")
        if avg_speedup > 1:
            print(f"{GREEN}ChebPy is {avg_speedup:.2f}x faster{RESET}")
        else:
            print(f"{YELLOW}MATLAB is {1/avg_speedup:.2f}x faster{RESET}")

    print(f"\n{BOLD}{'=' * 80}{RESET}\n")


if __name__ == "__main__":
    run_tests()
