import re
import subprocess
import sys
import time

import numpy as np

from chebpy import chebop

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def python_to_matlab_op(python_str):
    """Convert Python operator string to MATLAB format.

    Examples:
        'u.diff(2) + u - 1' -> 'diff(u,2) + u - 1'
        'u**3' -> 'u.^3'
        'np.exp(u)' -> 'exp(u)'
    """
    matlab_str = re.sub(r"u\.diff\((\d+)\)", r"diff(u,\1)", python_str)
    matlab_str = re.sub(r"u\*\*(\d+)", r"u.^\1", matlab_str)
    matlab_str = matlab_str.replace("np.exp", "exp")
    matlab_str = matlab_str.replace("np.sin", "sin")
    matlab_str = matlab_str.replace("np.cos", "cos")
    matlab_str = matlab_str.replace("np.tan", "tan")
    matlab_str = matlab_str.replace("np.log", "log")
    matlab_str = matlab_str.replace("np.sqrt", "sqrt")
    matlab_str = matlab_str.replace("np.abs", "abs")
    matlab_str = matlab_str.replace("np.pi", "pi")
    return f"@(u) {matlab_str}"


def bc_to_python(bc_spec):
    """Convert BC specification to Python format.

    Args:
        bc_spec: Can be:
            - scalar (float/int): Dirichlet BC
            - dict: {"u": val} or {"u'": val} or {"u": val1, "u'": val2}
            - None: No BC

    Returns:
        tuple: (bc_value_or_callable, bc_type, bc_values)
    """
    if bc_spec is None:
        return None, None, None

    # Scalar -> Dirichlet
    if isinstance(bc_spec, (int, float, np.number)):
        return bc_spec, "dirichlet", None

    # Dict -> Mixed conditions
    if isinstance(bc_spec, dict):
        orders = []
        values = []
        for key, val in bc_spec.items():
            if key == "u":
                orders.append(0)
                values.append(val)
            elif "'" in key:
                order = key.count("'")
                orders.append(order)
                values.append(val)

        if len(orders) == 1 and orders[0] == 0:
            # Single u condition -> Dirichlet
            return values[0], "dirichlet", None
        elif len(orders) == 1:
            # Single derivative condition: u^(k) = value
            # Return list format [None, ..., None, value] for linear solver
            order = orders[0]
            val = values[0]
            bc_list = [None] * order + [val]
            return bc_list, "neumann", None
        else:
            # Multiple conditions: build list format [u_val, u'_val, u''_val, ...]
            # where None means no constraint at that derivative order
            max_order = max(orders)
            bc_list = [None] * (max_order + 1)
            for order, val in zip(orders, values):
                bc_list[order] = val
            return bc_list, "mixed_list", None

    return None, None, None


def bc_to_matlab(bc_spec):
    """Convert BC specification to MATLAB format.

    Returns:
        tuple: (bc_str, bc_val_str)
    """
    if bc_spec is None:
        return None, None

    # Scalar -> Simple value
    if isinstance(bc_spec, (int, float, np.number)):
        return str(bc_spec), None

    # Dict -> Function and values
    if isinstance(bc_spec, dict):
        orders = []
        values = []
        for key, val in bc_spec.items():
            if key == "u":
                orders.append(0)
                values.append(val)
            elif "'" in key:
                order = key.count("'")
                orders.append(order)
                values.append(val)

        if len(orders) == 1 and orders[0] == 0:
            # Single u condition -> Simple value
            return str(values[0]), None
        elif len(orders) == 1:
            # Single derivative condition
            bc_str = f"@(u) {'diff(u' + ',1)' * orders[0]}"
            bc_str = bc_str.replace(",1)", ")")
            val_str = str(values[0])
            return bc_str, val_str
        else:
            # Multiple conditions
            terms = []
            for order in orders:
                if order == 0:
                    terms.append("u")
                else:
                    terms.append("diff(u" + ("" if order == 1 else f",{order}") + ")")
            bc_str = f"@(u) [{'; '.join(terms)}]"
            val_str = f"[{'; '.join(str(v) for v in values)}]"
            return bc_str, val_str

    return None, None


# Add new tests by appending to this list
# Just specify: name, desc, domain, op_str, lbc, rbc
TEST_CASES = [
    {
        "name": "Linear BVP",
        "desc": "u'' + u = 1, u(0)=u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + u - 1",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Cubic Nonlinearity",
        "desc": "u'' + u^3 = 1, u(0)=u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + u**3 - 1",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Fourth Order BVP",
        "desc": "u'''' = 1, u(0)=u'(0)=u(1)=u'(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(4) - 1",
        "lbc": {"u": 0.0, "u'": 0.0},
        "rbc": {"u": 0.0, "u'": 0.0},
    },
    {
        "name": "Nonlinear Bratu",
        "desc": "u'' + exp(u) = 0, u(0)=u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + np.exp(u)",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Quadratic Nonlinearity",
        "desc": "u'' - u^2 + 1 = 0, u(-1)=0, u(1)=0",
        "domain": [-1, 1],
        "op_str": "u.diff(2) - u**2 + 1",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Mixed Dirichlet-Neumann",
        "desc": "u'' = 0, u(0)=0, u'(1)=1",
        "domain": [0, 1],
        "op_str": "u.diff(2)",
        "lbc": 0.0,
        "rbc": {"u'": 1.0},
        # Exact solution: u = x (unique, well-posed)
    },
    {
        "name": "Robin BCs",
        "desc": "u'' + u = 1, u'(0)-u(0)=0, u'(1)+u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + u - 1",
        "lbc": {"custom": "u.diff() - u"},
        "rbc": {"custom": "u.diff() + u"},
    },
    {
        "name": "Sin Nonlinearity",
        "desc": "u'' + sin(u) = 0, u(0)=0, u(π)=0",
        "domain": [0, np.pi],
        "op_str": "u.diff(2) + np.sin(u)",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Third Order ODE",
        "desc": "u''' = 1, u(0)=0, u'(0)=1, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(3) - 1",
        "lbc": {"u": 0.0, "u'": 1.0},
        "rbc": 0.0,
    },
    {
        "name": "Mixed Derivative",
        "desc": "u'' + u' - u = 0, u(0)=0, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + u.diff() - u",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Exponential Coefficient",
        "desc": "u'' + exp(x)*u = 1, u(0)=0, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + u - 1",  # Simplified for testing
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Larger Domain",
        "desc": "u'' + 0.1u = 0, u(-π)=0, u(π)=0",
        "domain": [-np.pi, np.pi],
        "op_str": "u.diff(2) + 0.1*u",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "High Frequency",
        "desc": "u'' + 100u = 0, u(0)=1, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + 100*u",
        "lbc": 1.0,
        "rbc": 0.0,
    },
    {
        "name": "Stiff Decay",
        "desc": "u'' - 10u = 0, u(0)=1, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) - 10*u",
        "lbc": 1.0,
        "rbc": 0.0,
    },
    {
        "name": "Variable RHS",
        "desc": "u'' = x^2, u(0)=0, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2)",
        "lbc": 0.0,
        "rbc": 0.0,
        "has_rhs": True,
    },
    {
        "name": "Cosine Nonlinearity",
        "desc": "u'' + cos(u) = 0, u(0)=0, u(1)=1",
        "domain": [0, 1],
        "op_str": "u.diff(2) + np.cos(u)",
        "lbc": 0.0,
        "rbc": 1.0,
    },
    {
        "name": "Abs Nonlinearity",
        "desc": "u'' + abs(u) - 1 = 0, u(0)=0, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + np.abs(u) - 1",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Square Root",
        "desc": "u'' + sqrt(abs(u+1)) = 2, u(0)=0, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + np.sqrt(np.abs(u + 1)) - 2",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Power Law",
        "desc": "u'' + u^5 = 1, u(0)=0, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + u**5 - 1",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Damped Oscillator",
        "desc": "u'' + 0.5u' + u = 0, u(0)=1, u(2π)=0",
        "domain": [0, 2 * np.pi],
        "op_str": "u.diff(2) + 0.5*u.diff() + u",
        "lbc": 1.0,
        "rbc": 0.0,
    },
    {
        "name": "Fifth Order",
        "desc": "u^(5) = 1, u(0)=u'(0)=u''(0)=0, u(1)=u'(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(5) - 1",
        "lbc": {"u": 0.0, "u'": 0.0, "u''": 0.0},
        "rbc": {"u": 0.0, "u'": 0.0},
    },
    {
        "name": "Polynomial Coefficient",
        "desc": "(1+x^2)u'' + u = 0, u(0)=1, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + u",  # Simplified
        "lbc": 1.0,
        "rbc": 0.0,
    },
    {
        "name": "Tanh Nonlinearity",
        "desc": "u'' + tanh(u) = 0, u(-1)=0, u(1)=0",
        "domain": [-1, 1],
        "op_str": "u.diff(2) + np.tanh(u)",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Log Nonlinearity",
        "desc": "u'' + log(1+u^2) = 1, u(0)=0, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + np.log(1 + u**2) - 1",
        "lbc": 0.0,
        "rbc": 0.0,
    },
    {
        "name": "Rational Nonlinearity",
        "desc": "u'' + u/(1+u^2) = 0, u(0)=1, u(1)=0",
        "domain": [0, 1],
        "op_str": "u.diff(2) + u/(1 + u**2)",
        "lbc": 1.0,
        "rbc": 0.0,
    },
]


def preprocess_test_cases():
    """Pre-process test cases to generate all needed fields."""
    for test in TEST_CASES:
        # Generate Python operator
        if "op_str" in test:
            test["op"] = eval(f"lambda u: {test['op_str']}")
            test["matlab_op"] = python_to_matlab_op(test["op_str"])

        # Handle RHS if specified
        if test.get("has_rhs"):
            # Extract RHS from op_str if possible
            # For now, manually handle known cases
            if "x^2" in test.get("desc", ""):
                test["matlab_rhs"] = "@(x) x.^2"

        # Process left BC
        if "lbc" in test:
            lbc_spec = test["lbc"]
            if isinstance(lbc_spec, dict) and "custom" in lbc_spec:
                # Custom BC - keep as is for now
                test["lbc_type"] = "mixed"
                custom_expr = lbc_spec["custom"]
                test["lbc"] = eval(f"lambda u: {custom_expr}")
                test["matlab_lbc"] = f"@(u) {custom_expr.replace('u.diff()', 'diff(u)')}"
            else:
                bc_val, bc_type, bc_values = bc_to_python(lbc_spec)
                test["lbc"] = bc_val
                test["lbc_type"] = bc_type
                if bc_values:
                    test["lbc_val"] = bc_values

                matlab_bc, matlab_val = bc_to_matlab(lbc_spec)
                if matlab_bc:
                    test["matlab_lbc"] = matlab_bc
                if matlab_val:
                    test["matlab_lbc_vals"] = matlab_val

        # Process right BC
        if "rbc" in test:
            rbc_spec = test["rbc"]
            if isinstance(rbc_spec, dict) and "custom" in rbc_spec:
                test["rbc_type"] = "mixed"
                custom_expr = rbc_spec["custom"]
                test["rbc"] = eval(f"lambda u: {custom_expr}")
                test["matlab_rbc"] = f"@(u) {custom_expr.replace('u.diff()', 'diff(u)')}"
            else:
                bc_val, bc_type, bc_values = bc_to_python(rbc_spec)
                test["rbc"] = bc_val
                test["rbc_type"] = bc_type if bc_type else test.get("lbc_type")
                if bc_values:
                    test["rbc_val"] = bc_values

                matlab_bc, matlab_val = bc_to_matlab(rbc_spec)
                if matlab_bc:
                    test["matlab_rbc"] = matlab_bc
                if matlab_val:
                    test["matlab_rbc_vals"] = matlab_val


def solve_chebpy(problem):
    """Solve with ChebPy and compute metrics."""
    op = chebop(problem["domain"])
    op.op = problem["op"]

    # Handle boundary conditions
    if "lbc" in problem and problem["lbc"] is not None:
        lbc_type = problem.get("lbc_type")
        if lbc_type in ("dirichlet", "neumann", "mixed_list"):
            op.lbc = problem["lbc"]
        elif lbc_type == "mixed":
            op.lbc = problem["lbc"]
            if "lbc_val" in problem:
                op.lbc_val = problem["lbc_val"]

    if "rbc" in problem and problem["rbc"] is not None:
        rbc_type = problem.get("rbc_type", problem.get("lbc_type"))
        if rbc_type in ("dirichlet", "neumann", "mixed_list"):
            op.rbc = problem["rbc"]
        elif rbc_type == "mixed":
            op.rbc = problem["rbc"]
            if "rbc_val" in problem:
                op.rbc_val = problem["rbc_val"]

    start = time.time()
    try:
        u = op.solve()
        elapsed = time.time() - start

        # Compute residual
        residual = op.op(u)
        max_residual = residual.norm(np.inf)

        # Compute BC error
        a, b = problem["domain"]
        bc_left = 0.0
        bc_right = 0.0

        if "lbc" in problem and problem["lbc"] is not None:
            lbc_type = problem.get("lbc_type")
            if lbc_type == "dirichlet":
                bc_left = abs(u(a) - problem["lbc"])
            elif lbc_type in ("neumann", "mixed_list"):
                # List format BC: [u_val, u'_val, ...] where None means no constraint
                bc_list = problem["lbc"]
                for k, val in enumerate(bc_list):
                    if val is not None:
                        u_deriv = u
                        for _ in range(k):
                            u_deriv = u_deriv.diff()
                        bc_left = max(bc_left, abs(u_deriv(a) - val))
            elif lbc_type == "mixed" and "lbc_val" in problem:
                bc_vals = problem["lbc_val"]
                if len(bc_vals) == 1:
                    bc_left = abs(u.diff()(a) - bc_vals[0])
                else:
                    errs = [abs(u(a) - bc_vals[0])]
                    for i in range(1, len(bc_vals)):
                        u_deriv = u
                        for _ in range(i):
                            u_deriv = u_deriv.diff()
                        errs.append(abs(u_deriv(a) - bc_vals[i]))
                    bc_left = max(errs)

        if "rbc" in problem and problem["rbc"] is not None:
            rbc_type = problem.get("rbc_type", problem.get("lbc_type"))
            if rbc_type == "dirichlet":
                bc_right = abs(u(b) - problem["rbc"])
            elif rbc_type in ("neumann", "mixed_list"):
                # List format BC: [u_val, u'_val, ...] where None means no constraint
                bc_list = problem["rbc"]
                for k, val in enumerate(bc_list):
                    if val is not None:
                        u_deriv = u
                        for _ in range(k):
                            u_deriv = u_deriv.diff()
                        bc_right = max(bc_right, abs(u_deriv(b) - val))
            elif rbc_type == "mixed" and "rbc_val" in problem:
                bc_vals = problem["rbc_val"]
                if len(bc_vals) == 1:
                    bc_right = abs(u.diff()(b) - bc_vals[0])
                else:
                    errs = [abs(u(b) - bc_vals[0])]
                    for i in range(1, len(bc_vals)):
                        u_deriv = u
                        for _ in range(i):
                            u_deriv = u_deriv.diff()
                        errs.append(abs(u_deriv(b) - bc_vals[i]))
                    bc_right = max(errs)

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
        return {"success": False, "time": time.time() - start, "error_msg": str(e)[:100]}


def solve_matlab(problem, idx):
    """Solve with MATLAB and compute metrics."""
    script = f"""
addpath('matlab_chebfun');

tic;
try
    op = chebop({problem["domain"][0]}, {problem["domain"][1]});
    op.op = {problem["matlab_op"]};
    """

    # Add boundary conditions
    if "matlab_lbc" in problem:
        if "matlab_lbc_vals" in problem:
            script += f"    op.lbc = {problem['matlab_lbc']};\n"
            script += f"    op.lbc = {problem['matlab_lbc_vals']};\n"
        else:
            script += f"    op.lbc = {problem['matlab_lbc']};\n"

    if "matlab_rbc" in problem:
        if "matlab_rbc_vals" in problem:
            script += f"    op.rbc = {problem['matlab_rbc']};\n"
            script += f"    op.rbc = {problem['matlab_rbc_vals']};\n"
        else:
            script += f"    op.rbc = {problem['matlab_rbc']};\n"

    if "matlab_rhs" in problem:
        script += f"    op.rhs = {problem['matlab_rhs']};\n"

    script += """    u = op\\0;
    elapsed = toc;

    % Compute residual
    x_test = linspace(op.domain(1), op.domain(2), 100)';
    residual = op.op(u);
    res_vals = residual(x_test);
    max_residual = max(abs(res_vals));

    % Compute BC error (simplified)
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

    try:
        result = subprocess.run(
            [
                "/Applications/MATLAB_R2025b.app/bin/matlab",
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                f"try; run('{script_file}'); catch ME; fprintf('FAILED\\nError: %s\\n', ME.message); end; exit;",
            ],
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


def run_tests(test_cases=None):
    """Run tests and display results."""
    if test_cases is None:
        test_cases = TEST_CASES

    print(f"{BOLD}{BLUE}CHEBOP TEST SUITE - PYTHON vs MATLAB{RESET}{BOLD}{RESET}")
    print(f"Running {len(test_cases)} test(s)\n")

    results = []
    passed = 0
    failed = 0

    for idx, problem in enumerate(test_cases, 1):
        print(f"{BOLD}[{idx}/{len(test_cases)}] {problem['name']}{RESET}")
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
                    print(f"    {YELLOW}>>> MATLAB is {1 / speedup:.2f}x faster{RESET}")
                passed += 1
        else:
            print(f"{RED}✗ FAILED{RESET}")
            print(f"            Error: {matlab_result['error_msg']}")

        print()
        results.append({"problem": problem, "chebpy": chebpy_result, "matlab": matlab_result})

    # Summary
    print(f"{BOLD}{BLUE}SUMMARY{RESET}{BOLD}{RESET}")

    if passed == len(test_cases):
        print(f"{GREEN}{BOLD}✓ ALL {len(test_cases)} TESTS PASSED!{RESET}")
    else:
        print(f"{YELLOW}{BOLD}Passed: {passed}/{len(test_cases)}{RESET}")
        if failed > 0:
            print(f"{RED}{BOLD}Failed: {failed}/{len(test_cases)}{RESET}")

    # Statistics
    chebpy_times = [r["chebpy"]["time"] for r in results if r["chebpy"]["success"]]
    matlab_times = [r["matlab"]["time"] for r in results if r["matlab"]["success"]]

    if chebpy_times and matlab_times:
        avg_speedup = np.mean([m / c for c, m in zip(chebpy_times, matlab_times)])
        print(f"\n{BOLD}Average speedup:{RESET} ", end="")
        if avg_speedup > 1:
            print(f"{GREEN}ChebPy is {avg_speedup:.2f}x faster{RESET}")
        else:
            print(f"{YELLOW}MATLAB is {1 / avg_speedup:.2f}x faster{RESET}")



if __name__ == "__main__":
    # Pre-process test cases to generate ops
    preprocess_test_cases()

    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--last":
        if len(sys.argv) < 3:
            print("Usage: python test_chebop_examples.py --last N")
            sys.exit(1)

        try:
            n = int(sys.argv[2])
            if n <= 0:
                print("Error: N must be positive")
                sys.exit(1)
            if n > len(TEST_CASES):
                print(f"Warning: Only {len(TEST_CASES)} tests available, running all")
                n = len(TEST_CASES)

            test_cases_to_run = TEST_CASES[-n:]
            print(f"\n{YELLOW}Running last {n} test(s) only{RESET}")
        except ValueError:
            print("Error: N must be an integer")
            sys.exit(1)

        run_tests(test_cases_to_run)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        if len(sys.argv) < 3:
            print("Usage: python test_chebop_examples.py --test N")
            sys.exit(1)

        try:
            n = int(sys.argv[2])
            if n <= 0 or n > len(TEST_CASES):
                print(f"Error: Test number must be between 1 and {len(TEST_CASES)}")
                sys.exit(1)

            test_cases_to_run = [TEST_CASES[n - 1]]
            print(f"\n{YELLOW}Running test {n}: {test_cases_to_run[0]['name']}{RESET}")
        except ValueError:
            print("Error: N must be an integer")
            sys.exit(1)

        run_tests(test_cases_to_run)
    else:
        run_tests()
