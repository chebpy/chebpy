"""Run pip-audit with a tiered vulnerability policy.

Fails the build for vulnerabilities in runtime dependencies.
Warns (without failing) for tooling packages: pip, setuptools, wheel, distribute.
Any extra arguments are forwarded to pip-audit (e.g. ``--ignore-vuln CVE-XXXX-YYYY``).
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
import sys

_RESET = "\033[0m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"

# Packages treated as build tooling — CVEs warn but do not fail CI.
_TOOLING: frozenset[str] = frozenset({"pip", "setuptools", "wheel", "distribute"})


def _vuln_ids(vuln: dict) -> str:  # type: ignore[type-arg]
    """Return a human-readable string of all IDs for a vulnerability entry."""
    ids = [vuln["id"]] + [a for a in vuln.get("aliases", []) if a != vuln["id"]]
    return ", ".join(ids)


def main() -> int:
    """Run pip-audit and apply tiered vulnerability policy."""
    uvx = shutil.which("uvx") or "uvx"
    cmd = [uvx, "pip-audit", "--format", "json", *sys.argv[1:]]
    proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603  # nosec B603

    if proc.returncode == 0:
        print(f"{_GREEN}[OK] pip-audit: no vulnerabilities found{_RESET}")
        return 0

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    deps = data.get("dependencies", [])
    tooling_vulns = [d for d in deps if d.get("vulns") and d["name"].lower() in _TOOLING]
    runtime_vulns = [d for d in deps if d.get("vulns") and d["name"].lower() not in _TOOLING]

    for dep in tooling_vulns:
        for v in dep["vulns"]:
            print(
                f"{_YELLOW}[WARN] {dep['name']}=={dep['version']}: {_vuln_ids(v)} (tooling — not failing build){_RESET}"
            )

    if not runtime_vulns:
        return 0

    for dep in runtime_vulns:
        for v in dep["vulns"]:
            print(f"{_RED}[FAIL] {dep['name']}=={dep['version']}: {_vuln_ids(v)}{_RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
