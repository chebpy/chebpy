"""Tests for module docstrings using doctest.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Automatically discovers all packages under `src/` and runs doctests for each.
"""

from __future__ import annotations

import doctest
import importlib
import warnings
from pathlib import Path

import pytest


def _iter_modules_from_path(logger, package_path: Path):
    """Recursively find all Python modules in a directory."""
    for path in package_path.rglob("*.py"):
        if path.name == "__init__.py":
            module_path = path.parent.relative_to(package_path.parent)
        else:
            module_path = path.relative_to(package_path.parent).with_suffix("")

        # Convert path to module name in an OS-independent way
        module_name = ".".join(module_path.parts)

        try:
            yield importlib.import_module(module_name)
        except ImportError as e:
            warnings.warn(f"Could not import {module_name}: {e}", stacklevel=2)
            logger.warning("Could not import module %s: %s", module_name, e)
            continue


def test_doctests(logger, root, monkeypatch: pytest.MonkeyPatch):
    """Run doctests for each package directory under src/."""
    src_path = root / "src"

    logger.info("Starting doctest discovery in: %s", src_path)
    if not src_path.exists():
        logger.info("Source directory not found: %s — skipping doctests", src_path)
        pytest.skip(f"Source directory not found: {src_path}")

    # Add src to sys.path with automatic cleanup
    monkeypatch.syspath_prepend(str(src_path))
    logger.debug("Prepended to sys.path: %s", src_path)

    total_tests = 0
    total_failures = 0
    failed_modules = []

    # Find all packages in src
    for package_dir in src_path.iterdir():
        if package_dir.is_dir() and (package_dir / "__init__.py").exists():
            # Import the package
            package_name = package_dir.name
            logger.info("Discovered package: %s", package_name)
            try:
                modules = list(_iter_modules_from_path(logger, package_dir))
                logger.debug("%d module(s) found in package %s", len(modules), package_name)

                for module in modules:
                    logger.debug("Running doctests for module: %s", module.__name__)
                    results = doctest.testmod(
                        module,
                        verbose=False,
                        optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE),
                    )
                    total_tests += results.attempted

                    if results.failed:
                        logger.warning(
                            "Doctests failed for %s: %d/%d failed",
                            module.__name__,
                            results.failed,
                            results.attempted,
                        )
                        total_failures += results.failed
                        failed_modules.append((module.__name__, results.failed, results.attempted))
                    else:
                        logger.debug("Doctests passed for %s (%d test(s))", module.__name__, results.attempted)

            except ImportError as e:
                warnings.warn(f"Could not import package {package_name}: {e}", stacklevel=2)
                logger.warning("Could not import package %s: %s", package_name, e)
                continue

    if failed_modules:
        formatted = "\n".join(f"  {name}: {failed}/{attempted} failed" for name, failed, attempted in failed_modules)
        msg = (
            f"Doctest summary: {total_tests} tests across {len(failed_modules)} module(s)\n"
            f"Failures: {total_failures}\n"
            f"Failed modules:\n{formatted}"
        )
        logger.error("%s", msg)
        assert total_failures == 0, msg
    else:
        logger.info("Doctest summary: %d tests, 0 failures", total_tests)

    if total_tests == 0:
        logger.info("No doctests were found in any module — skipping")
        pytest.skip("No doctests were found in any module")
