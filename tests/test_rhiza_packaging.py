"""The first test a freshly synced Python project has.

This file flows down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

**Why it exists.** Two reasons, and the second is the one that is easy to lose.

It checks a real invariant: that the version the project *declares* in
``pyproject.toml`` is the version actually installed into the environment. Those drift
apart more often than anything else on a fresh checkout — an editable install left over
from a rename, a `uv sync` that never ran, a package directory the build backend is not
configured to pick up. Each shows up here as a mismatch rather than as a confusing
ImportError three files later.

And it is the only test a freshly synced project has. ``make test`` searches
``TESTS_FOLDER`` for ``test_*.py``/``*_test.py``, and finding none it prints a warning
and **exits 0** — so a new repo passed ``make test``, and therefore ``make all``, while
measuring nothing (#1476). That was the third instance of one pattern: Go had it until
``go-core`` shipped ``internal/version/version_test.go`` (#1467), and ``rhiza-test`` had
it until the ``.rhiza/tests`` suite was actually delivered to every layer (#1469). Rust
never did, because ``cargo init --lib`` leaves an ``it_works`` test behind.

Writing your own tests alongside this is the point. Deleting it and shipping nothing
else puts the vacuum back.

**Why not simply fail when no tests exist?** Because that turns ``make all`` red on the
output of ``/rhiza:init``, before the author has written a line. The warning branch is
deliberate; what it needed was something to find.

Deliberately self-contained: it uses no fixtures, because this lives in *your* ``tests/``
directory and must not depend on the ``conftest.py`` that ships with ``.rhiza/tests``.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest

# tests/ sits at the project root, so the parent of this file's directory is the root.
#
# `.absolute()`, never `.resolve()`. In a synced project the difference is nil, but in
# rhiza's own repository this file is a *symlink* into `bundles/python-core/tests/`, and
# `.resolve()` follows it — making the "project root" come out as `bundles/python-core`,
# which has no pyproject.toml. The suite then skips for a plausible-looking wrong reason
# instead of running. `.rhiza/tests/conftest.py` avoids the same trap the same way.
_ROOT = Path(__file__).absolute().parent.parent


def _project_table() -> dict:
    """Return the ``[project]`` table from pyproject.toml, skipping when unusable.

    Returns:
        The parsed ``[project]`` table.
    """
    pyproject = _ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("no pyproject.toml at the project root")
    with pyproject.open("rb") as handle:
        table = tomllib.load(handle).get("project")
    if not isinstance(table, dict):
        pytest.skip("pyproject.toml declares no [project] table")
    return table


def test_the_installed_version_matches_pyproject() -> None:
    """The distribution installed in the environment must match the declared version.

    Skips rather than fails when the project is not installed as a distribution at all —
    a ``uv`` *virtual* project (no ``[build-system]``, ``source = {{ virtual = "." }}`` in
    the lockfile) has no metadata to read, and rhiza's own repository is one. That is a
    real configuration, not a broken one, so it is not this test's business.
    """
    project = _project_table()

    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        pytest.skip("[project].name is missing or not a plain string")

    declared = project.get("version")
    if not isinstance(declared, str):
        pytest.skip("[project].version is dynamic — there is no static value to compare")

    try:
        found = installed_version(name)
    except PackageNotFoundError:
        pytest.skip(f"{name!r} is not installed as a distribution (a virtual project has no metadata)")

    assert found == declared, (
        f"pyproject.toml declares version {declared!r} but the installed {name!r} reports "
        f"{found!r}. The environment is stale or the build backend is picking up a different "
        f"tree — re-run `make install`, and check [build-system] and the packages it is told "
        f"to include if that does not fix it."
    )
