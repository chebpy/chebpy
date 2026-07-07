"""Tests for pyproject.toml structure and required fields.

This file and its associated tests flow down via a SYNC action from the
jebel-quant/rhiza repository (https://github.com/jebel-quant/rhiza).

Validates that pyproject.toml:
- is syntactically valid TOML
- contains all required [project] fields
- declares a semver-compatible version
- specifies a minimum Python version via requires-python
- lists at least one named author
- provides [project.urls] with Homepage and Repository
- includes at least one Python version classifier
- declares a [dependency-groups] test group containing pytest
- declares a [dependency-groups] lint group
- version matches the latest git tag (vX.Y.Z → X.Y.Z)
"""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

_GIT = shutil.which("git") or "/usr/bin/git"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+")
_REQUIRED_PROJECT_FIELDS = ("name", "version", "description", "readme", "requires-python", "license", "authors")


@pytest.fixture(scope="module")
def pyproject(root: Path) -> dict:
    """Load and return pyproject.toml as a parsed dict."""
    path = root / "pyproject.toml"
    if not path.exists():
        pytest.skip("pyproject.toml not found")
    with path.open("rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def project(pyproject: dict) -> dict:
    """Return the [project] table from pyproject.toml."""
    table = pyproject.get("project")
    if not isinstance(table, dict):
        pytest.fail("pyproject.toml is missing a [project] table")
    return table


class TestPyprojectToml:
    """Tests for basic pyproject.toml existence and validity."""

    def test_pyproject_toml_exists(self, root: Path) -> None:
        """pyproject.toml must exist at the project root."""
        assert (root / "pyproject.toml").is_file(), "pyproject.toml not found at project root"

    def test_pyproject_toml_is_valid_toml(self, root: Path) -> None:
        """pyproject.toml must be syntactically valid TOML."""
        with (root / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        assert isinstance(data, dict), "Parsed pyproject.toml must be a TOML table"

    def test_project_table_present(self, pyproject: dict) -> None:
        """pyproject.toml must contain a [project] table."""
        assert "project" in pyproject, "pyproject.toml is missing a [project] table"
        assert isinstance(pyproject["project"], dict), "[project] must be a TOML table"


class TestProjectFields:
    """Tests for required fields within the [project] table."""

    @pytest.mark.parametrize("field", _REQUIRED_PROJECT_FIELDS)
    def test_required_field_present(self, project: dict, field: str) -> None:
        """Each required [project] field must be present."""
        assert field in project, f"[project] is missing required field '{field}'"

    def test_name_is_non_empty_string(self, project: dict) -> None:
        """[project].name must be a non-empty string."""
        name = project.get("name", "")
        assert isinstance(name, str), "[project].name must be a string"
        assert name.strip(), "[project].name must be a non-empty string"

    def test_version_follows_semver(self, project: dict) -> None:
        """[project].version must follow semver (MAJOR.MINOR.PATCH)."""
        version = project.get("version", "")
        assert _SEMVER_RE.match(str(version)), (
            f"[project].version {version!r} does not follow semver (expected MAJOR.MINOR.PATCH)"
        )

    def test_requires_python_is_set(self, project: dict) -> None:
        """[project].requires-python must be set to a non-empty constraint."""
        rp = project.get("requires-python", "")
        assert isinstance(rp, str), "[project].requires-python must be a string"
        assert rp.strip(), "[project].requires-python must be a non-empty version constraint"

    def test_authors_have_names(self, project: dict) -> None:
        """[project].authors must contain at least one entry with a non-empty 'name'."""
        authors = project.get("authors", [])
        assert isinstance(authors, list), "[project].authors must be a list"
        assert len(authors) >= 1, "[project].authors must list at least one author"
        named = [a for a in authors if isinstance(a, dict) and a.get("name", "").strip()]
        assert len(named) >= 1, "At least one entry in [project].authors must have a non-empty 'name'"

    def test_description_is_non_empty_string(self, project: dict) -> None:
        """[project].description must be a non-empty string."""
        desc = project.get("description", "")
        assert isinstance(desc, str), "[project].description must be a string"
        assert desc.strip(), "[project].description must be a non-empty string"


class TestProjectUrls:
    """Tests for [project.urls] — Homepage and Repository links."""

    @pytest.fixture
    def urls(self, project: dict) -> dict:
        """Return the [project.urls] table."""
        table = project.get("urls")
        if not isinstance(table, dict):
            pytest.skip("[project.urls] not present")
        return table

    def test_urls_table_present(self, project: dict) -> None:
        """[project.urls] must be present."""
        assert "urls" in project, "pyproject.toml is missing a [project.urls] table"

    def test_homepage_configured(self, urls: dict) -> None:
        """[project.urls] must include a Homepage entry."""
        assert "Homepage" in urls, "[project.urls] is missing a 'Homepage' entry"
        assert urls["Homepage"].strip(), "[project.urls] 'Homepage' must be non-empty"

    def test_repository_configured(self, urls: dict) -> None:
        """[project.urls] must include a Repository entry."""
        assert "Repository" in urls, "[project.urls] is missing a 'Repository' entry"
        assert urls["Repository"].strip(), "[project.urls] 'Repository' must be non-empty"


class TestProjectClassifiers:
    """Tests for [project].classifiers — Python version and licence entries."""

    @pytest.fixture
    def classifiers(self, project: dict) -> list[str]:
        """Return the classifiers list."""
        cl = project.get("classifiers", [])
        if not cl:
            pytest.skip("No classifiers declared in [project]")
        return cl

    def test_python_version_classifier_present(self, classifiers: list[str]) -> None:
        """At least one 'Programming Language :: Python :: 3.X' classifier must be present."""
        python_classifiers = [c for c in classifiers if re.match(r"Programming Language :: Python :: 3\.\d+", c)]
        assert len(python_classifiers) >= 1, (
            "classifiers must include at least one 'Programming Language :: Python :: 3.X' entry"
        )

    def test_license_classifier_present(self, classifiers: list[str]) -> None:
        """At least one 'License :: ' classifier must be present."""
        license_classifiers = [c for c in classifiers if c.startswith("License ::")]
        assert len(license_classifiers) >= 1, "classifiers must include at least one 'License :: ' entry"


class TestDependencyGroups:
    """Tests for [dependency-groups] — ensures required groups are declared."""

    @pytest.fixture
    def dependency_groups(self, pyproject: dict) -> dict:
        """Return the [dependency-groups] table."""
        dg = pyproject.get("dependency-groups")
        if not isinstance(dg, dict):
            pytest.skip("[dependency-groups] not present")
        return dg

    def test_test_group_present(self, dependency_groups: dict) -> None:
        """A 'test' dependency group must be declared."""
        assert "test" in dependency_groups, "[dependency-groups] must include a 'test' group"

    def test_test_group_includes_pytest(self, dependency_groups: dict) -> None:
        """The 'test' dependency group must include pytest."""
        test_deps = dependency_groups.get("test", [])
        assert any("pytest" in str(dep).lower() for dep in test_deps), (
            "[dependency-groups.test] must list pytest as a dependency"
        )

    def test_lint_group_present(self, dependency_groups: dict) -> None:
        """A 'lint' dependency group must be declared."""
        assert "lint" in dependency_groups, "[dependency-groups] must include a 'lint' group"


class TestGitTagVersion:
    """Tests for harmony between the latest git tag and pyproject.toml version."""

    @pytest.fixture
    def latest_tag(self, root: Path) -> str:
        """Return the latest semver git tag, or skip if none exist."""
        result = subprocess.run(  # nosec B603
            [_GIT, "tag", "--list", "v*", "--sort=-version:refname"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not tags:
            pytest.skip("No version tags found in repository")
        return tags[0]

    def test_latest_tag_matches_pyproject_version(self, latest_tag: str, project: dict) -> None:
        """The latest git tag (vX.Y.Z) must match [project].version in pyproject.toml."""
        tag_version = str(Version(latest_tag.lstrip("v")))
        pyproject_version = str(Version(project.get("version", "")))
        assert tag_version == pyproject_version, (
            f"Latest git tag {latest_tag!r} (→ {tag_version!r}) does not match "
            f"[project].version {pyproject_version!r} in pyproject.toml"
        )
