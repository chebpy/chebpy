"""Tests to validate the bundle-centric directory layout and template-bundles.yml.

File ownership is expressed through the filesystem: each bundle has a directory
under bundles/<name>/ containing symlinks (for regular files that live at their
deployment paths) or real files (for overlay injection content that was previously
in .rhiza/stubs/).

This module ensures:
  - every bundle in template-bundles.yml has a directory under bundles/
  - every bundle directory is non-empty
  - no symlinks inside bundle dirs are broken
  - no deployment path is claimed by more than one bundle (ownership conflict)
  - github-* bundles only deliver .github/ paths; gitlab-* only .gitlab/ paths
  - the local profile resolves to zero .github/workflows/* files
  - bundle dependency references (requires/recommends) point to existing bundles
  - no circular dependencies exist in bundle requires chains
  - profile definitions reference existing bundles
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


def _find_cycle(start: str, bundles: dict, visited: set, path: list) -> list | None:
    """DFS helper returning the cycle path if one exists, else None."""
    visited.add(start)
    path.append(start)
    for dep in bundles.get(start, {}).get("requires", []):
        if dep not in bundles:
            continue
        if dep in path:
            return [*path[path.index(dep) :], dep]
        if dep not in visited:
            result = _find_cycle(dep, bundles, visited, path)
            if result:
                return result
    path.pop()
    return None


def _deployment_paths(bundle_dir: Path) -> list[str]:
    """Return all deployment paths for files/symlinks in a bundle directory.

    Walks the bundle dir shallowly for symlinks (each symlink is one entry)
    and recursively for real sub-directories and files.
    """
    paths = []
    for root, dirs, files in os.walk(bundle_dir, followlinks=False):
        root_path = Path(root)
        # Yield symlinks (both file- and dir-type)
        for name in dirs[:]:
            child = root_path / name
            if child.is_symlink():
                dirs.remove(name)  # don't recurse into symlinked dirs
                paths.append(str(child.relative_to(bundle_dir)))
        for name in files:
            child = root_path / name
            paths.append(str(child.relative_to(bundle_dir)))
    return paths


class TestTemplateBundles:
    """Tests for template-bundles.yml and the bundles/ directory layout."""

    @pytest.fixture
    def bundles_file(self, root):
        """Return the path to template-bundles.yml."""
        return root / ".rhiza" / "template-bundles.yml"

    @pytest.fixture
    def bundles_data(self, bundles_file):
        """Load and parse the template-bundles.yml file."""
        if not bundles_file.exists():
            pytest.skip("template-bundles.yml does not exist in this project")

        with open(bundles_file) as f:
            data = yaml.safe_load(f)

        if not data or "bundles" not in data:
            pytest.fail("Invalid template-bundles.yml format - missing 'bundles' key")

        return data

    @pytest.fixture
    def bundle_names(self, bundles_data):
        """Return the set of all defined bundle names."""
        return set(bundles_data["bundles"].keys())

    @pytest.fixture
    def profiles_data(self, bundles_data):
        """Return the profiles section, or an empty dict if absent."""
        return bundles_data.get("profiles", {})

    @pytest.fixture
    def bundles_root(self, root):
        """Return the path to the bundles/ directory."""
        return root / "bundles"

    # ------------------------------------------------------------------
    # Bundle directory structure
    # ------------------------------------------------------------------

    def test_bundles_directory_exists(self, bundles_data, bundles_root):
        """Test that the bundles/ directory exists."""
        assert bundles_root.is_dir(), "bundles/ directory must exist at the repo root"

    def test_all_bundles_have_directories(self, bundles_data, bundles_root):
        """Test that every bundle in template-bundles.yml has a directory under bundles/."""
        missing = [name for name in bundles_data["bundles"] if not (bundles_root / name).is_dir()]
        if missing:
            pytest.fail(f"\nBundles missing a directory under bundles/: {missing}")

    def test_all_bundle_dirs_are_non_empty(self, bundles_data, bundles_root):
        """Test that no bundle directory is empty."""
        empty = []
        for name in bundles_data["bundles"]:
            bundle_dir = bundles_root / name
            if bundle_dir.is_dir() and not any(bundle_dir.iterdir()):
                empty.append(name)
        if empty:
            pytest.fail(f"\nBundle directories with no files: {empty}")

    def test_no_broken_symlinks_in_bundle_dirs(self, bundles_data, bundles_root):
        """Test that every symlink inside a bundle directory resolves to an existing path."""
        errors = []
        for name in bundles_data["bundles"]:
            bundle_dir = bundles_root / name
            if not bundle_dir.is_dir():
                continue
            for root_dir, dirs, files in os.walk(bundle_dir, followlinks=False):
                for entry in list(dirs) + files:
                    path = Path(root_dir) / entry
                    if path.is_symlink() and not path.exists():
                        errors.append(
                            f"  [{name}] broken symlink: {path.relative_to(bundles_root)} → {os.readlink(path)}"
                        )
        if errors:
            pytest.fail("\nBroken symlinks found in bundle dirs:\n" + "\n".join(errors))

    def test_no_file_ownership_conflicts(self, bundles_data, bundles_root):
        """Test that no deployment path is claimed by more than one bundle."""
        dest_owners: dict[str, list[str]] = {}
        for name in bundles_data["bundles"]:
            bundle_dir = bundles_root / name
            if not bundle_dir.is_dir():
                continue
            for dep_path in _deployment_paths(bundle_dir):
                dest_owners.setdefault(dep_path, []).append(name)

        conflicts = {dest: owners for dest, owners in dest_owners.items() if len(owners) > 1}
        if conflicts:
            lines = [f"  {dest}: {owners}" for dest, owners in sorted(conflicts.items())]
            pytest.fail("\nFile ownership conflicts (same deployment path in multiple bundles):\n" + "\n".join(lines))

    @pytest.mark.parametrize(
        ("prefix", "namespace"),
        [
            ("github-", ".github/"),
            ("gitlab-", ".gitlab/"),
        ],
    )
    def test_overlay_bundle_files_stay_within_namespace(self, bundles_data, bundles_root, prefix, namespace):
        """Test that github-* bundles only deliver .github/ files, gitlab-* only .gitlab/."""
        violations = []
        for name in bundles_data["bundles"]:
            if not name.startswith(prefix):
                continue
            bundle_dir = bundles_root / name
            if not bundle_dir.is_dir():
                continue
            for dep_path in _deployment_paths(bundle_dir):
                if not dep_path.startswith(namespace):
                    violations.append(f"  [{name}] {dep_path!r} is outside '{namespace}'")

        if violations:
            pytest.fail(
                f"\n'{prefix}*' overlay bundles must only deliver files under '{namespace}':\n" + "\n".join(violations)
            )

    # ------------------------------------------------------------------
    # Local profile invariant
    # ------------------------------------------------------------------

    def test_local_profile_contains_no_github_workflows(self, bundles_data, profiles_data, bundles_root):
        """Test that the local profile resolves to no .github/workflows/* files."""
        if not profiles_data:
            pytest.skip("No profiles defined in template-bundles.yml")
        if "local" not in profiles_data:
            pytest.skip("No 'local' profile defined")

        local_bundles = profiles_data["local"].get("bundles", [])
        workflow_files = []

        for bundle_name in local_bundles:
            bundle_dir = bundles_root / bundle_name
            if not bundle_dir.is_dir():
                continue
            for dep_path in _deployment_paths(bundle_dir):
                if ".github/workflows/" in dep_path:
                    workflow_files.append(f"  [{bundle_name}] {dep_path}")

        if workflow_files:
            pytest.fail(
                "\nThe 'local' profile must not resolve to any .github/workflows/* files.\n"
                "The following workflow files were found:\n" + "\n".join(workflow_files)
            )

    # ------------------------------------------------------------------
    # Bundle metadata
    # ------------------------------------------------------------------

    def test_all_bundles_have_descriptions(self, bundles_data):
        """Test that every bundle has a non-empty description field."""
        missing = [name for name, cfg in bundles_data["bundles"].items() if not str(cfg.get("description", "")).strip()]
        if missing:
            pytest.fail(f"\nBundles missing description: {missing}")

    def test_bundle_requires_reference_existing_bundles(self, bundles_data, bundle_names):
        """Test that all requires and recommends entries point to existing bundles."""
        bundles = bundles_data["bundles"]
        errors = []

        for bundle_name, bundle_config in bundles.items():
            for field in ("requires", "recommends"):
                for dep in bundle_config.get(field, []):
                    if dep not in bundle_names:
                        errors.append(f"  [{bundle_name}] {field}: '{dep}' does not exist")

        if errors:
            pytest.fail("\nBundle dependency references missing:\n" + "\n".join(errors))

    def test_no_circular_bundle_dependencies(self, bundles_data, bundle_names):
        """Test that no circular dependencies exist in bundle requires chains."""
        bundles = bundles_data["bundles"]
        visited: set = set()
        cycles = []
        for name in bundle_names:
            if name not in visited:
                cycle = _find_cycle(name, bundles, visited, [])
                if cycle:
                    cycles.append(" -> ".join(cycle))
        if cycles:
            pytest.fail("\nCircular bundle dependencies detected:\n" + "\n".join(f"  {c}" for c in cycles))

    @pytest.mark.parametrize("bundle_name", ["github-marimo", "github-tests", "github-book"])
    def test_github_overlay_bundles_require_github(self, bundles_data, bundle_names, bundle_name):
        """Test that key github-* overlay bundles require the github bundle."""
        if bundle_name not in bundle_names:
            pytest.skip(f"Bundle '{bundle_name}' not defined in this project")

        requires = bundles_data["bundles"][bundle_name].get("requires", [])
        assert "github" in requires, f"Bundle '{bundle_name}' must list 'github' in its requires, got: {requires}"

    @pytest.mark.parametrize("bundle_name", ["gitlab-marimo", "gitlab-tests", "gitlab-book"])
    def test_gitlab_overlay_bundles_require_gitlab(self, bundles_data, bundle_names, bundle_name):
        """Test that key gitlab-* overlay bundles require the gitlab bundle."""
        if bundle_name not in bundle_names:
            pytest.skip(f"Bundle '{bundle_name}' not defined in this project")

        requires = bundles_data["bundles"][bundle_name].get("requires", [])
        assert "gitlab" in requires, f"Bundle '{bundle_name}' must list 'gitlab' in its requires, got: {requires}"

    # ------------------------------------------------------------------
    # Profile definitions
    # ------------------------------------------------------------------

    def test_profiles_reference_existing_bundles(self, profiles_data, bundle_names):
        """Test that all bundles listed in profiles exist."""
        if not profiles_data:
            pytest.skip("No profiles defined in template-bundles.yml")

        errors = []
        for profile_name, profile_config in profiles_data.items():
            for bundle_ref in profile_config.get("bundles", []):
                if bundle_ref not in bundle_names:
                    errors.append(f"  [profile:{profile_name}] bundle '{bundle_ref}' does not exist")

        if errors:
            pytest.fail("\nProfile bundle references missing:\n" + "\n".join(errors))

    def test_each_profile_has_description(self, profiles_data):
        """Test that every profile has a non-empty description."""
        if not profiles_data:
            pytest.skip("No profiles defined in template-bundles.yml")

        missing = [name for name, cfg in profiles_data.items() if not cfg.get("description", "").strip()]
        if missing:
            pytest.fail(f"\nProfiles missing description: {missing}")

    def test_each_profile_has_at_least_one_bundle(self, profiles_data):
        """Test that every profile references at least one bundle."""
        if not profiles_data:
            pytest.skip("No profiles defined in template-bundles.yml")

        empty = [name for name, cfg in profiles_data.items() if not cfg.get("bundles")]
        if empty:
            pytest.fail(f"\nProfiles with no bundles listed: {empty}")

    def test_profile_transitive_closure_is_self_consistent(self, bundles_data, profiles_data, bundle_names):
        """Test that each profile's full transitive bundle closure has no unresolvable references.

        Also checks for cross-platform contamination (github-* in gitlab profiles or vice versa).
        """
        if not profiles_data:
            pytest.skip("No profiles defined in template-bundles.yml")
        bundles = bundles_data["bundles"]

        def full_closure(seeds: list) -> tuple:
            closure: set = set()
            missing: list = []
            queue = list(seeds)
            while queue:
                b = queue.pop()
                if b in closure:
                    continue
                if b not in bundles:
                    missing.append(b)
                    continue
                closure.add(b)
                queue.extend(bundles[b].get("requires", []))
            return closure, missing

        errors = []
        for profile_name, profile_config in profiles_data.items():
            closure, missing = full_closure(profile_config.get("bundles", []))
            if missing:
                errors.append(f"  [profile:{profile_name}] unresolvable bundles: {sorted(missing)}")
            if profile_name.startswith("github"):
                cross = sorted(b for b in closure if b.startswith("gitlab"))
                if cross:
                    errors.append(f"  [profile:{profile_name}] gitlab bundles in closure: {cross}")
            elif profile_name.startswith("gitlab"):
                cross = sorted(b for b in closure if b.startswith("github"))
                if cross:
                    errors.append(f"  [profile:{profile_name}] github bundles in closure: {cross}")
        if errors:
            pytest.fail("\nProfile transitive closure issues:\n" + "\n".join(errors))
