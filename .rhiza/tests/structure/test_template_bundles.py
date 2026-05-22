"""Tests to validate that all files/folders referenced in template-bundles.yml exist.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

This module ensures that:
  - template bundle definitions reference only files that exist in the repository
  - profile definitions reference only bundles that exist
  - the local profile resolves to zero .github/workflows/* files
  - bundle dependency references (requires/recommends) point to existing bundles
"""

import pytest
import yaml


class TestTemplateBundles:
    """Tests for template-bundles.yml validation."""

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

    def test_bundles_file_exists_or_skip(self, bundles_file):
        """Test that template-bundles.yml exists, or skip if not present."""
        if not bundles_file.exists():
            pytest.skip("template-bundles.yml does not exist in this project")

    def test_all_bundle_files_exist(self, root, bundles_data):
        """Test that all files referenced in template-bundles.yml exist."""
        bundles = bundles_data["bundles"]
        all_missing = []
        total_files = 0

        # Check each bundle
        for bundle_name, bundle_config in bundles.items():
            if "files" not in bundle_config:
                continue

            files = bundle_config["files"]

            for file_path in files:
                total_files += 1
                path = root / file_path

                if not path.exists():
                    all_missing.append((bundle_name, file_path))

        # Report results
        if all_missing:
            error_msg = f"\nValidation failed: {len(all_missing)} of {total_files} files/folders are missing:\n\n"
            for bundle_name, file_path in all_missing:
                error_msg += f"  [{bundle_name}] {file_path}\n"
            pytest.fail(error_msg)

    def test_each_bundle_files_exist(self, root, bundles_data):
        """Test that files exist for each individual bundle."""
        bundles = bundles_data["bundles"]

        for bundle_name, bundle_config in bundles.items():
            if "files" not in bundle_config:
                continue

            files = bundle_config["files"]
            missing_in_bundle = []

            for file_path in files:
                path = root / file_path

                if not path.exists():
                    missing_in_bundle.append(file_path)

            if missing_in_bundle:
                error_msg = f"\nBundle '{bundle_name}' has {len(missing_in_bundle)} missing path(s):\n"
                for missing in missing_in_bundle:
                    error_msg += f"   - {missing}\n"
                pytest.fail(error_msg)

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

    def test_local_profile_contains_no_github_workflows(self, bundles_data, profiles_data):
        """Test that the local profile resolves to no .github/workflows/* files.

        The local profile is designed for projects that do not use hosted GitHub Actions.
        If any bundle it references includes .github/workflows/ paths, the invariant is broken.
        """
        if not profiles_data:
            pytest.skip("No profiles defined in template-bundles.yml")

        if "local" not in profiles_data:
            pytest.skip("No 'local' profile defined")

        bundles = bundles_data["bundles"]
        local_bundles = profiles_data["local"].get("bundles", [])
        workflow_files = []

        for bundle_name in local_bundles:
            bundle_config = bundles.get(bundle_name, {})
            for file_path in bundle_config.get("files", []):
                if ".github/workflows/" in file_path:
                    workflow_files.append(f"  [{bundle_name}] {file_path}")

        if workflow_files:
            pytest.fail(
                "\nThe 'local' profile must not resolve to any .github/workflows/* files.\n"
                "The following workflow files were found:\n" + "\n".join(workflow_files)
            )

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
