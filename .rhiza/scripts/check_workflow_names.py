#!/usr/bin/env python3
"""Script to ensure GitHub Actions workflows have the (RHIZA) prefix."""

import sys

import yaml


def check_file(filepath):
    """Check if the workflow file has the correct name prefix and update if needed.

    Args:
        filepath: Path to the workflow file.

    Returns:
        bool: True if file is correct, False if it was updated or has errors.
    """
    with open(filepath) as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML {filepath}: {exc}")
            return False

    if not isinstance(content, dict):
        # Empty file or not a dict
        return True

    name = content.get("name")
    if not name:
        print(f"Error: {filepath} missing 'name' field.")
        return False

    if not name.startswith("(RHIZA) "):
        print(f"Updating {filepath}: name '{name}' -> '(RHIZA) {name}'")

        # Read file lines to perform replacement while preserving comments
        with open(filepath) as f_read:
            lines = f_read.readlines()

        with open(filepath, "w") as f_write:
            replaced = False
            for line in lines:
                # Replace only the top-level name field (assumes it starts at beginning of line)
                if not replaced and line.startswith("name:"):
                    # Check if this line corresponds to the extracted name.
                    # Simple check: does it contain reasonable parts of the name?
                    # Or just blinding replace top-level name:
                    # We'll use quotes to be safe
                    f_write.write(f'name: "(RHIZA) {name}"\n')
                    replaced = True
                else:
                    f_write.write(line)

        return False  # Fail so pre-commit knows files were modified

    return True


def main():
    """Execute the script."""
    files = sys.argv[1:]
    failed = False
    for f in files:
        if not check_file(f):
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
