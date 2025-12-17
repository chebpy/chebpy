# Rhiza Test Suite

This directory contains the core test suite that flows down via SYNC action from the [jebel-quant/rhiza](https://github.com/jebel-quant/rhiza) repository.

## Purpose

These tests validate the foundational infrastructure and workflows that are shared across all Rhiza-synchronized projects:

- **Git-based workflows**: Version bumping, releasing, and tagging
- **Project structure**: Ensuring required files and directories exist
- **Build automation**: Makefile targets and commands
- **Documentation**: README code examples and docstring validation
- **Synchronization**: Template file exclusion and sync script behavior
- **Development tools**: Mock fixtures for testing in isolation

## Test Organization

- `conftest.py` - Pytest fixtures including the `git_repo` fixture for sandboxed testing
- `test_bump_script.py` - Tests for version bumping workflow
- `test_docstrings.py` - Doctest validation across all modules
- `test_git_repo_fixture.py` - Validation of the mock git repository fixture
- `test_makefile.py` - Makefile target validation using dry-runs
- `test_readme.py` - README code example execution and validation
- `test_release_script.py` - Release and tagging workflow tests
- `test_structure.py` - Project structure and file existence checks
- `test_sync_script.py` - Template synchronization exclusion tests

## Exclusion from Sync

While it is **technically possible** to exclude these tests from synchronization by adding them to the `exclude` section of your `template.yml` file, this is **not recommended**.

These tests ensure that the shared infrastructure components work correctly in your project. Excluding them means:

- ❌ No validation of version bumping and release workflows
- ❌ No automated checks for project structure requirements
- ❌ Missing critical integration tests for synced scripts
- ❌ Potential breakage when shared components are updated

## When to Exclude

You should only consider excluding specific tests if:

1. Your project has fundamentally different workflow requirements
2. You've replaced the synced scripts with custom implementations
3. You have equivalent or better test coverage for the same functionality

If you must exclude tests, do so selectively rather than excluding the entire `test_rhiza/` directory.

## Running the Tests

```bash
# Run all Rhiza tests
make test

# Run specific test files
pytest tests/test_rhiza/test_bump_script.py -v

# Run tests with detailed output
pytest tests/test_rhiza/ -vv
```

## Customization

If you need to customize or extend these tests for your project-specific needs, consider:

1. Creating additional test files in `tests/` (outside `test_rhiza/`)
2. Adding project-specific fixtures to a separate `conftest.py`
3. Keeping the synced tests intact for baseline validation

This approach maintains the safety net of standardized tests while accommodating your unique requirements.
