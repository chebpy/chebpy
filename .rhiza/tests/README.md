# Rhiza Test Suite (`.rhiza/tests/`)

This directory is **synced from [jebel-quant/rhiza](https://github.com/jebel-quant/rhiza)**
and runs in your project with `make rhiza-test`. Its job is to validate the parts of *your*
repository that Rhiza cares about — the metadata, release config, docs and docstrings that
vary per project — using the shared fixtures below.

> Tests that only exercise Rhiza's *own* template files (Makefile targets, workflow stubs,
> the project skeleton) live in Rhiza's mother-repo `tests/` suite and are **not** synced
> here — they would be identical in every consumer and can't be changed downstream. Put
> your project's own tests under your `tests/` directory, not here.

## Layout

The suite is flat — one file per concern — but **which files you get depends on the
bundles you sync**. Each is owned by whichever bundle the assertion belongs to, so a Rust
project gets the Rust manifest checks and none of the Python ones:

| file | owned by | checks |
| --- | --- | --- |
| `conftest.py` | `core` | shared fixtures (`root`, `logger`, `latest_tag`) |
| `test_release_tags.py` | `core` | the newest tag is reachable from a branch |
| `test_readme.py` | `core` | README exists; every `bash` fence parses |
| `test_pyproject.py` | `python-core` | `pyproject.toml` structure, and its `[tool.bumpversion]` block |
| `test_docstrings.py` | `python-core` | doctests across the modules in your source folder |
| `test_readme_validation.py` | `tests` | executes `python` fences and diffs them against `result` (see below) |
| `test_cargo_toml.py` | `rust-core` | `Cargo.toml` structure and the `.bumpversion.toml` wiring |
| `test_go_module.py` | `go-core` | `go.mod`, the `Version` constant, and the same wiring |

Every profile pairs `core` with exactly one language layer, so `conftest.py` is always
present alongside whichever layer's modules arrived.

### Skipping README code blocks with `+RHIZA_SKIP`

By default, every `bash` fence in `README.md` is syntax-checked (`test_readme.py`, any
language) and every `python` fence is executed (`test_readme_validation.py`, Python
projects). To mark a block as intentionally non-runnable — an illustrative snippet, an
environment-specific command — add `+RHIZA_SKIP` to the opening fence line:

~~~markdown
```python +RHIZA_SKIP
# This block will NOT be executed or syntax-checked
from my_env import some_function
some_function()
```

```bash +RHIZA_SKIP
# This bash block will NOT be syntax-checked
run-something --only-on-ci
```
~~~

Markdown renderers (including GitHub) ignore everything after the first word on
a fence line, so the block still renders as a normal highlighted code block.
Blocks without `+RHIZA_SKIP` continue to be validated as before.

## Running Tests

```bash
make rhiza-test                                  # run this suite (the usual entry point)
uv run pytest .rhiza/tests/                       # equivalent, direct invocation
uv run pytest .rhiza/tests/test_pyproject.py      # a single file
uv run pytest .rhiza/tests/ -v                    # verbose
```

## Fixtures

Defined in `conftest.py` and available to every test without import:

- `root` — repository root path (session-scoped)
- `logger` — configured logger instance (session-scoped)

`.rhiza/tests` is on `pythonpath` (see `pytest.ini`), so intra-suite imports resolve
without any `sys.path` manipulation.

## Writing Tests

- Use descriptive test names that explain what is being tested
- Group related tests in classes when appropriate
- Add docstrings to test modules and complex test functions
- Use `pytest.mark.skip` for tests that depend on optional features
