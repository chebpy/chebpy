# Benchmarks

This folder contains benchmark analysis scripts for the project.
It does **not** contain the benchmark tests themselves,
which are expected to be located in `tests/benchmarks/`.

## Files

- `analyze_benchmarks.py` – Script to analyze benchmark results and generate reports.
- `README.md` – This file.

## Running Benchmarks

Benchmarks are executed via the Makefile or `pytest`:

```bash
# Using Makefile target
make benchmark

# Or manually via uv
uv run pytest tests/benchmarks/ \
    --benchmark-only \
    --benchmark-histogram=tests/test_rhiza/benchmarks/benchmarks \
    --benchmark-json=tests/test_rhiza/benchmarks/benchmarks.json

# Analyze results
uv run tests/test_rhiza/benchmarks/analyze_benchmarks.py
```
## Output

* benchmarks.json – JSON file containing benchmark results.
* Histogram plots – Generated in the folder specified by --benchmark-histogram (by default tests/test_rhiza/benchmarks/benchmarks).

## Notes

* Ensure pytest-benchmark (v5.2.3) and pygal (v3.1.0) are installed.
* The Makefile target handles this automatically.
* analyze_benchmarks.py reads the JSON output and generates human-readable summaries and plots.

## Example benchmark tests

```python
import time

def something(duration=0.001):
    """
    Function that needs some serious benchmarking.
    """
    time.sleep(duration)
    # You may return anything you want, like the result of a computation
    return 123

def test_my_stuff(benchmark):
    # benchmark something
    result = benchmark(something)

    # Extra code, to verify that the run completed correctly.
    # Sometimes you may want to check the result, fast functions
    # are no good if they return incorrect results :-)
    assert result == 123
```

Please note the usage of the `@pytest.mark.benchmark` fixture
which becomes available after installing pytest-benchmark.

See https://pytest-benchmark.readthedocs.io/en/stable/ for more details.



