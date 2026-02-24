"""Blueprint benchmark tests for downstream repositories.

This file contains example benchmark tests that demonstrate how to use
pytest-benchmark. These are **placeholder tests you should replace** with
your own meaningful benchmarks.

The Rhiza project's own tests live in ``.rhiza/tests/``.

Uses pytest-benchmark to measure and compare execution times.
"""

from __future__ import annotations


class TestExampleBenchmarks:
    """Example benchmark tests demonstrating basic usage."""

    def test_string_concatenation(self, benchmark):
        """Example: Benchmark string concatenation."""

        def concatenate_strings():
            result = ""
            for i in range(100):
                result += str(i)
            return result

        result = benchmark(concatenate_strings)
        assert len(result) > 0

    def test_list_comprehension(self, benchmark):
        """Example: Benchmark list comprehension."""

        def create_list():
            return [i * 2 for i in range(1000)]

        result = benchmark(create_list)
        assert len(result) == 1000

    def test_dictionary_operations(self, benchmark):
        """Example: Benchmark dictionary operations."""

        def dictionary_ops():
            data = {}
            for i in range(100):
                data[f"key_{i}"] = i * 2
            return sum(data.values())

        result = benchmark(dictionary_ops)
        assert result > 0

    def test_simple_computation(self, benchmark):
        """Example: Benchmark simple computation."""

        def compute_sum():
            total = 0
            for i in range(1000):
                total += i
            return total

        result = benchmark(compute_sum)
        assert result == sum(range(1000))
