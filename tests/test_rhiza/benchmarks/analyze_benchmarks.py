# /// script
# dependencies = [
#   "pandas",
#   "plotly",
# ]
# ///

"""Analyze pytest-benchmark results and visualize them.

This script reads a local ``benchmarks.json`` file produced by pytest-benchmark,
prints a reduced table with benchmark name, mean milliseconds, and operations
per second, and renders an interactive Plotly bar chart of mean runtimes.
"""

# Python script: read JSON, create reduced table, and Plotly chart
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

# check if the file exists at all
if not Path(__file__).parent.joinpath("benchmarks.json").exists():
    logging.warning("benchmarks.json not found; skipping analysis and exiting successfully.")
    sys.exit(0)

# Load pytest-benchmark JSON
with open(Path(__file__).parent / "benchmarks.json") as f:
    # Do not continue if JSON is invalid (e.g. empty file)
    try:
        data = json.load(f)
    except json.JSONDecodeError:
        logging.warning("benchmarks.json is invalid or empty; skipping analysis and exiting successfully.")
        sys.exit(0)

# Validate structure: require a 'benchmarks' list
if not isinstance(data, dict) or "benchmarks" not in data or not isinstance(data["benchmarks"], list):
    logging.warning("benchmarks.json missing valid 'benchmarks' list; skipping analysis and exiting successfully.")
    sys.exit(0)

# Extract relevant info: Benchmark name, Mean (ms), OPS
benchmarks = []
for bench in data["benchmarks"]:
    mean_s = bench["stats"]["mean"]
    benchmarks.append(
        {
            "Benchmark": bench["name"],
            "Mean_ms": mean_s * 1000,  # convert seconds → milliseconds
            "OPS": 1 / mean_s,
        }
    )

# Create DataFrame and sort fastest → slowest
df = pd.DataFrame(benchmarks)
df = df.sort_values("Mean_ms")

# 3️⃣ Display reduced table
print(df[["Benchmark", "Mean_ms", "OPS"]].to_string(index=False, float_format="%.3f"))

# 4️⃣ Create interactive Plotly bar chart
fig = px.bar(
    df,
    x="Benchmark",
    y="Mean_ms",
    color="Mean_ms",
    color_continuous_scale="Viridis_r",
    title="Benchmark Mean Runtime (ms) per Test",
    text="Mean_ms",
)

fig.update_traces(texttemplate="%{text:.2f} ms", textposition="outside")
fig.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Mean Runtime (ms)",
    coloraxis_colorbar=dict(title="ms"),
    height=600,
    margin=dict(t=100, b=200),
)

fig.show()

# plotly fig to html
fig.write_html(Path(__file__).parent / "benchmarks.html")
