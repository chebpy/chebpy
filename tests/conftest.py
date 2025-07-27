import os
import matplotlib

if os.environ.get("CI") == "true":
    matplotlib.use("Agg")
