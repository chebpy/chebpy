"""Generate a conda-forge recipe for a published release and verify it against PyPI.

This module *generates* a recipe. It publishes nothing: a package reaches conda-forge
only through a feedstock, which a conda-forge maintainer reviews and merges. The
recipe it writes is the input to that process -- the initial ``staged-recipes``
submission, or a comparison point for an update the feedstock's bot already proposed.

Two workflows run it, so a recipe produced on release and one regenerated later come
from the same code:

* ``rhiza_release.yml``'s ``conda`` job, after the tagged version was published to PyPI;
* ``rhiza_conda_recipe.yml``, dispatched by hand for any existing release tag, which
  regenerates the recipe without rebuilding or republishing anything.

Commands (run from the checked-out project root, Python 3.11+, no dependencies beyond
grayskull itself for ``generate``):

``generate``
    Ask grayskull for the exact tagged version (``name==version``) and copy the
    ``meta.yaml`` it writes to ``conda-recipe/meta.yaml``. PyPI metadata for a
    just-published release can lag the upload, so a run that writes no recipe is
    retried; a grayskull crash is not (#1701).
``verify``
    Check the recipe's version, source filename and SHA256 against PyPI's record of
    the tagged release, so an older release's recipe never counts as success (#1746).

Configuration comes from the environment: ``TAG`` (required), ``PYPI_JSON_API``
(default ``https://pypi.org/pypi``) and the optional ``GITHUB_STEP_SUMMARY``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess  # nosec B404 -- runs grayskull with a fixed argument list, never a shell
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

ATTEMPTS = 5
RETRY_SECONDS = 60
PYPI_JSON_API = "https://pypi.org/pypi"
RECIPE = Path("conda-recipe") / "meta.yaml"

# The same pattern the release workflow's tag job enforces.
_TAG_RE = re.compile(r"v[0-9]+(\.[0-9]+)*([a-zA-Z0-9.+-]*)")

# --no-use-v1-format keeps the meta.yaml this job ships: grayskull >= 3.2 otherwise writes a
# V1 recipe.yaml and crashes without conda-recipe-manager (#1701).
GRAYSKULL_FLAGS = ("--strict-conda-forge", "--no-use-v1-format")


class RecipeError(RuntimeError):
    """A recipe could not be generated, or does not describe the tagged release."""


def release_version(tag: str) -> str:
    """Return the version a release tag names, rejecting anything that is not a release tag.

    >>> release_version("v1.2.3")
    '1.2.3'
    """
    if not _TAG_RE.fullmatch(tag):
        message = f"invalid release tag {tag!r}; expected vX.Y.Z"
        raise RecipeError(message)
    return tag.removeprefix("v")


def package_name(project: Path) -> str:
    """Read ``[project].name`` from the project's ``pyproject.toml``."""
    try:
        with (project / "pyproject.toml").open("rb") as stream:
            return str(tomllib.load(stream)["project"]["name"])
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        message = f"cannot read [project].name from {project / 'pyproject.toml'}: {error}"
        raise RecipeError(message) from error


def generate(
    project: Path,
    tag: str,
    *,
    attempts: int = ATTEMPTS,
    retry_seconds: float = RETRY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Run grayskull for the tagged version and copy its recipe to ``project/conda-recipe``.

    Success is judged by the recipe on disk, not by grayskull's exit status:
    ``grayskull pypi`` prints "Package seems to be missing" on a PyPI 404 and still exits
    0, so only a missing ``meta.yaml`` is retried. A non-zero exit is a real failure that
    no amount of waiting fixes, so it fails at once (#1701).

    Args:
        project: The checked-out project root holding ``pyproject.toml``.
        tag: The release tag, e.g. ``v1.2.3``.
        attempts: How many times to ask grayskull before giving up.
        retry_seconds: How long to wait between attempts for PyPI metadata to appear.
        sleep: Injected for tests.

    Returns:
        The path of the copied recipe.
    """
    requirement = f"{package_name(project)}=={release_version(tag)}"
    grayskull = shutil.which("grayskull")
    if grayskull is None:
        message = "grayskull is not installed"
        raise RecipeError(message)
    with tempfile.TemporaryDirectory(prefix="conda-recipe-") as scratch:
        attempt = 0
        while True:
            attempt += 1
            status = subprocess.run(  # noqa: S603  # nosec B603 -- fixed argv, no shell
                [grayskull, "pypi", requirement, *GRAYSKULL_FLAGS],
                cwd=scratch,
                check=False,
            ).returncode
            if status != 0:
                message = (
                    f"grayskull exited {status} on attempt {attempt} -- a real failure, "
                    "not a PyPI propagation delay, so not retrying"
                )
                raise RecipeError(message)
            found = sorted(Path(scratch).rglob("meta.yaml"))
            if found:
                target = project / RECIPE
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(found[0], target)
                print(f"grayskull wrote a recipe for {requirement} -> {RECIPE}")
                return target
            if attempt >= attempts:
                message = (
                    f"grayskull produced no meta.yaml after {attempts} attempts -- "
                    f"PyPI metadata for {requirement} may not be available yet"
                )
                raise RecipeError(message)
            print(
                f"grayskull attempt {attempt}/{attempts} produced no recipe -- "
                f"waiting {retry_seconds:g}s for PyPI metadata to propagate"
            )
            sleep(retry_seconds)


def _render(text: str, variables: Mapping[str, str]) -> str:
    """Expand just enough Jinja for the expressions grayskull writes into a source URL.

    Anything outside ``{{ var }}``, ``{{ var[0] }}``, ``|lower`` and ``|replace(a, b)``
    fails as "cannot verify" rather than passing unchecked.

    >>> _render("{{ name[0] }}/{{ name|replace('-', '_') }}-{{ version }}", {"name": "a-b", "version": "1"})
    'a/a_b-1'
    """

    def expand(match: re.Match[str]) -> str:
        parts = [p.strip() for p in match.group(1).split("|")]
        head = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", parts[0])
        if not head or head.group(1) not in variables:
            message = f"cannot verify recipe: unsupported expression {{{{ {match.group(1).strip()} }}}}"
            raise RecipeError(message)
        value = variables[head.group(1)]
        if head.group(2) is not None:
            value = value[int(head.group(2))]
        for pipe in parts[1:]:
            replace = re.fullmatch(r"replace\(\s*[\"']([^\"']*)[\"']\s*,\s*[\"']([^\"']*)[\"']\s*\)", pipe)
            if pipe == "lower":
                value = value.lower()
            elif replace:
                value = value.replace(replace.group(1), replace.group(2))
            else:
                message = f"cannot verify recipe: unsupported filter {pipe!r}"
                raise RecipeError(message)
        return value

    return re.sub(r"{{(.*?)}}", expand, text)


def _published_sdist(api: str, name: str, version: str) -> tuple[str, dict[str, object]]:
    """Return PyPI's published version string and its single sdist entry for a release."""
    url = f"{api.rstrip('/')}/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310  # nosec B310 -- PyPI JSON API
            release = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as error:
        message = f"could not read PyPI metadata for {name} {version}: {error}"
        raise RecipeError(message) from error
    published = str(release["info"]["version"])
    sdists = [f for f in release.get("urls", []) if f.get("packagetype") == "sdist"]
    if len(sdists) != 1:
        message = (
            f"PyPI lists {len(sdists)} sdists for {name} {published}; the recipe needs exactly one to verify against"
        )
        raise RecipeError(message)
    return published, sdists[0]


def verify(project: Path, tag: str, api: str = PYPI_JSON_API) -> str:
    """Check the generated recipe against PyPI's record of the tagged release.

    A recipe on disk proves only that grayskull wrote something. It counts as success
    when it describes *this* release: the version PyPI published for the tag, and the
    exact sdist file and SHA256 PyPI serves for it (#1746).

    Args:
        project: The checked-out project root holding ``pyproject.toml`` and the recipe.
        tag: The release tag, e.g. ``v1.2.3``.
        api: Base URL of PyPI's JSON API.

    Returns:
        A one-line description of what matched.
    """
    name = package_name(project)
    published, sdist = _published_sdist(api, name, release_version(tag))
    try:
        recipe = (project / RECIPE).read_text(encoding="utf-8")
    except OSError as error:
        message = f"no generated recipe at {RECIPE}: {error}"
        raise RecipeError(message) from error
    variables = dict(re.findall(r'{%-?\s*set\s+(\w+)\s*=\s*"([^"]*)"\s*-?%}', recipe))

    def field(key: str) -> str:
        found = re.search(rf"^\s+{key}:\s*(.+?)\s*$", recipe, re.MULTILINE)
        if not found:
            message = f"generated recipe has no {key}: line"
            raise RecipeError(message)
        return _render(found.group(1), variables)

    digests = sdist.get("digests")
    expected_sha = digests.get("sha256") if isinstance(digests, dict) else None
    problems = []
    recipe_version = field("version")
    if recipe_version != published:
        problems.append(f"version {recipe_version} != {published} published for {tag}")
    recipe_file = field("url").rsplit("/", 1)[-1]
    if recipe_file != sdist["filename"]:
        problems.append(f"source {recipe_file} != published sdist {sdist['filename']}")
    recipe_sha = field("sha256")
    if recipe_sha != expected_sha:
        problems.append(f"sha256 {recipe_sha} != {expected_sha} of {sdist['filename']}")
    if problems:
        message = "conda recipe does not match the published release: " + "; ".join(problems)
        raise RecipeError(message)
    return f"conda recipe matches {sdist['filename']} ({published}, sha256 {recipe_sha})"


_SUMMARY = """\
### Conda recipe generated (not published)

`conda-recipe/meta.yaml` for **{name} {version}** was generated with grayskull and verified
against PyPI: {matched}.

It is uploaded as the `conda-recipe` workflow artifact. Nothing was published to conda-forge;
that happens only through a feedstock. Submit a new package to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes), or compare this
recipe with the update PR the feedstock's bot opens for an existing one.
"""


def main(argv: Sequence[str] | None = None, environment: Mapping[str, str] | None = None) -> int:
    """Run ``generate`` or ``verify`` and report failures as workflow error annotations."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("generate", "verify"))
    parser.add_argument("--project", type=Path, default=Path(), help="project root (default: current directory)")
    args = parser.parse_args(argv)
    env = os.environ if environment is None else environment
    try:
        tag = env.get("TAG", "")
        if args.command == "generate":
            generate(args.project, tag)
            return 0
        matched = verify(args.project, tag, env.get("PYPI_JSON_API") or PYPI_JSON_API)
        print(matched)
        if env.get("GITHUB_STEP_SUMMARY"):
            summary = _SUMMARY.format(name=package_name(args.project), version=release_version(tag), matched=matched)
            with Path(env["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
                stream.write(summary)
    except RecipeError as error:
        print(f"::error::{error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
