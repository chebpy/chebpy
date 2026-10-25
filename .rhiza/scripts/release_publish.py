"""Fail-closed publisher guards shared by the shipped GitHub release workflows.

Run with Python 3.11+ from the checked-out release root. Configuration comes from
the step environment; only validated values are appended to GITHUB_OUTPUT.
These guards never publish packages or print untrusted configuration values.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit


class PublisherError(ValueError):
    """Reject publisher configuration with a credential-free diagnostic."""


def _valid_hostname(host: str) -> bool:
    """Check DNS label syntax without resolving or contacting the host."""
    return len(host) <= 253 and all(
        re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in host.split(".")
    )


def _unsafe_characters(value: str) -> bool:
    """Detect characters unsafe for the single-line URL output contract."""
    return any(c.isspace() or ord(c) < 32 or ord(c) == 127 or c in '<>\\`"' for c in value)


def valid_endpoint(value: str) -> bool:
    """Accept credential-free HTTPS upload endpoints, including private hosts."""
    try:
        url = urlsplit(value)
        host = url.hostname or ""
        try:
            ipaddress.ip_address(host)
            valid_host = True
        except ValueError:
            valid_host = len(host) <= 253 and _valid_hostname(host.removesuffix("."))
        return bool(
            url.scheme == "https"
            and url.netloc
            and not url.username
            and not url.password
            and "@" not in url.netloc
            and valid_host
            and not _unsafe_characters(value)
            and (url.port is None or 1 <= url.port <= 65535)
            and not url.netloc.endswith(":")
        )
    except ValueError:
        return False


def validate_artifact_url(value: str) -> dict[str, str]:
    """Return the optional public artifact URL output, or reject it without echoing it."""
    if not value:
        return {}
    try:
        url = urlsplit(value)
        host = url.hostname or ""
        try:
            valid_host = ipaddress.ip_address(host).is_global
        except ValueError:
            valid_host = bool(
                "." in host
                and not host.endswith(".")
                and re.search(r"[A-Za-z]", host.rsplit(".", 1)[-1])
                and not host.endswith((".localhost", ".local", ".internal"))
                and _valid_hostname(host)
            )
        valid = (
            value.startswith("https://")
            and url.scheme == "https"
            and valid_host
            and not url.username
            and not url.password
            and "@" not in url.netloc
            and (url.port is None or 1 <= url.port <= 65535)
            and not url.netloc.endswith(":")
            and not _unsafe_characters(unquote(value))
        )
    except ValueError:
        valid = False
    if not valid:
        message = "Publisher artifact-url must be one public credential-free HTTPS URL"
        raise PublisherError(message)
    return {"artifact_url": value}


def validate_publisher(mode: str, buildable: str, endpoint: str, tag: str, root: Path = Path(".")) -> dict[str, str]:
    """Resolve publisher outputs, checking metadata and custom-action availability."""
    mode = mode or "pypi"
    if mode not in {"pypi", "custom", "none"}:
        message = "RELEASE_PUBLISHER must be pypi, custom, or none"
        raise PublisherError(message)
    if buildable not in {"true", "false"}:
        message = "Missing or invalid buildable output"
        raise PublisherError(message)
    publish = mode != "none" and buildable == "true"
    if mode == "pypi" and publish:
        with (root / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
        publish = "Private :: Do Not Upload" not in project.get("classifiers", [])
    if mode == "pypi" and publish and endpoint and not valid_endpoint(endpoint):
        message = "Invalid PYPI_REPOSITORY_URL; expected a credential-free HTTPS endpoint"
        raise PublisherError(message)
    if mode == "custom":
        if not publish:
            message = "Custom publishing requires a buildable Python package"
            raise PublisherError(message)
        action = root / ".github/actions/release-publish"
        if not any((action / name).is_file() for name in ("action.yml", "action.yaml")):
            message = "Custom publishing requires .github/actions/release-publish/action.yml or action.yaml"
            raise PublisherError(message)
    return {
        "publisher": mode,
        "should_publish": str(publish).lower(),
        "public_pypi": str(publish and mode == "pypi" and not endpoint).lower(),
        "version": tag.removeprefix("v"),
    }


def require_distributions(directory: Path) -> None:
    """Require at least one already-built wheel or sdist, without modifying files."""
    if not any(p.is_file() for pattern in ("*.whl", "*.tar.gz") for p in directory.glob(pattern)):
        message = "Required dist artifact contains no wheel or sdist"
        raise PublisherError(message)


def main(argv: Sequence[str] | None = None, environment: Mapping[str, str] | None = None) -> int:
    """Run a workflow guard and report expected errors without leaking configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-publisher", "require-distributions", "validate-artifact-url"))
    args = parser.parse_args(argv)
    env = os.environ if environment is None else environment
    try:
        outputs = {}
        if args.command == "validate-publisher":
            outputs = validate_publisher(
                env["RELEASE_PUBLISHER"], env["BUILDABLE"], env["PYPI_REPOSITORY_URL"], env["TAG"]
            )
        elif args.command == "require-distributions":
            require_distributions(Path(env["ARTIFACTS_DIR"]))
        else:
            outputs = validate_artifact_url(env["ARTIFACT_URL"])
        if outputs:
            with Path(env["GITHUB_OUTPUT"]).open("a") as output:
                for key, value in outputs.items():
                    print(f"{key}={value}", file=output)
    except PublisherError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, ValueError, TypeError, AttributeError):
        print("::error::Missing or invalid publisher configuration or output file", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
