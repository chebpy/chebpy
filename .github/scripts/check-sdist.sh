#!/usr/bin/env bash
# Rebuild the release source distribution without access to the repository's Git tags.
set -euo pipefail
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
uv build --sdist --out-dir "$workspace/dist"
python - "$workspace" <<'PY'
import email
from pathlib import Path
import sys
import tarfile

workspace = Path(sys.argv[1])
archive, = (workspace / 'dist').glob('*.tar.gz')
with tarfile.open(archive) as source:
    source.extractall(workspace / 'source', filter='data')
root, = (workspace / 'source').iterdir()
assert not (root / '.git').exists(), 'Source distribution must not contain Git metadata'
metadata = email.message_from_bytes((root / 'PKG-INFO').read_bytes())
assert (root / 'LICENSE.rst').read_bytes() == Path('LICENSE.rst').read_bytes()
assert 'LICENSE.rst' in metadata.get_all('License-File', []), 'Missing license metadata'
(workspace / 'version').write_text(metadata['Version'])
print(f'Source distribution: {metadata["Name"]} {metadata["Version"]}, license included')
PY
# Run outside the checkout, without a version override or access to its Git metadata.
(
  cd "$workspace"
  unset SETUPTOOLS_SCM_PRETEND_VERSION SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CHEBFUN
  uv build --wheel --out-dir wheels source/*
)
python - "$workspace" <<'PY'
import email
import os
from pathlib import Path
import sys
import zipfile

workspace = Path(sys.argv[1])
wheel, = (workspace / 'wheels').glob('*.whl')
with zipfile.ZipFile(wheel) as built:
    metadata_path, = (name for name in built.namelist() if name.endswith('.dist-info/METADATA'))
    metadata = email.message_from_bytes(built.read(metadata_path))
    assert metadata['Name'] == 'chebfun'
    assert metadata['Version'] == (workspace / 'version').read_text(), 'Rebuild changed the version'
    license_path, = (name for name in built.namelist() if name.endswith('.dist-info/licenses/LICENSE.rst'))
    assert built.read(license_path) == Path('LICENSE.rst').read_bytes()
    assert 'LICENSE.rst' in metadata.get_all('License-File', [])
ref = os.environ.get('GITHUB_REF', '')
if ref.startswith('refs/tags/v'):
    assert metadata['Version'] == ref.removeprefix('refs/tags/v'), 'Version does not match release tag'
print(f'Rebuilt wheel: {metadata["Name"]} {metadata["Version"]}, license included')
PY
