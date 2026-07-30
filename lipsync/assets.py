"""Download and cache model files.

Everything this project needs is a direct HTTPS download with no account, no
token and no click-through licence. Files land in a user-level cache so a given
machine downloads each one once.

Override the location with ``LIPSYNC_CACHE``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

_USER_AGENT = "lipsync/0.1 (+https://github.com/TridentIntelFree/Lipsync-)"


class DownloadError(RuntimeError):
    """Raised when an asset cannot be fetched or fails verification."""


def cache_dir() -> Path:
    """Return the directory where downloaded model files are kept."""
    override = os.environ.get("LIPSYNC_CACHE")
    if override:
        base = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        base = base / "lipsync"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    url: str,
    name: str,
    sha256: str | None = None,
    progress: bool = True,
) -> Path:
    """Fetch ``url`` into the cache as ``name`` and return the local path.

    Downloads to a temporary file and renames on success, so an interrupted run
    can never leave a truncated file that later looks like a valid cache hit.
    """
    target = cache_dir() / name
    if target.exists():
        if sha256 is None or _sha256(target) == sha256:
            return target
        target.unlink()  # cached copy is corrupt or stale; re-fetch

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    tmp_path: Path | None = None
    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length") or 0)
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".part", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                read = 0
                while chunk := response.read(1 << 20):
                    tmp.write(chunk)
                    read += len(chunk)
                    if progress and total:
                        pct = 100 * read / total
                        print(
                            f"\r  {name}: {pct:5.1f}%  "
                            f"({read >> 20}/{total >> 20} MiB)",
                            end="",
                            flush=True,
                        )
        if progress and total:
            print()
    except urllib.error.HTTPError as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise DownloadError(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise DownloadError(
            f"Could not reach {url}: {exc.reason}. "
            "Check network access to the host, or pre-place the file at "
            f"{target}"
        ) from exc

    assert tmp_path is not None
    if sha256 is not None:
        got = _sha256(tmp_path)
        if got != sha256:
            tmp_path.unlink(missing_ok=True)
            raise DownloadError(
                f"Checksum mismatch for {name}: expected {sha256}, got {got}"
            )

    shutil.move(str(tmp_path), str(target))
    return target
