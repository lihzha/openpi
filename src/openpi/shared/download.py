import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import time
import urllib.parse

from etils import epath  # optional, but handy
import filelock
import fsspec
import fsspec.generic
import tensorflow as tf  # new

# Environment variable to control cache directory path, ~/.cache/openpi will be used by default.
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"

logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path | epath.Path:
    """Return the cache directory, creating it if necessary.

    Environment variable `_OPENPI_DATA_HOME` must be set to either
    a local POSIX path **or** a `gs://` URI.
    """
    cache_dir_str = os.getenv(_OPENPI_DATA_HOME)
    if cache_dir_str is None:
        raise ValueError(f"Environment variable {_OPENPI_DATA_HOME} must be set.")

    # gs:// --> Cloud Storage
    if cache_dir_str.startswith("gs://"):
        cache_dir = epath.Path(cache_dir_str)
        tf.io.gfile.makedirs(str(cache_dir))  # no-op if it already exists
        return cache_dir  # behaves like pathlib.Path

    # Local filesystem
    cache_dir = pathlib.Path(cache_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # _set_folder_permission(cache_dir)           # keep if still relevant
    return cache_dir


# Helper ────────────────────────────────────────────────────────────────────────
def _is_gcs(path: str | pathlib.Path) -> bool:
    return str(path).startswith("gs://")


def _join(*parts: str) -> str:
    """`os.path.join` that works for both local FS and GCS."""
    if _is_gcs(parts[0]):
        return tf.io.gfile.join(*parts)
    return str(pathlib.Path(parts[0], *parts[1:]))


# Main API ──────────────────────────────────────────────────────────────────────
def maybe_download(
    url: str,
    *,
    force_download: bool = False,
    **kwargs,
) -> pathlib.Path | epath.Path:
    """
    Download a file/dir to the cache (local or `gs://`) and return its absolute path.
    """
    parsed = urllib.parse.urlparse(url)

    # ── 1. Short-circuit for local *input* URLs ────────────────────────────────
    if parsed.scheme == "":
        p = pathlib.Path(url)
        if not p.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return p.resolve()

    if parsed.scheme == "gs" and parsed.netloc in ("pi0-cot", "droid-cot"):
        return url

    # ── 2. Build cache path ────────────────────────────────────────────────────
    cache_dir = get_cache_dir()  # could be local or gs://
    remote_cache = _is_gcs(cache_dir)

    cache_path = _join(cache_dir, parsed.netloc, parsed.path.lstrip("/"))
    scratch_path = f"{cache_path}.partial"
    lock_path = f"{cache_path}.lock"
    scratch_commit_success = _join(scratch_path, "COMMIT_SUCCESS")

    # ── 3. Cache-validation check ─────────────────────────────────────────────
    def _exists(p: str) -> bool:
        return tf.io.gfile.exists(p) if remote_cache else pathlib.Path(p).exists()

    def _is_complete_remote_dir(p: str) -> bool:
        # Directory is considered complete if Orbax metadata exists.
        return tf.io.gfile.isdir(p) and tf.io.gfile.exists(_join(p, "_METADATA"))

    invalidate_cache = False
    if _exists(cache_path):
        if force_download:
            invalidate_cache = True
        elif remote_cache:
            # Remote cache: if it's a directory, require Orbax metadata; if it's a file, existence is enough.
            if tf.io.gfile.isdir(cache_path):
                if not _is_complete_remote_dir(cache_path):
                    invalidate_cache = True
                else:
                    print(f"Cache hit: {cache_path}")
                    return epath.Path(cache_path)
            else:
                # Remote file exists → cache hit
                print(f"Cache hit: {cache_path}")
                return epath.Path(cache_path)
        else:
            # Local cache invalidation policy
            if _should_invalidate_cache(cache_dir, pathlib.Path(cache_path)):
                invalidate_cache = True
            else:
                print(f"Cache hit: {cache_path}")
                return pathlib.Path(cache_path)

    # Ensure scratch location is clean before starting a new copy
    if _exists(scratch_path):
        logger.info("Removing existing scratch path: %s", scratch_path)
        if remote_cache:
            if tf.io.gfile.isdir(scratch_path):
                try:
                    tf.io.gfile.rmtree(scratch_path)
                except tf.errors.NotFoundError:
                    # If it no longer exists, nothing to do.
                    pass
            else:
                try:
                    tf.io.gfile.remove(scratch_path)
                except tf.errors.NotFoundError:
                    # If it no longer exists, nothing to do.
                    pass
        else:
            p = pathlib.Path(scratch_path)
            if p.exists():
                shutil.rmtree(p) if p.is_dir() else p.unlink()

    # ── 4. Acquire lock (local FS only) ────────────────────────────────────────
    # GCS locking is best-effort with atomic object creation; we skip `filelock`.
    lock = filelock.FileLock(lock_path) if not remote_cache else None

    try:
        if lock:
            lock.acquire()

        # Remove expired/incomplete cache entry
        if invalidate_cache and _exists(cache_path):
            logger.info("Removing expired cached entry: %s", cache_path)
            if remote_cache:
                try:
                    if tf.io.gfile.isdir(cache_path):
                        tf.io.gfile.rmtree(cache_path)
                    elif tf.io.gfile.exists(cache_path):
                        tf.io.gfile.remove(cache_path)
                except tf.errors.NotFoundError:
                    # If it no longer exists, nothing to do.
                    pass
            else:
                p = pathlib.Path(cache_path)
                if p.exists():
                    shutil.rmtree(p) if p.is_dir() else p.unlink()

        # ── 5. Download via fsspec to a scratch location ──────────────────────
        logger.info("Downloading %s to %s", url, scratch_path)
        print(f"****Downloading {url} to {scratch_path}****")
        _download_fsspec(url, scratch_path, **kwargs)

        # ── 6. Atomic rename to final location ────────────────────────────────
        # Mark completion inside the scratch directory (only for directories)
        try:
            if remote_cache and tf.io.gfile.isdir(scratch_path):
                # Create an Orbax-style completion marker so the cached checkpoint
                # is considered complete during restore if upstream is missing it.
                with tf.io.gfile.GFile(scratch_commit_success, "w") as f:
                    f.write("ok")
            elif not remote_cache and pathlib.Path(scratch_path).is_dir():
                pathlib.Path(scratch_commit_success).write_text("ok")
        except Exception:
            # Marker is best-effort; continue even if it fails.
            pass

        if remote_cache:
            try:
                tf.io.gfile.rename(scratch_path, cache_path, overwrite=True)
            except Exception:
                # Marker is best-effort; continue even if it fails.
                pass
        else:
            shutil.move(scratch_path, cache_path)
            _ensure_permissions(pathlib.Path(cache_path))

    except PermissionError as e:
        msg = f"Permission error while downloading {url}. Try removing the cache entry: rm -rf {cache_path}*"
        raise PermissionError(msg) from e

    finally:
        if lock and lock.is_locked:
            lock.release()

    return epath.Path(cache_path) if remote_cache else pathlib.Path(cache_path)


# def maybe_download(url: str, *, force_download: bool = False, **kwargs) -> pathlib.Path:
#     """Download a file or directory from a remote filesystem to the local cache, and return the local path.

#     If the local file already exists, it will be returned directly.

#     It is safe to call this function concurrently from multiple processes.
#     See `get_cache_dir` for more details on the cache directory.

#     Args:
#         url: URL to the file to download.
#         force_download: If True, the file will be downloaded even if it already exists in the cache.
#         **kwargs: Additional arguments to pass to fsspec.

#     Returns:
#         Local path to the downloaded file or directory. That path is guaranteed to exist and is absolute.
#     """
#     # Don't use fsspec to parse the url to avoid unnecessary connection to the remote filesystem.
#     parsed = urllib.parse.urlparse(url)

#     # Short circuit if this is a local path.
#     if parsed.scheme == "":
#         path = pathlib.Path(url)
#         if not path.exists():
#             raise FileNotFoundError(f"File not found at {url}")
#         return path.resolve()

#     cache_dir = get_cache_dir()

#     local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
#     local_path = local_path.resolve()

#     # Check if the cache should be invalidated.
#     invalidate_cache = False
#     if local_path.exists():
#         if force_download or _should_invalidate_cache(cache_dir, local_path):
#             invalidate_cache = True
#         else:
#             return local_path

#     try:
#         lock_path = local_path.with_suffix(".lock")
#         with filelock.FileLock(lock_path):
#             # Ensure consistent permissions for the lock file.
#             _ensure_permissions(lock_path)
#             # First, remove the existing cache if it is expired.
#             if invalidate_cache:
#                 logger.info(f"Removing expired cached entry: {local_path}")
#                 if local_path.is_dir():
#                     shutil.rmtree(local_path)
#                 else:
#                     local_path.unlink()

#             # Download the data to a local cache.
#             logger.info(f"Downloading {url} to {local_path}")
#             scratch_path = local_path.with_suffix(".partial")
#             _download_fsspec(url, scratch_path, **kwargs)

#             shutil.move(scratch_path, local_path)
#             _ensure_permissions(local_path)

#     except PermissionError as e:
#         msg = (
#             f"Local file permission error was encountered while downloading {url}. "
#             f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
#         )
#         raise PermissionError(msg) from e

#     return local_path


def _copy_dir_gcs(src: str, dst: str) -> None:
    """Recursively copy gs://src → gs://dst (preserves tree)."""
    for root, _, files in tf.io.gfile.walk(src):
        rel_root = root[len(src) :].lstrip("/")
        for fname in files:
            s = tf.io.gfile.join(root, fname)
            d = tf.io.gfile.join(dst, rel_root, fname)
            dst_dir = os.path.dirname(str(d))
            tf.io.gfile.makedirs(dst_dir)
            tf.io.gfile.copy(s, d, overwrite=True)


def _download_fsspec(url: str, local_path: pathlib.Path | str, **kwargs) -> None:
    # ── Fast-path: src & dst are both gs:// ───────────────────────────────────
    if _is_gcs(url) and _is_gcs(local_path):
        fs, _ = fsspec.core.url_to_fs(url, **kwargs)
        info = fs.info(url)
        if info["type"] == "directory":
            _copy_dir_gcs(url, str(local_path))
        else:
            dst_dir = os.path.dirname(str(local_path))
            tf.io.gfile.makedirs(dst_dir)
            tf.io.gfile.copy(url, str(local_path), overwrite=True)
        return
    raise NotImplementedError(
        "Downloading from remote filesystem to local cache is only supported for gs:// URLs. "
        "Please use a local file path or a gs:// URL."
    )


def ensure_commit_success(dir_path: str) -> None:
    """Create commit_success.txt in dir_path if missing.

    Works for both local and gs:// paths. Best-effort: ignores errors.
    """
    try:
        if _is_gcs(dir_path):
            marker = _join(dir_path, "commit_success.txt")
            if not tf.io.gfile.exists(marker):
                with tf.io.gfile.GFile(marker, "w") as f:
                    f.write("ok")
        else:
            marker_path = pathlib.Path(dir_path) / "commit_success.txt"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            if not marker_path.exists():
                marker_path.write_text("ok")
    except Exception:
        # Best-effort only
        pass


def mirror_checkpoint_to_remote_cache(url: str, **kwargs) -> str:
    """Ensure a checkpoint at `url` (gs://...) is mirrored into the remote cache.

    The mirror location is: gs://<OPENPI_DATA_HOME>/<bucket>/<path>

    Returns the mirror path (as a gs:// string). If OPENPI_DATA_HOME is not a
    GCS path, returns the original url unchanged.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "gs":
        return url

    cache_dir = get_cache_dir()
    if not _is_gcs(cache_dir):
        return url

    cache_root = str(cache_dir)
    mirror_path = _join(cache_root, parsed.netloc, parsed.path.lstrip("/"))
    scratch_path = f"{mirror_path}.partial"
    scratch_commit_success = _join(scratch_path, "COMMIT_SUCCESS")

    def _exists(p: str) -> bool:
        return tf.io.gfile.exists(p)

    # If checkpoint already present at mirror location, skip copying if it appears
    # complete (directory with _METADATA file present).
    metadata_marker = _join(mirror_path, "_METADATA")
    if tf.io.gfile.isdir(mirror_path) and _exists(metadata_marker):
        ensure_commit_success(mirror_path)
        return mirror_path

    # Clean scratch
    if _exists(scratch_path):
        if tf.io.gfile.isdir(scratch_path):
            try:
                tf.io.gfile.rmtree(scratch_path)
            except tf.errors.NotFoundError:
                pass
        else:
            try:
                tf.io.gfile.remove(scratch_path)
            except tf.errors.NotFoundError:
                pass

    # Copy upstream → scratch
    logger.info("Mirroring %s → %s", url, mirror_path)
    _download_fsspec(url, scratch_path, **kwargs)

    # Add markers for Orbax compatibility (best-effort)
    try:
        if tf.io.gfile.isdir(scratch_path):
            with tf.io.gfile.GFile(scratch_commit_success, "w") as f:
                f.write("ok")
    except Exception:
        pass

    # Rename into place
    try:
        tf.io.gfile.rename(scratch_path, mirror_path, overwrite=True)
    except Exception:
        pass

    return mirror_path

# def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
#     """Download a file from a remote filesystem to the local cache, and return the local path."""
#     fs, _ = fsspec.core.url_to_fs(url, **kwargs)
#     info = fs.info(url)
#     # Folders are represented by 0-byte objects with a trailing forward slash.
#     if is_dir := (info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))):
#         total_size = fs.du(url)
#     else:
#         total_size = info["size"]
#     with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
#         executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
#         future = executor.submit(fs.get, url, local_path, recursive=is_dir)
#         while not future.done():
#             current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
#             pbar.update(current_size - pbar.n)
#             time.sleep(1)
#         pbar.update(total_size - pbar.n)


def _set_permission(path: pathlib.Path, target_permission: int):
    """chmod requires executable permission to be set, so we skip if the permission is already match with the target."""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """Set folder permission to be read, write and searchable."""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """Since we are sharing cache directory with containerized runtime as well as training script, we need to
    ensure that the cache directory has the correct permissions.
    """

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """Set all files to be read & writable, if it is a script, keep it as a script."""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _get_mtime(year: int, month: int, day: int) -> float:
    """Get the mtime of a given date at midnight UTC."""
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
    return time.mktime(date.timetuple())


# Map of relative paths, defined as regular expressions, to expiration timestamps (mtime format).
# Partial matching will be used from top to bottom and the first match will be chosen.
# Cached entries will be retained only if they are newer than the expiration timestamp.
_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """Invalidate the cache if it is expired. Return True if the cache was invalidated."""

    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            # Remove if not newer than the expiration timestamp.
            return local_path.stat().st_mtime <= expire_time

    return False
