"""Shared bundle identity, copy, fingerprint and macOS promotion primitives."""
import ctypes
from hashlib import sha256
import os
from pathlib import Path
import plistlib
import stat
import subprocess

APPS = {
    'KmzSuite.app': 'org.kmzsuite.launcher',
    'KmzConvert.app': 'org.kmzsuite.convert',
    'KmzViewer.app': 'org.kmzsuite.viewer',
    'Kmz2Viewer.app': 'org.kmzsuite.viewer2',
    'KmzCoMet.app': 'org.kmzsuite.comet',
}


def validate_bundle(app):
    app = Path(app)
    if app.is_symlink() or not app.is_dir():
        raise ValueError(f'Expected a real app bundle: {app}')
    with (app / 'Contents/Info.plist').open('rb') as handle:
        info = plistlib.load(handle)
    executable = app / 'Contents/MacOS' / app.stem
    if (info.get('CFBundleIdentifier') != APPS[app.name]
            or info.get('CFBundleExecutable') != app.stem
            or info.get('CFBundlePackageType') != 'APPL'
            or not executable.is_file() or not os.access(executable, os.X_OK)):
        raise ValueError(f'Invalid bundle identity/executable: {app}')
    for name in ('Frameworks', 'Resources', '_CodeSignature'):
        if not (app / 'Contents' / name).is_dir():
            raise ValueError(f'Missing Contents/{name}: {app}')
    for path in app.rglob('*'):
        if path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(app.resolve())):
            raise ValueError(f'Broken or external bundle symlink: {path}')
    subprocess.run(['/usr/bin/codesign', '--verify', '--deep', '--strict', str(app)], check=True)
    details = subprocess.check_output(['/usr/bin/codesign', '-d', '--verbose=2', str(app)], stderr=subprocess.STDOUT, text=True)
    if f'Identifier={APPS[app.name]}\n' not in details:
        raise ValueError(f'Signing identifier differs from bundle identifier: {app}')


def extended_attributes(path):
    # CPython exposes os.listxattr only on Linux; use macOS's no-follow APIs.
    lib = ctypes.CDLL(None, use_errno=True)
    lib.listxattr.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.listxattr.restype = ctypes.c_ssize_t
    lib.getxattr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
    lib.getxattr.restype = ctypes.c_ssize_t
    raw = os.fsencode(path)
    size = lib.listxattr(raw, None, 0, 1)  # XATTR_NOFOLLOW
    if size < 0:
        raise OSError(ctypes.get_errno(), f'Cannot inspect attributes: {path}')
    names = ctypes.create_string_buffer(size)
    if lib.listxattr(raw, names, size, 1) != size:
        raise OSError(f'Attributes changed while reading: {path}')
    for name in sorted(names.raw.rstrip(b'\0').split(b'\0')) if size else []:
        length = lib.getxattr(raw, name, None, 0, 0, 1)
        if length < 0:
            raise OSError(ctypes.get_errno(), f'Cannot read attribute: {path}')
        value = ctypes.create_string_buffer(length)
        if lib.getxattr(raw, name, value, length, 0, 1) != length:
            raise OSError(f'Attribute changed while reading: {path}')
        yield name, value.raw


def fingerprint(app):
    """Bind validation to bytes, POSIX modes, symlinks and stable extended attributes."""
    app = Path(app)
    digest = sha256()
    for path in [app, *sorted(app.rglob('*'))]:
        mode = path.lstat().st_mode
        digest.update(str(path.relative_to(app)).encode() + b'\0' + str(mode).encode() + b'\0')
        for key, value in extended_attributes(path):
            # macOS adds/updates its App Management label when signed apps run.
            # It is host access-control state, not part of the shipped payload.
            # Keep every other attribute, including quarantine and resource forks.
            if key == b'com.apple.macl':
                continue
            digest.update(key + b'\0' + sha256(value).digest())
        if stat.S_ISLNK(mode):
            digest.update(os.readlink(path).encode() + b'\0')
        elif stat.S_ISREG(mode):
            with path.open('rb') as handle:
                digest.update(sha256(handle.read()).digest())
        elif not stat.S_ISDIR(mode):
            raise ValueError(f'Unexpected bundle entry: {path}')
    return digest.hexdigest()


def bundle_hashes(directory, apps=None):
    return {name: fingerprint(Path(directory) / name) for name in (APPS if apps is None else apps)}


def copy_bundle(source, target):
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    subprocess.run(['/usr/bin/ditto', '--rsrc', '--extattr', '--acl', str(source), str(target)], check=True)


def running_apps(directory):
    """Inspect actual executable paths, rather than matching arbitrary command text."""
    lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    lib.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.proc_listallpids.restype = ctypes.c_int
    lib.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    lib.proc_pidpath.restype = ctypes.c_int
    count = lib.proc_listallpids(None, 0)
    if count <= 0:
        raise OSError('Cannot enumerate processes; refusing promotion')
    pids = (ctypes.c_int * (count + 1024))()
    count = lib.proc_listallpids(pids, ctypes.sizeof(pids))
    if count <= 0 or count >= len(pids):
        raise OSError('Cannot safely enumerate processes; retry')
    prefixes = [str(Path(directory).resolve() / name) + '/' for name in APPS]
    found = []
    for pid in pids[:count]:
        buffer = ctypes.create_string_buffer(4096)
        if lib.proc_pidpath(pid, buffer, len(buffer)) > 0:
            path = os.fsdecode(buffer.value)
            if any(path.startswith(prefix) for prefix in prefixes):
                found.append((pid, path))
    return found


def require_stopped(directory):
    found = running_apps(directory)
    if found:
        raise RuntimeError(f'Quit the running apps before continuing (nothing was killed): {found}')


def atomic_publish(stage, current):
    """Same-volume directory publish; a failed syscall leaves current untouched."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renamex_np
    rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # sys/stdio.h: RENAME_SWAP=2, RENAME_EXCL=4. EXCL prevents first-publish races.
    flag = 2 if current.exists() else 4
    if rename(os.fsencode(stage), os.fsencode(current), flag) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
