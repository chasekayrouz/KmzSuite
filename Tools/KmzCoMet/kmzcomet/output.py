"""Same-directory, no-clobber publication, including macOS ExFAT destinations."""
from contextlib import contextmanager
import ctypes
import errno
import os
from pathlib import Path
import sys
import tempfile


def publish(temporary, destination):
    try:
        os.link(temporary, destination)
    except OSError as error:
        if sys.platform != 'darwin' or error.errno not in {errno.EOPNOTSUPP, errno.ENOTSUP, errno.EPERM, errno.EXDEV}:
            raise
        # Same-volume RENAME_EXCL, already used by the suite installer. Unlike
        # an exists()+rename sequence, the filesystem rejects a racing writer.
        rename = ctypes.CDLL(None, use_errno=True).renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(os.fsencode(temporary), os.fsencode(destination), 4):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))


@contextmanager
def new_output(destination, *, checkpoint=lambda: None):
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f'Output already exists: {destination.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.kmzcomet-', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        checkpoint()
        publish(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
