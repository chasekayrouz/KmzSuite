"""Optional shared WIFF codec and viewer mzML acceleration; interpreted development remains supported."""
from setuptools import Extension, setup

# Contraction would change float rounding; the native results must equal NumPy's.
# Compression links the system zlib that CPython's zlib module uses (checked at runtime).
setup(ext_modules=[
    Extension("kmzconvert_core._native", ["Sources/KmzConvertCore/kmzconvert_core/_native.c"], optional=True,
              extra_compile_args=["-ffp-contract=off"], libraries=["z"]),
    Extension("kmzviewer._native", ["Apps/KmzViewer/kmzviewer/_native.c"], libraries=["z"], optional=True),
])
