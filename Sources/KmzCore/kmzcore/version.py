"""Canonical public version and release metadata for every KmzSuite component."""

__version__ = "0.12.0"
# Oldest macOS release the apps declare (LSMinimumSystemVersion). Every bundled
# binary must support it; packaging and the release audit enforce both.
MINIMUM_MACOS = "26.0"
COPYRIGHT = "Copyright © 2026 Chase Kayrouz"
LICENSE_NOTICE = COPYRIGHT + ". MIT License."


def about_text(name, summary=None):
    """About box text for the Qt apps; LGPLv3 asks for Qt's notice beside ours."""
    return '\n'.join([f'{name} {__version__}', *([summary] if summary else []), LICENSE_NOTICE, '',
                      'Uses Qt (© The Qt Company Ltd. and other contributors) under the GNU '
                      'LGPLv3, and other open-source software. Help → Dependency Notices '
                      'lists their licenses.'])
