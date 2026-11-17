from pathlib import Path
from xml.parsers import expat

from .cancel import Cancelled


LEGACY_INPUT_MESSAGE = "Legacy input received. Please reconvert using the current KmzConvert."


class LegacyInputError(ValueError):
    """An obsolete conversion contract must not be displayed as ordinary data."""


def require_current_input(path: str | Path, *, is_cancelled=None) -> None:
    """Check processing metadata before arrays, without retaining an XML tree.

    Recognize the conversion-contract field roles, not a retired product name.
    Ordinary vendor mzML has no such contract and remains supported. Only fields
    within processing methods are considered; filenames, notes and binary data
    do not establish a conversion contract.
    """
    owners = {
        " decoding limitations": "KmzConvert",
        " sparse profile version": "KmzSuite",
        " mass correction version": "KmzSuite",
    }
    stack = []

    class MetadataComplete(Exception):
        pass

    def start(tag, attrs):
        tag = tag.rsplit("}", 1)[-1]
        if tag in {"spectrumList", "chromatogramList"}:
            raise MetadataComplete()
        if tag == "userParam" and stack[-2:] == ["dataProcessing", "processingMethod"]:
            name = attrs.get("name", "")
            for suffix, owner in owners.items():
                if name.endswith(suffix) and name != owner + suffix:
                    raise LegacyInputError(LEGACY_INPUT_MESSAGE)
        stack.append(tag)

    parser = expat.ParserCreate(namespace_separator="}")
    parser.StartElementHandler = start
    parser.EndElementHandler = lambda tag: stack.pop()
    try:
        with open(path, "rb") as handle:
            while True:
                if is_cancelled is not None and is_cancelled():
                    raise Cancelled()
                block = handle.read(64 * 1024)
                parser.Parse(block, not block)
                if not block:
                    break
    except MetadataComplete:
        return
    except expat.ExpatError as error:
        raise ValueError("Invalid mzML metadata.") from error

def looks_like_mzml(path: str | Path) -> bool:
    with open(path, "rb") as handle:
        head = handle.read(4096)
    return b"<mzML" in head or b"<indexedmzML" in head
