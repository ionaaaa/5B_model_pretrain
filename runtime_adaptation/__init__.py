"""Canonical runtime adaptation contracts and shared FBMAE processing."""

from runtime_adaptation.contracts import CanonicalEEGWindow
from runtime_adaptation.canonical import MetadataDrivenCanonicalDecoder
from runtime_adaptation.core import SharedFBMAECore
from runtime_adaptation.inria import INRIA_CHANNELS, InriaDecoder

__all__ = [
    "CanonicalEEGWindow",
    "MetadataDrivenCanonicalDecoder",
    "SharedFBMAECore",
    "INRIA_CHANNELS",
    "InriaDecoder",
]
