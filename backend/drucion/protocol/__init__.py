"""Bench protocol generation with explicit provenance for every number.

See :mod:`drucion.protocol.model` for the provenance rule the whole package
exists to enforce: a protocol may not contain an invented number presented as
a measurement.
"""

from .model import Fact, Protocol, ProtocolError, Section, Tier
from .generate import generate_protocol
from .render import render_markdown
from .uniprot import UniProtError, UniProtRecord, fetch_uniprot

__all__ = [
    "Fact",
    "Protocol",
    "ProtocolError",
    "Section",
    "Tier",
    "generate_protocol",
    "render_markdown",
    "UniProtError",
    "UniProtRecord",
    "fetch_uniprot",
]
