"""Outbound connections to hosted services, over IPv4 only.

The model and speech APIs sit behind Cloudflare, which lists IPv6 addresses first. On the Pi's phone hotspot
IPv6 is configured but unroutable, and Python tries each IPv6 address before IPv4 (curl races them): every new
connection to Jev took 6.2 s to open instead of 0.03 s. Every one of these services is reachable over IPv4.
"""

from __future__ import annotations

import socket

_getaddrinfo = socket.getaddrinfo


def ipv4_getaddrinfo(host: bytes | str | None, port: bytes | str | int | None, family: int = 0,
                     type: int = 0, proto: int = 0, flags: int = 0) -> list:
    """socket.getaddrinfo that answers an unspecified family with IPv4 addresses only."""
    return _getaddrinfo(host, port, family or socket.AF_INET, type, proto, flags)


def ipv4_only() -> None:
    """Make every connection this process opens resolve to IPv4. httpx, google-genai (its Live websocket takes no
    socket family), websockets and urllib all resolve through socket.getaddrinfo; call once at startup."""
    socket.getaddrinfo = ipv4_getaddrinfo
