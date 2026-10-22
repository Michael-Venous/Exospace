"""Outbound connections to hosted services, over IPv4.

The model and speech APIs sit behind Cloudflare, which lists IPv6 addresses first. On the Pi's phone hotspot
IPv6 is configured but unroutable, and Python tries each IPv6 address before IPv4 (curl races them): every new
connection to Jev took 6.2 s to open instead of 0.03 s. Every one of these services is reachable over IPv4.
"""

from __future__ import annotations

import socket

import httpx

IPV4 = socket.AF_INET  # for asyncio/websockets connections: `family=IPV4`


def ipv4_transport() -> httpx.HTTPTransport:
    """An httpx transport bound to an IPv4 source address, so it only ever connects over IPv4."""
    return httpx.HTTPTransport(local_address="0.0.0.0")
