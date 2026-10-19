"""IPv4-only name resolution for hosted services."""

import socket
import unittest

from brain.net import ipv4_getaddrinfo


class NetTests(unittest.TestCase):
    def test_unspecified_family_resolves_to_ipv4_only(self):
        families = {info[0] for info in ipv4_getaddrinfo("localhost", 443, type=socket.SOCK_STREAM)}
        self.assertEqual(families, {socket.AF_INET})

    def test_explicit_family_is_kept(self):
        info = ipv4_getaddrinfo("::1", 443, socket.AF_INET6, socket.SOCK_STREAM)
        self.assertEqual(info[0][0], socket.AF_INET6)


if __name__ == "__main__":
    unittest.main()
