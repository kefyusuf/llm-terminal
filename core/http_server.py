"""Local HTTP server helpers shared by loopback-only services."""

from __future__ import annotations

import socketserver
from http.server import ThreadingHTTPServer


class LocalThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that binds without reverse/FQDN resolution.

    ``HTTPServer.server_bind`` calls ``socket.getfqdn`` for every bind. That
    lookup is unnecessary for loopback-only services and can block for tens of
    seconds on constrained or misconfigured hosts. Bind through ``TCPServer``
    directly, then populate the metadata fields expected by ``HTTPServer``.
    """

    def server_bind(self) -> None:
        """Bind the listening socket without the stdlib HTTPServer FQDN lookup."""
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)
