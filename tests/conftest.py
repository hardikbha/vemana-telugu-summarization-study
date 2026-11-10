import socket

import pytest


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    original = socket.socket.connect
    def deny(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"Unexpected network connection: {address}")
        return original(sock, address)
    monkeypatch.setattr(socket.socket, "connect", deny)
