"""Validation-only guard, loaded through PYTHONPATH in parent and CLI subprocesses."""
import sys


def reject_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        raise RuntimeError("Network access blocked by workshop validation")


sys.addaudithook(reject_network)
