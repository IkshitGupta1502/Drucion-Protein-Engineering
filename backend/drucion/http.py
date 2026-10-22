"""Verified HTTPS for Python installations without a system CA bundle."""

import ssl

import certifi


def verified_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context
