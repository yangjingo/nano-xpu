"""Bounded-memory file hashing, including on the supported Python 3.10 runtime."""
import hashlib


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
