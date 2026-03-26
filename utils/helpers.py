# helpers.py

"""Utility helpers."""


def ensure_directory(path):
    import os
    os.makedirs(path, exist_ok=True)
