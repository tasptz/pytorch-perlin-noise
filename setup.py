"""Perlin noise install setup."""
from setuptools import setup

setup(
    name="perlin-noise",
    version="1.0.0",
    packages=["perlin_noise"],
    install_requires=[
        "torch",
    ],
)
