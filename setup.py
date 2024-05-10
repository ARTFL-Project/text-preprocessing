#!/usr/bin/env python3
"""Python package install script"""

from setuptools import setup

setup(
    name="text_preprocessing",
    version="1.0.5",
    author="The ARTFL Project",
    author_email="clovisgladstone@gmail.com",
    packages=["text_preprocessing", "text_preprocessing.lang"],
    install_requires=["unidecode", "PyStemmer", "spacy>=3.7,<3.8", "orjson", "requests", "lz4", "regex"],
)
