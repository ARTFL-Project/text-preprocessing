#!/usr/bin/env python3
"""Python package install script"""

from setuptools import setup


setup(
    name="text_preprocessing",
    version="0.8.3",
    author="The ARTFL Project",
    author_email="clovisgladstone@gmail.com",
    packages=["text_preprocessing", "text_preprocessing.lang"],
    install_requires=["unidecode", "PyStemmer", "spacy>=3.0", "python-rapidjson", "mmh3", "requests"],
)
