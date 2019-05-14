#!/usr/bin/env python3
"""Python package install script"""

from setuptools import setup


setup(
    name="text_preprocessing",
    version="0.6",
    author="The ARTFL Project",
    author_email="clovisgladstone@gmail.com",
    packages=["text_preprocessing", "text_preprocessing.lang"],
    install_requires=["unidecode", "PyStemmer", "spacy", "python-rapidjson", "msgpack==0.5.6", "mmh3"],
)
