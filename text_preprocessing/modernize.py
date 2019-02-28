#!/usr/bin/env python3
"""Modernizer wrapper for all languages"""

from typing import Dict


class modernizer:
    """Modernizes supported languages
    Emulates a basic dict interface with dict[key] and .get method"""

    language_dict: Dict[str, str]

    def __init__(self, language: str):
        self.language_dict = {}
        self.loaded = False
        self.language = language

    def __call__(self, word: str):
        if self.loaded is False:
            if self.language == "french":
                from .lang.fr_dict import french_dict

                self.language_dict = french_dict
            elif self.language == "english":
                from .lang.en_dict import english_dict

                self.language_dict = english_dict
            self.loaded = True
        return self.language_dict.get(word, word)

    def get(self, word):
        """Retrieve modernizer word"""
        return self(word)

    def __getitem__(self, item):
        """Retrieve modernized word"""
        return self(item)
