#!/usr/bin/env python3
"""Modernizer wrapper for all languages"""

import imp
import os

class modernizer:

    def __init__(self, language):
        self.language_dict = {}
        self.loaded = False
        self.language = language

    def __call__(self, word):
        if self.loaded is False:
            if self.language == "french":
                from .lang.fr_dict import french_dict
                self.language_dict = french_dict
            elif self.language == "english":
                from .lang.en_dict import english_dict
                self.language_dict = english_dict
            self.loaded = True
        return self.language_dict.get(word, word)
