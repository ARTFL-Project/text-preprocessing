import os
import sys


sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import lang
from .modernizer import Modernizer
from .preprocessor import PreProcessor, Token, Tokens
from .text_loader import text_loader
