import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from .modernizer import Modernizer
from .preprocessor_new import PreProcessor, Token, Tokens
from .text_loader import text_loader
