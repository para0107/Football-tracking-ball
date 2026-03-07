"""
conftest.py — pytest configuration for dotlumen_cv

Adds the project root to sys.path so that all test files can import
project modules (config, detector, evaluate_stage1, etc.) regardless
of which directory pytest is invoked from.

This file must sit in the dotlumen_cv/ directory (the project root).
"""

import sys
import os

# Insert project root at the front of sys.path
sys.path.insert(0, os.path.dirname(__file__))