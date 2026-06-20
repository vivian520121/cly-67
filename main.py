#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ocr_tool.cli import cli

if __name__ == '__main__':
    cli()
