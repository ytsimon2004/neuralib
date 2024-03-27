"""
Output files parsing for the scanimage dataset
https://docs.scanimage.org/
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from ScanImageTiffReader import ScanImageTiffReader

__all__ = ['ScanImageWrapper']


class ScanImageWrapper:
    META_INFO: dict[str, Any] = {}

    def __init__(self, sequences: np.ndarray):
        self.sequences = sequences

    @classmethod
    def load(cls, file: Path | str) -> 'ScanImageWrapper':
        sit = ScanImageTiffReader(file)
        cls._parse_meta(sit.metadata())
        return ScanImageWrapper(sit.data())

    @classmethod
    def _parse_meta(cls, meta: str):
        """parse metadata in the tif file"""
        pattern = r'(\w+\.\w+)\s*=\s*(.*)'
        matches = re.findall(pattern, meta)

        for match in matches:
            key = match[0]
            value = match[1]

            if value.isdigit():
                value = int(value)
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('[') and value.endswith(']'):
                value = value[1:-1].split()
                if all(item.isdigit() for item in value):
                    value = [int(item) for item in value]
            elif value.startswith('{') and value.endswith('}'):
                value = value[1:-1].split()
            cls.META_INFO[key] = value
