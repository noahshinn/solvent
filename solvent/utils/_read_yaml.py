"""
STATUS: NOT TESTED

"""

import yaml
from pathlib import Path

from typing import Dict


def read_yaml(file: str) -> Dict:
    """
    Reads a yml file and returns a Python dictionary.

    Args:
        file (str): .yml file.

    Returns:
        (Dict): Data dictionary

    """
    assert file.endswith('.yml')
    return yaml.safe_load(Path(file).read_text())['instance']
