"""
STATUS: NOT TESTED

"""

import psutil


def get_ram_avail() -> float:
    """Gets the remaining ram usage in %"""
    return round(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, 2)
