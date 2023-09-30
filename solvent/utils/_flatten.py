from typing import List


def flatten(l: List[List]) -> List:
    """
    Flattens a two-dimensional list.

    Args:
        l (list(list)): A two-dimensional Python list.

    Returns:
        (list): A one-dimensional Python list.
          
    """
    return [item for sublist in l for item in sublist]
