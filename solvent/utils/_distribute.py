from typing import List
from solvent import types


def distribute(n: int, k: int) -> List[types.PosIntTuple]:
    """
    Distribute N task indexes across K cores.
    
    Args:
        n (int): Number of tasks.
        k (int): Number of available cores.

    Returns:
        sets (List(tuple(int, int))): Sets (i_0, i_1) in which i_0 is the
            start index and i_1 is the end index of M, where M is a list
            of tasks.
        
    """
    min_tasks = int((n - n % k) / k)
    tasks = [min_tasks]*k
    idx = 0

    while n - (min_tasks * k + idx) > 0:
        tasks[idx] += 1
        idx += 1

    sets = []
    cur = 0
    for num_tasks in tasks:
        sets.append((cur, cur+num_tasks))
        cur+= num_tasks

    return sets
