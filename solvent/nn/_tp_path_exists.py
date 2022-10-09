"""
STATUS: DEV

"""

from e3nn import o3
from typing import Union


def tp_path_exists(
        irreps_in1: Union[int, str, tuple, o3.Irrep],
        irreps_in2: Union[int, str, tuple, o3.Irrep],
        ir_out: Union[int, str, tuple, o3.Irrep]
    ) -> bool:
    irreps_in1_ = o3.Irreps(irreps_in1).simplify() # type: ignore
    irreps_in2_ = o3.Irreps(irreps_in2).simplify() # type: ignore
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1_:
        for _, ir2 in irreps_in2_:
            if ir_out in ir1 * ir2:
                return True
    return False
