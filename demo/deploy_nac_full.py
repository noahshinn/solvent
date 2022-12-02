from torch_geometric.data.data import Data

from solvent.models import MCModel, NACModel
from solvent.deploy import DeployNAC

mc_model: MCModel = ... # type: ignore
nac_model_bin_0: NACModel = ... # type: ignore
nac_model_bin_1: NACModel = ... # type: ignore
nac_model_bin_2: NACModel = ... # type: ignore
nac_model_map: dict = {
    0: nac_model_bin_0,
    1: nac_model_bin_1,
    2: nac_model_bin_2,
}
STD = 3.42 # precomputed standard deviation from training parameters
sample_structure: Data = ... # type: ignore

deployed_nac_model = DeployNAC(
    nac_model=None,
    mc_model=mc_model,
    nac_model_map=nac_model_map,
    shift=0.0,
    scale=STD
)
print('model deployed')

sample_out = deployed_nac_model(sample_structure)
print(sample_out)
