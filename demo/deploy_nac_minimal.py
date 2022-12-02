from torch_geometric.data.data import Data

from solvent.models import NACModel
from solvent.deploy import DeployNAC

nac_model: NACModel = ... # type: ignore
sample_structure: Data = ... # type: ignore

deployed_nac_model = DeployNAC(nac_model=nac_model)
print('model deployed')

sample_out = deployed_nac_model(sample_structure)
print(sample_out)
