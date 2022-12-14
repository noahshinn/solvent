import torch

from solvent.data import NACDeployedDataset
from solvent.models import MCModel, NACModel
from solvent.deploy import DeployNAC

NBINS = 6
_MC_MODEL_FILE = f'./models/nac-6-bin-mc-model.pth'
NAC_BIN_1_MODEL_FILE = './models/nac-6-bin-1-inf-model.pth'
NAC_BIN_2_MODEL_FILE = './models/nac-6-bin-2-inf-model.pth'
NAC_BIN_3_MODEL_FILE = './models/nac-6-bin-3-inf-model.pth'
NAC_BIN_4_MODEL_FILE = './models/nac-6-bin-4-inf-model.pth'
DATA_FILE = './nac-meci.json'
NSTRUCTURES = 696
ONE_HOT_KEY = {
    'H': [1., 0., 0.],
    'C': [0., 1., 0.],
    'O': [0., 0., 1.]
}
_NATOM_TYPES = len(ONE_HOT_KEY.keys())
BATCH_SIZE = 1
SPLIT = 0.9
NATOMS = 6
NAC_BIN_1_MU = 0.0
NAC_BIN_2_MU = 0.0
NAC_BIN_3_MU = 0.0
NAC_BIN_4_MU = 0.0
NAC_BIN_1_STD = 2.503755928074828
NAC_BIN_2_STD = 3.3093652451394022
NAC_BIN_3_STD = 4.254107338141916
NAC_BIN_4_STD = 6.685324855015012


ds = NACDeployedDataset(
    json_file=DATA_FILE,
    nstructures=NSTRUCTURES,
    one_hot_key=ONE_HOT_KEY,
)
ds.load()
print(f'loaded dataset: {len(ds)} structures')

dataloader = ds.gen_dataloader()
print('loader initialized')

mc_model = MCModel(
    irreps_in=f'{_NATOM_TYPES}x0e',
    hidden_sizes=[125, 40, 25, 15],
    natoms=NATOMS,
    nclasses=NBINS,
    nlayers=4,
    max_radius=4.6,
    nbasis_funcs=8,
    nradial_layers=3,
    nradial_neurons=128,
    navg_neighbors=5.0
)
mc_model.load_state_dict(torch.load(_MC_MODEL_FILE)['model'])
mc_model.eval()
print('classification model initialized')

nac_regr_model_args = {
    "irreps_in": f'{_NATOM_TYPES}x0e',
    "hidden_sizes": [125, 40, 25, 15],
    "natoms": NATOMS,
    "nlayers": 4,
    "max_radius": 4.6,
    "nbasis_funcs": 8,
    "nradial_layers": 3,
    "nradial_neurons": 128,
    "navg_neighbors": 5.0
}
nac_bin_1_model = NACModel(**nac_regr_model_args)
nac_bin_2_model = NACModel(**nac_regr_model_args)
nac_bin_3_model = NACModel(**nac_regr_model_args)
nac_bin_4_model = NACModel(**nac_regr_model_args)
nac_bin_1_model.load_state_dict(torch.load(NAC_BIN_1_MODEL_FILE)['model'])
nac_bin_2_model.load_state_dict(torch.load(NAC_BIN_2_MODEL_FILE)['model'])
nac_bin_3_model.load_state_dict(torch.load(NAC_BIN_3_MODEL_FILE)['model'])
nac_bin_4_model.load_state_dict(torch.load(NAC_BIN_4_MODEL_FILE)['model'])
nac_bin_1_model.eval()
nac_bin_2_model.eval()
nac_bin_3_model.eval()
nac_bin_4_model.eval()
print('regression models initialized')

nac_model_map = {
    1: {
        'model': nac_bin_1_model,
        'scale': NAC_BIN_1_STD,
        'shift' : NAC_BIN_1_MU
        },
    2: {
        'model': nac_bin_2_model,
        'scale': NAC_BIN_2_STD,
        'shift' : NAC_BIN_2_MU
        },
    3: {
        'model': nac_bin_3_model,
        'scale': NAC_BIN_3_STD,
        'shift' : NAC_BIN_3_MU
        },
    4: {
        'model': nac_bin_4_model,
        'scale': NAC_BIN_4_STD,
        'shift' : NAC_BIN_4_MU
        },
}
nac_deployed = DeployNAC(
    mc_model=mc_model,
    nac_model_map=nac_model_map,
    nbins=NBINS
)
print('nac models deployed')

tp_discarded = 0
tn_discarded = 0
fp_discarded = 0
fn_discarded = 0
total_discarded = 0
non_discarded = 0
correct = 0
for structure in dataloader:
    bin_, pred = nac_deployed(structure)
    if bin_ == structure['bin']:
        correct += 1
    print(f"target: {structure['bin']}")
    if structure['bin'] == 0 or structure['bin'] == NBINS - 1:
        if bin_ == structure['bin']:
            tp_discarded += 1
        else:
            fn_discarded += 1
        non_discarded += 1
    else: 
        if bin_ == structure['bin']:
            tn_discarded += 1
        else:
            fp_discarded += 1
        total_discarded += 1
print(f"""True positive: {tp_discarded}
True negative: {tn_discarded}
False positive: {fp_discarded}
False negative: {fn_discarded}
Total discarded: {total_discarded}
Not discarded: {non_discarded}
Accuracy: {round(correct / 696, 2)}
""")
    # target = structure['nacs']
    # print(f'pred: {pred}\ntarget: {target}')













