from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.batch_size = 400
_CN.sum_freq = 100
_CN.val_freq = 50000
_CN.critical_params = []
_CN.mixed_precision = False

_CN.training_data_dir = "~/datasets/training_data_count_3model.pkl"
_CN.test_data_dir = "~/datasets/training_data_count_3model.pkl"

_CN.restore_ckpt = "./logs/resd6h512i10w_update1/base/hidden_dim[512]spn_input_dims[[3, 3, 3]]useTree[False]depth[6]use_norm[False](02_09_11_36)/final.pth"
#_CN.restore_ckpt = None

_CN.model = 'base'
#_CN.spn_input_dims = [778, 1368, 760, 1341, 784, 1374]
#_CN.spn_input_dims = [782, 782, 782]
_CN.spn_input_dims = [3, 3, 3]
#_CN.spn_input_dims = [388, 388, 379, 379, 391, 391]

##########################################
# base network

_CN.base = CN()
#_CN.base.spn_input_dims = [778, 1368,760, 1341, 784, 1374]
#_CN.base.spn_input_dims = [388, 379, 391]
_CN.base.spn_input_dims = [3, 3, 3]
#_CN.base.spn_input_dims = [388, 388, 379, 379, 391, 391]
_CN.base.hidden_dim = 512
_CN.base.use_norm = False
_CN.base.depth = 6
_CN.base.device = 'cuda'
_CN.base.crop = 3
# tree convolution
_CN.base.useTree = False
_CN.base.useMul = False
_CN.base.count = False
_CN.base.critical_params = ["hidden_dim", "spn_input_dims", "useTree", "depth", "use_norm"]
_CN.base.zero_debug = False
_CN.base.res_mlp = True
#_CN.base.dataset_batch = "/home/qym/datasets/testSetCount_norm_1000-mean_crop3.pkl"

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'

_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 25e-4
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1
_CN.trainer.num_steps = 100000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
