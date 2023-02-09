from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.batch_size = 400
_CN.sum_freq = 100
_CN.val_freq = 50000
_CN.critical_params = []
_CN.mixed_precision = False

#_CN.restore_ckpt = "logs/bs40/base/hidden_dim[256]spn_input_dims[[776, 758, 782]](06_05_22_17)/final.pth"
#_CN.restore_ckpt = "logs/bs20/base/hidden_dim[256]spn_input_dims[[776, 758, 782]](06_05_20_38)/final.pth"
#_CN.restore_ckpt = "logs/training/base/hidden_dim[256]spn_input_dims[[776, 758, 782]](06_06_10_41)/final.pth"
#_CN.restore_ckpt = "/home/qym/spn_ensemble/logs/resd6h512i50w_crop3/base/hidden_dim[512]spn_input_dims[[3, 3, 3]]useTree[False]depth[6]use_norm[False](07_16_07_15)/final.pth"
#_CN.restore_ckpt = "/home/qym/spn_ensemble/logs/resd9h512i100w_6model/base/hidden_dim[512]spn_input_dims[[3, 3, 3, 3, 3, 3]]useTree[False]depth[9]use_norm[False](09_14_07_20)/final.pth"
_CN.restore_ckpt = "/home/qym/spn_ensemble/logs/resd6h512i10w_update1/base/hidden_dim[512]spn_input_dims[[3, 3, 3]]useTree[False]depth[6]use_norm[False](02_09_11_36)/final.pth"
#_CN.restore_ckpt = None

#_CN.restore_ckpt = "/home/qym/spn_ensemble/logs/d7i10zero_debug/base/hidden_dim[256]spn_input_dims[[388, 379, 391]]useTree[False]depth[7]use_norm[False](07_06_04_58)/final.pth"
#_CN.restore_ckpt = "/home/qym/spn_ensemble/logs/d5i10zero_debug/base/hidden_dim[256]spn_input_dims[[388, 379, 391]]useTree[False]depth[5]use_norm[False](07_03_13_02)/final.pth"
#_CN.restore_ckpt ="/home/qym/spn_ensemble/logs/d9i10useMul/base/hidden_dim[256]spn_input_dims[[782, 782, 782]]useTree[False]depth[9]use_norm[False](06_25_04_37)/final.pth"
#_CN.restore_ckpt = "/home/qym/spn_ensemble/logs/countd9i10useMul/base/hidden_dim[256]spn_input_dims[[388, 379, 391]]useTree[False]depth[9]use_norm[False](06_27_02_54)/final.pth"
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

"""
###########################################
# norm network
_CN.norm = CN()
_CN.norm.spn_input_dims = [388, 388, 379, 379, 391, 391]
_CN.norm.hidden_dim = 256
_CN.norm.use_norm = False
_CN.norm.depth = 5
_CN.norm.device = 'cuda'
_CN.norm.useTree = False
_CN.norm.useMul = False
_CN.norm.count = False
_CN.norm.critical_params = ["hidden_dim", "spn_input_dims", "useTree", "depth", "use_norm"]
"""

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'

_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 25e-6
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1
_CN.trainer.num_steps = 100000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
