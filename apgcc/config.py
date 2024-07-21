import os
from easydict import EasyDict as edict
import time
import numpy as np

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = edict()
cfg = _C  # Alias for easy usage

_C.TAG = 'APGCC'
_C.SEED = 1229 # seed
_C.GPU_ID = 0 # gpu_id, the gpu used for training
_C.OUTPUT_DIR = './output/temp/' # output_dir, path where to save, empty for no saving
_C.VIS = False # vis the predict sample

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = edict()
_C.MODEL.ENCODER = 'vgg16_bn' # ['vgg16', 'vgg16_bn'] # backbone, Name of the convolutional backbone to use.
_C.MODEL.ENCODER_kwargs = {"last_pool": False, # last layer downsample, False:feat4(H/16,W/16), True:feat4(H/32,W/32)  
						  } 
						#   "layers": 4,  # select number of bodies.
						#   "fpn": True,  
						#   "multi_grid": True,
						#   "zero_init_residual": True,
						#   "replace_stride_with_dilation": [False, False, False],
						#   "feat_dims":64} 

_C.MODEL.DECODER = 'basic' # ['basic', 'IFA'] # decoder 
_C.MODEL.DECODER_kwargs = { "num_classes": 2,        # output num_classes, default:2 means confindence.
							"inner_planes": 256,     # basic: 256, IFA: 64
							"feat_layers":[3,4],     # control the number of decoder features. [1,2,3,4]
							"pos_dim": 2,            #
							"ultra_pe": False,       # additional position encoding. x -> (x, sin(x), cos(x))
							"learn_pe": False,       # additional position encoding. x -> (trainable variable)
							"unfold": False,         # unfold feat channel, make the feat dim be 3x3 times.
							"local": False,          # enable local patch, 3x3 mapping near by the center point.
							"no_aspp": True,         # final feat encoding add the aspp module.
							"require_grad": True,	
							"out_type": 'Normal',	 # out_type = 'Normal' / 'Conv' / 'Deconv'
							"head_layers":[1024,512,256,256]}  # head layers is n+1, last layers is num_of_proposals  

_C.MODEL.STRIDE = 8 					# the size of anchor map by image.shape/stride, ex: input 128x128, stride=8, anchor_map = 16x16
_C.MODEL.ROW = 2 				# row, row number of anchor points
_C.MODEL.LINE = 2 				# line, line number of anchor points
_C.MODEL.FROZEN_WEIGHTS = None 	# frozen_weights, Path to the pretrained model. If set, only the mask head will be trained.

# mixed to the unity loss kwargs
_C.MODEL.POINT_LOSS_COEF = 0.0002 # point_loss_coef
_C.MODEL.EOS_COEF = 0.5 # eos_coef, Relative classification weight of the no-object class

_C.MODEL.LOSS = ['L2']
_C.MODEL.WEIGHT_DICT = {'loss_ce': 1, 'loss_points': 0., 'loss_aux': 0.} 
_C.MODEL.AUX_EN = False
_C.MODEL.AUX_NUMBER = [1, 1] # the number of pos/neg anchors
_C.MODEL.AUX_RANGE = [1, 4]  # the randomness range of auxiliary anchors
_C.MODEL.AUX_kwargs = {'pos_coef': 1., 'neg_coef': 1., 'pos_loc': 0., 'neg_loc': 0.} 

_C.RESUME = False # resume, resume from checkpoint
_C.RESUME_PATH = '' # keep training weights.

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = edict()
_C.DATASETS.DATASET = 'SHHA' # dataset_file
_C.DATASETS.DATA_ROOT = './dataset_path/' # data_root, path where the dataset is

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = edict()
_C.DATALOADER.AUGUMENTATION = ['Normalize', 'Crop', 'Flip']
_C.DATALOADER.CROP_SIZE = 128 		# radnom crip size for training
_C.DATALOADER.CROP_NUMBER = 4 		# the number of training sample
_C.DATALOADER.UPPER_BOUNDER = -1 	# the upper bounder of size
_C.DATALOADER.NUM_WORKERS = 8 		# num_workers

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = edict()
_C.SOLVER.BATCH_SIZE = 8 # batch_size\
_C.SOLVER.START_EPOCH = 0 # start_epoch
_C.SOLVER.EPOCHS = 3500  # epochs
_C.SOLVER.LR = 1e-4    # lr
_C.SOLVER.LR_BACKBONE = 1e-5   # lr_backbone
_C.SOLVER.WEIGHT_DECAY = 1e-4  # weight_decay
_C.SOLVER.LR_DROP = 3500 # lr_drop
_C.SOLVER.CLIP_MAX_NORM = 0.1 # clip_max_norm, gradient clipping max norm

_C.SOLVER.EVAL_FREQ = 5 # eval_freq, frequency of evaluation, default setting is evaluating in every 5 epoch
_C.SOLVER.LOG_FREQ = 1 # log_freq, frequency of recording training.
# ---------------------------------------------------------------------------- #
# Matcher
# ---------------------------------------------------------------------------- #
_C.MATCHER = edict()
_C.MATCHER.SET_COST_CLASS = 1. # set_cost_class, Class coefficient in the matching cost
_C.MATCHER.SET_COST_POINT = 0.05 # set_cost_point, L1 point coefficient in the matching cost

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = edict()
_C.TEST.THRESHOLD = 0.5
_C.TEST.WEIGHT = ""

################ modules ################
def cfg_merge_a2b(a, b):
    if type(a) is not edict and type(a) is not dict:
        raise KeyError('a is not a edict.')
    
    for k, v in a.items():
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(b[k]) # original type
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif not isinstance(v, dict) and not isinstance(b[k], type(None)):
                raise ValueError(('Type mismatch ({} vs. {})'
                                'for config key: {}').format(type(b[k]), type(v), k))
        if type(v) is edict or type(v) is dict:
            try:
                cfg_merge_a2b(a[k], b[k])
            except:
                print(('Error under config key: {}').format(k))
                raise
        else:
            if v == 'None':
                b[k] = None
            else:
                b[k] = v

    return b

def cfg_from_file(filename):
    import yaml
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return data

def cfg_from_list(args_opts):
	from ast import literal_eval
	print(args_opts)
	if len(args_opts) == 0:
		return None
	assert len(args_opts)%2 == 0
	for k, v in zip(args_opts[0::2], args_opts[1::2]):
		key_list = k.split('.')
		d = _C
		for subkey in key_list[:-1]:
			assert subkey in d
			d = d[subkey]
		subkey = key_list[-1]
		assert subkey in d
		try:
			value = literal_eval(v)
		except:
			value = v
		assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(type(value), type(d[subkey]))
		d[subkey] = value
	return d

def merge_from_file(cfg, filename):
    file_cfg = cfg_from_file(filename)
    cfg = cfg_merge_a2b(file_cfg, cfg)
    return cfg

def merge_from_list(cfg, args):
	args_cfg = cfg_from_list(args)
	if args_cfg == None:
		return cfg
	else:
		cfg = cfg_merge_a2b(args_cfg, cfg)
		return cfg