##########################################################################
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw 
##########################################################################
import os, sys
import shutil
from pathlib import Path
import argparse
import datetime
import random
import time

import torch
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# Custom Modules
from engine import *
from datasets import build_dataset
from models import build_model
import util.misc as utils
from util.logger import setup_logger, AvgerageMeter, EvaluateMeter

# In[0]: Parser
def parse_args():
    from config import cfg, merge_from_file, merge_from_list
    parser = argparse.ArgumentParser('APGCC')
    parser.add_argument('-c', '--config_file', type=str, default="", help='the path to the training config')
    parser.add_argument('-t', '--test', action='store_true', default=False, help='Model test')
    parser.add_argument('opts', help='overwriting the training config' 
                        'from commandline', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg = merge_from_file(cfg, args.config_file)
    cfg = merge_from_list(cfg, args.opts)
    cfg.config_file = args.config_file
    cfg.test = args.test
    return cfg

# In[1]: Main
def main():
    # Initial Config and environment
    cfg = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(cfg.GPU_ID)
    device = torch.device('cuda')
    seed = cfg.SEED
    if seed != None:
        g = torch.Generator()
        g.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

    # Function
    if cfg.test:
        test(cfg)
    else:
        train(cfg)

# In[2]: Training Function
def train(cfg):
    # makedir 
    output_dir = cfg.OUTPUT_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(cfg.config_file, output_dir)  # copy config file

    # logging
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('APGCC', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("##############################################################")
    logger.info("# % TRAIN .....  ")
    logger.info("# Config is {%s}" %(cfg.config_file))
    logger.info("# SEED is {%s}" %(cfg.SEED))
    logger.info("# DATASET is {%s}" %(cfg.DATASETS.DATASET))
    logger.info("# DATA_RT is {%s}" %(cfg.DATASETS.DATA_ROOT))
    logger.info("# MODEL.ENCODER is {%s}" %(cfg.MODEL.ENCODER))
    logger.info("# MODEL.DECODER is {%s}" %(cfg.MODEL.DECODER))
    logger.info("# MODEL.CONFIG is {%s}" %(cfg.MODEL.DECODER_kwargs))
    logger.info("# LOSS.WEIGHT is {%s}" %(cfg.MODEL.WEIGHT_DICT))
    logger.info("# AUXILIARY MODE is {%s}" %(cfg.MODEL.AUX_EN))
    logger.info("# RESUME is {%s}" %(cfg.RESUME))
    logger.info("# BATCH is {%d*%d*%d}"%(cfg.SOLVER.BATCH_SIZE, cfg.MODEL.ROW, cfg.MODEL.LINE))
    logger.info("# OUTPUT_DIR is {%s}" %(cfg.OUTPUT_DIR))
    logger.info("##############################################################")
    logger.info('Eval Log %s' % time.strftime("%c"))

    # Define the dataloader
    train_dl, val_dl = build_dataset(cfg=cfg)
    torch.multiprocessing.set_sharing_strategy('file_system')  # avoid limitation of number of open files.

    # Building the Model & Optimizer
    model, criterion = build_model(cfg=cfg, training=True)
    model.cuda()
    criterion.cuda()

    # Build Trainier
    trainer = Trainer(cfg, model, train_dl, val_dl, criterion)
    print("Start training")
    start_time = time.time()
    for epoch in range(trainer.train_epoch, trainer.epochs):
        for batch in trainer.train_dl:
            trainer.step(batch)
            trainer.handle_new_batch()
        trainer.handle_new_epoch()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time [%s]'%(total_time_str))

# In[3]: Testing Function
def test(cfg):
    # makedir
    source_dir = cfg.OUTPUT_DIR
    output_dir = os.path.join(source_dir, "%s_%.2f"%(cfg.DATASETS.DATASET, cfg.TEST.THRESHOLD))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vis_val_path = None
    if cfg.VIS:  
        vis_val_path = os.path.join(output_dir, 'sample_result_for_val/')
        if not os.path.exists(vis_val_path):
            os.makedirs(vis_val_path)

    # logging
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('AGPCC', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("##############################################################")
    logger.info("# % TEST .....  ")
    logger.info("# Config is {%s}" %(cfg.config_file))
    logger.info("# SEED is {%s}" %(cfg.SEED))
    logger.info("# DATASET is {%s}" %(cfg.DATASETS.DATASET))
    logger.info("# DATA_RT is {%s}" %(cfg.DATASETS.DATA_ROOT))
    logger.info("# MODEL.ENCODER is {%s}" %(cfg.MODEL.ENCODER))
    logger.info("# MODEL.DECODER is {%s}" %(cfg.MODEL.DECODER))
    logger.info("# RESUME is {%s}" %(cfg.RESUME))
    logger.info("# BATCH is {%d*%d*%d}"%(cfg.SOLVER.BATCH_SIZE, cfg.MODEL.ROW, cfg.MODEL.LINE))
    logger.info("# OUTPUT_DIR is {%s}" %(cfg.OUTPUT_DIR))
    logger.info("# RESULT_DIR is {%s}" %(output_dir))
    logger.info("##############################################################")
    logger.info('Eval Log %s' % time.strftime("%c"))

    # Define the dataset.
    train_dl, val_dl = build_dataset(cfg=cfg)
    torch.multiprocessing.set_sharing_strategy('file_system')  # avoid limitation of number of open files.

    # Building the Model & Optimizer 
    model = build_model(cfg=cfg, training=False)
    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:%d \n' % n_parameters)

    pretrained_dict = torch.load(os.path.join(source_dir, 'best.pth'), map_location='cpu')
    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)

    # Starting Testing
    print("Start testing")
    t1 = time.time()
    result = evaluate_crowd_counting(model, val_dl, next(model.parameters()).device, cfg.TEST.THRESHOLD, vis_val_path)
    t2 = time.time()
    logger.info('Eval: MAE=%.6f, MSE=%.6f time:%.2fs \n'%(result[0], result[1], t2 - t1))

# %%
if __name__ == '__main__':
    main()
