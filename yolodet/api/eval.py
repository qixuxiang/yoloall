import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# from config.data import *
from yolodet.utils.general import increment_path,set_logging,check_img_size,labels_to_image_weights
from yolodet.utils.torch_utils import select_device
from yolodet.api.train import create_model,create_optimizer
from yolodet.api.test import test
from yolodet.api.train_two_stage import train_two_stage
from yolodet.api.train_transformer import train_transformer
from yolodet.api.train_distiller import train_distiller
logger = logging.getLogger(__name__)

loss_version_dic = {} #loss_version 与损失超参的对应关系
def show_opts(args):
    logger.info('\nargs params:')
    for arg in vars(args):
        if (arg in args.version_info.values() and arg != args.version_info[args.train_cfg['version']]) or arg == 'version_info':
            continue
        attr = getattr(args, arg)
        if isinstance(attr, dict):
            logger.info('- {}:'.format(arg))
            for key in attr:
                logger.info('     # {:20}: {}'.format(key, attr[key]))
        else:
            logger.info('- {:25}: {}'.format(arg, attr))


def eval(opt, device, tb_writer):
    show_opts(opt)
    weights                 = opt.train_cfg['weights_one']
    plots                   = not opt.train_cfg['evolve']  # create plots
    opt.data['version']     = opt.train_cfg['version']
    # System
    batch_size              = opt.batch_size
    total_batch_size        = opt.total_batch_size
    cuda                    = device.type != 'cpu'
    rank                    = opt.global_rank
    # Data
    test_path               = opt.data['val']
    nc                      = opt.data['nc']
    names                   = opt.data['names']
    # Directories
    save_dir                = Path(opt.save_dir)
    results_file            = save_dir / 'results.txt'
    results_file_per_class  = save_dir / 'results_per_class.txt'
    assert weights != None,'weights is None'
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    with open(results_file_per_class,'w') as f:
        f.write(('%20s' + '%12s' * 6) % ('class', 'seen', 'nt', 'mp', 'mr', 'map50', 'map') + '\n')
    with open(results_file, 'w') as f:
        f.write('%10s' * 7 % ('mp', 'mr', 'map50', 'map', 'box', 'obj', 'cls') + '\n')
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    # Create model
    model = create_model(opt,device)[0]
    model.eval()
    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples img_size-> [h,w]

    # Trainloader
    if 'mmdet' in opt.train_cfg['version']:
        logger.info('from yolodet.dataset.datasets_mmdet import create_dataloader')
        from yolodet.dataset.datasets_mmdet import create_dataloader_mmdet as create_dataloader
        testloader = create_dataloader(test_path, imgsz_test, batch_size, model.stride, opt, rank=-1, pad=0)[0] 
    else:
        logger.info('from yolodet.dataset.datasets import create_dataloader')
        from yolodet.dataset.datasets import create_dataloader_yolo as create_dataloader
        testloader = create_dataloader(test_path, imgsz_test, batch_size, model.stride, opt, rank=-1, pad=0)[0] 
    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.names = names
    logger.info('\nImage sizes %g,%g test\n'
                'Using %g dataloader workers\nLogging results to %s\n' % (imgsz_test[1],imgsz_test[0], testloader.num_workers, save_dir))
    logger.info('>> '+ time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    results, maps, times, tables = test(opt,
                                    batch_size=total_batch_size,
                                    imgsz=imgsz_test,
                                    model=model,
                                    single_cls=opt.train_cfg.get('single_cls',False),
                                    dataloader=testloader,
                                    save_dir=save_dir,
                                    plots=plots,
                                    log_imgs= 0)
    #results = list(results).insert(0,f'{imgsz_test[1]}*{imgsz_test[0]}')
    # Write
    with open(results_file, 'a') as f:
        f.write('%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                # Write
    with open(results_file_per_class, 'a') as f:
        #f.write(('%20s' + '%12s' * 6) % ('class', 'seen', 'nt', 'mp', 'mr', 'map50', 'map') + '\n')
        [f.write(table + '\n') for table in tables]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c', type=str, default='config/objectdet/test.yaml', help='config file path')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    #opt = parser.parse_args()
    parser_, remaining = parser.parse_known_args()

    if parser_.config:
        with open(parser_.config,'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    opt = parser.parse_args()
    assert opt.config, f"config is null"

    if 'name_conf_cls_map' in opt.data:
        nc = opt.data['nc']
        name_conf_cls_map = opt.data['name_conf_cls_map']
        assert len(name_conf_cls_map) == nc, "name_conf_cls_map is wrong"
        opt.data['names'] = [name_conf_cls_map[i][0] for i in range(nc)]
        opt.data['conf_thres'] = [name_conf_cls_map[i][1] for i in range(nc)]
        opt.data['cls_map'] = {i:name_conf_cls_map[i][2] for i in range(nc)}

    # Set DDP variables
    opt.local_rank = parser_.local_rank
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    gpu_num = int(os.environ['GPU_NUM']) if 'GPU_NUM' in os.environ else 1
    opt.batch_size = int(opt.train_cfg['batch_size']) #int(eval(opt.train_cfg['batch_size']))
    opt.total_batch_size = opt.batch_size * gpu_num

    save_dir = ''
    if 'JinnTrainResult' in os.environ:
        if 'jinn_path' in os.environ['JinnTrainResult']:
            save_dir = os.environ['JinnTrainResult']
    #切换不同的训练模式
    two_stage_enabel = opt.two_stage['two_stage_enabel']
    transformer_enable = opt.transformer['transformer_enabl']
    distill_enable  = opt.distiller['distill_enable']
    if distill_enable:
        save_dir = os.path.join(save_dir,'runs_det',os.path.splitext(os.path.basename(opt.config))[0]+'_eval', opt.train_cfg['project'] + "_" + os.path.splitext(opt.train_cfg['teacher_model'])[0] + '_' + os.path.splitext(opt.train_cfg['student_model'])[0] + '_' + opt.train_cfg['version'] + '_' + opt.train_cfg['name'])
    else:
        save_dir = os.path.join(save_dir,'runs_det',os.path.splitext(os.path.basename(opt.config))[0]+'_eval', opt.train_cfg['project'] + "_" + os.path.splitext(opt.train_cfg['model'])[0] + '_' + opt.train_cfg['version'] + '_' + opt.train_cfg['name'])
    save_dir = increment_path(Path(save_dir),exist_ok=0)  # increment run
    opt.save_dir = save_dir
    logger.info(f'save_dir {opt.save_dir}\n')

    # DDP mode
    device = select_device(opt.device, batch_size=opt.total_batch_size)

    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.total_batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    else:
        opt.batch_size = opt.total_batch_size

    opt.img_size = [opt.train_cfg['height'],opt.train_cfg['width']]
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    # eval
    tb_writer = None
    eval(opt, device, tb_writer)