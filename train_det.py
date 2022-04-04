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
from yolodet.utils.general import increment_path,set_logging
from yolodet.utils.torch_utils import select_device
from yolodet.api.train import train
from yolodet.api.train_two_stage import train_two_stage
from yolodet.api.train_transformer import train_transformer
from yolodet.api.train_distiller import train_distiller
from yolodet.api.train_multi_head import train_multi_head
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c', type=str, default='configs/objectdet/test_multi_head.yaml', help='config file path')
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


    #.py文件的解析
    # assert os.path.exists(opt.config),"{opt.config} config file not exist!"
    # temp = opt.config.replace('/','.').split('.py')[0]
    # exec("from {} import train_cfg".format(temp))
    # exec("from {} import hyp".format(temp))
    # opt.train_cfg = train_cfg
    # opt.hyp = hyp
    # opt.data = eval(train_cfg["data"])

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
    two_stage_enabel = opt.two_stage.get('two_stage_enabel',0)
    transformer_enable = opt.transformer.get('transformer_enabl',0)
    distill_enable  = opt.distiller.get('distill_enable',0)
    multi_head_enable = opt.multi_head.get('multi_head_enable',0)

    if distill_enable:
        save_dir = os.path.join(save_dir,'runs_det',os.path.splitext(os.path.basename(opt.config))[0], opt.train_cfg['project'] + "_" + os.path.splitext(opt.train_cfg['teacher_model'])[0] + '_' + os.path.splitext(opt.train_cfg['student_model'])[0] + '_' + opt.train_cfg['version'] + '_' + opt.train_cfg['name'])
    else:
        save_dir = os.path.join(save_dir,'runs_det',os.path.splitext(os.path.basename(opt.config))[0], opt.train_cfg['project'] + "_" + os.path.splitext(opt.train_cfg['model'])[0] + '_' + opt.train_cfg['version'] + '_' + opt.train_cfg['name'])
    save_dir = increment_path(Path(save_dir),exist_ok=0)  # increment run
    #save_dir = os.path.join(save_dir, 'runs_det', os.path.splitext(os.path.basename(args.cfg))[0], args.base['project'] + '_' + os.path.splitext(args.base['model'])[0] + '_' + args.base['loss_type'] + '_' + args.base['name'])
    #save_dir = increment_path(Path(save_dir), exist_ok=0)
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
    # # Resume
    # if opt.train_cfg['resume']:  # resume an interrupted run
    #     ckpt = opt.train_cfg['resume'] if isinstance(opt.train_cfg['resume'], str) else get_latest_run()  # specified or most recent path
    #     assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
    #     with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
    #         opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
    #     opt.cfg, opt.weights, opt.resume = '', ckpt, True
    #     logger.info('Resuming training from %s' % ckpt)
    # else:
    #     # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    #     #opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    #     assert len(opt.train_cfg['config']) or len(opt.train_cfg['weights']), 'either --cfg or --weights must be specified'
    #     opt.img_size = [opt.train_cfg['hight'],opt.train_cfg['width']]
    #     opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    #     opt.name = 'evolve' if opt.train_cfg['evolve'] else opt.train_cfg['name']
        #opt.save_dir = increment_path(Path(opt.train_cfg['project']) / opt.train_cfg['name'], exist_ok=opt.train_cfg['exist_ok'] | opt.train_cfg['evolve'])  # increment run


    # Train
    #logger.info(opt)
    if not opt.train_cfg['evolve']:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            if 'JinnTrainLog' in os.environ and 'dataSets' in os.environ:
                jinn_log = os.path.dirname(os.environ['dataSets']) + '/logs/'
                logger.info(f'jinn_log: {jinn_log}')
                os.makedirs(jinn_log, exist_ok=True)
                tb_writer = SummaryWriter(log_dir=jinn_log)   # jinn
                logger.info(f'Start Tensorboard with \"tensorboard --logdir {jinn_log}\", view at http://localhost:6006/')
            else:
                tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
                logger.info(f'Start Tensorboard with \"tensorboard --logdir {opt.save_dir}\", view at http://localhost:6006/')
        if two_stage_enabel:
            train_two_stage(opt, device, tb_writer)
        elif transformer_enable:
            train_transformer(opt, device, tb_writer)
        elif multi_head_enable:
            train_multi_head(opt, device, tb_writer)
        elif distill_enable:
            train_distiller(opt, device, tb_writer)
        else:
            train(opt, device, tb_writer)
            

    # # Evolve hyperparameters (optional)
    # else:
    #     # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
    #     meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
    #             'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
    #             'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
    #             'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
    #             'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
    #             'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
    #             'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
    #             'box': (1, 0.02, 0.2),  # box loss gain
    #             'cls': (1, 0.2, 4.0),  # cls loss gain
    #             'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
    #             'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
    #             'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
    #             'iou_t': (0, 0.1, 0.7),  # IoU training threshold
    #             'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
    #             'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
    #             'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
    #             'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
    #             'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
    #             'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
    #             'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
    #             'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
    #             'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
    #             'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
    #             'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
    #             'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
    #             'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
    #             'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
    #             'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

    #     assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
    #     opt.notest, opt.nosave = True, True  # only test/save final epoch
    #     # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
    #     yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
    #     if opt.bucket:
    #         os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

    #     for _ in range(300):  # generations to evolve
    #         if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
    #             # Select parent(s)
    #             parent = 'single'  # parent selection method: 'single' or 'weighted'
    #             x = np.loadtxt('evolve.txt', ndmin=2)
    #             n = min(5, len(x))  # number of previous results to consider
    #             x = x[np.argsort(-fitness(x))][:n]  # top n mutations
    #             w = fitness(x) - fitness(x).min()  # weights
    #             if parent == 'single' or len(x) == 1:
    #                 # x = x[random.randint(0, n - 1)]  # random selection
    #                 x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
    #             elif parent == 'weighted':
    #                 x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

    #             # Mutate
    #             mp, s = 0.8, 0.2  # mutation probability, sigma
    #             npr = np.random
    #             npr.seed(int(time.time()))
    #             g = np.array([x[0] for x in meta.values()])  # gains 0-1
    #             ng = len(meta)
    #             v = np.ones(ng)
    #             while all(v == 1):  # mutate until a change occurs (prevent duplicates)
    #                 v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
    #             for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
    #                 hyp[k] = float(x[i + 7] * v[i])  # mutate

    #         # Constrain to limits
    #         for k, v in meta.items():
    #             hyp[k] = max(hyp[k], v[1])  # lower limit
    #             hyp[k] = min(hyp[k], v[2])  # upper limit
    #             hyp[k] = round(hyp[k], 5)  # significant digits

    #         # Train mutation
    #         results = train(hyp.copy(), opt, device, wandb=wandb)

    #         # Write mutation results
    #         print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

    #     # Plot results
    #     plot_evolution(yaml_file)
    #     print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
    #           f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}