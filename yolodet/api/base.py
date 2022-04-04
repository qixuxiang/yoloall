import logging
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from yolodet.utils.torch_utils import torch_distributed_zero_first, intersect_dicts
from yolodet.utils.google_utils import attempt_download
import torch.nn as nn
import math
logger = logging.getLogger(__name__)
import re
def patterns_match(patterns, string):
    """
    Args:
        patterns: "prefix" or "r:regex"
        string: string
    """

    def match(pattern, string):
        if pattern.startswith('r:'):  # regex
            pattern = pattern[2:]
            if len(re.findall(pattern, string)) > 0:
                return True
        elif string.startswith(pattern):  # prefix
            return True
        return False

    if not isinstance(patterns, (tuple, list)):
        patterns = [patterns]

    if string == "":
        return False

    for pattern in patterns:
        if match(pattern, string):
            return True

    return False


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

def create_model(opt,device):
    loss_version        = opt.train_cfg['version']
    model_name          = opt.train_cfg['model']
    weights             = opt.train_cfg['weights_one']
    rank                = opt.global_rank
    nc                  = opt.data['nc']
    #some detector need exetor parmter like 'fcos'
    # if loss_version in ['yolo-fcos-mmdet','yolo-gfl-mmdet']:
    #     opt.train_cfg[loss_version] = eval('opt.'+ opt.version_info[loss_version])
    # else:
    #     pass
    #creat model follow model_name
    if loss_version == 'darknet':
        from yolodet.models.darknet_model import Darknet as Model
        logger.info('Loading model from yolodet.models.darknet_model import Darknet')
        #model = Model(opt.train_cfg['cfg'],(opt.train_cfg['hight'],opt.train_cfg['width'])).to(device)
    else:
        if model_name.lower().endswith('.yaml'):
            from yolodet.models.yolo_yaml import Model
            logger.info('Loading model from yolodet.models.yolo_yaml import Model')
            #model = Model(opt.train_cfg, ch=3, nc=nc).to(device)  # create
        elif model_name.lower().endswith('.cfg'):
            from yolodet.models.yolo_darknet import Model
            logger.info('Loading model from yolodet.models.yolo_darknet import Model')

        else:# opt.train_cfg['cfg'].lower().endswith('.py'):
            if hasattr(opt, 'multi_head') and opt.multi_head['multi_head_enable'] and opt.multi_head['num'] > 1:
                from yolodet.models.yolo_multi_head import Model
                logger.info('Loading model from yolodet.models.yolo_multi_head import Model')
            else: 
                from yolodet.models.yolo_py import Model
                logger.info('Loading model from yolodet.models.yolo_py import Model')
            #model = Model(opt.train_cfg, ch=3, nc=nc).to(device)  # create
    
    #Model
    pretrained = weights and (weights.endswith('.pt') or weights.endswith('.pth'))
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint
        if 'model' not in ckpt:
            ckpt['model'] = ckpt.pop('model_state_dict')
        model = Model(opt, ch=3, nc=nc)
        exclude = ['anchor'] if model else []   # exclude keys

        try:
            state_dict = ckpt['model'].float().state_dict() # to FP32
        except:
            state_dict = ckpt['model']
        
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        model = model.to(device)
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        ckpt = None
        model = Model(opt, ch=3, nc=nc).to(device)  # create

    #添加固定BN不优化 其他层进行优化的方法：
    def freeze_layer(model, mode=True):
        self.training = mode
        for name, m in model.named_modules():
            if name == "":  # self
                continue
            if patterns_match(self.freeze_patterns, name):
                m.eval()
            else:
                m.train(mode)
        return self


    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    return model,ckpt


def create_optimizer(opt, model, epochs, ckpt, results_file):

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.LayerNorm):# 这里nn.LayerNorm
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.hyp['adam']:
        optimizer = optim.Adam(pg0, lr=opt.hyp['lr0'], betas=(opt.hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=opt.hyp['lr0'], momentum=opt.hyp['momentum'], nesterov=True)
    if 'L2' in opt.hyp['norm']:
        optimizer.add_param_group({'params': pg1, 'weight_decay': opt.hyp['weight_decay']})  # add pg1 with weight_decay
    else:
        optimizer.add_param_group({'params': pg1})  # add pg2 (biases)
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.hyp['lr_cos']:
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - opt.hyp['lrf']) + opt.hyp['lrf']  # cosine
    else:
        def lf(x):
            factor = 1.0
            for step in opt.hyp['lr_step']:
                if (x+1) >= step:
                    factor *= 0.1
            return factor
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    weights = opt.train_cfg['weights_one']
    if weights and (weights.endswith('.pth') or weights.endswith('.pt')):
        #epoch
        start_epoch = ckpt['epoch'] + 1
        #optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        
        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt
        del ckpt

    return start_epoch, best_fitness,  optimizer, lf, scheduler



def choose_dataloader_and_test(opt):
    
    # mmdet multi_head Trainloader
    if hasattr(opt, 'multi_head') and opt.multi_head['multi_head_enable']:
        logger.info('from yolodet.dataset.datasets import create_dataloader')
        from yolodet.dataset.datasets_mmdet_multi_head import create_dataloader_mmdet_multi_head as create_dataloader
    #mmdet Trainloader
    elif 'mmdet' in opt.train_cfg['version']:
        logger.info('from yolodet.dataset.datasets_mmdet import create_dataloader')
        from yolodet.dataset.datasets_mmdet import create_dataloader_mmdet as create_dataloader
    #yolov3、v4、v5 Trainloader
    else:
        logger.info('from yolodet.dataset.datasets import create_dataloader')
        from yolodet.dataset.datasets import create_dataloader_yolo as create_dataloader

    
    if hasattr(opt, 'two_stage') and opt.two_stage['two_stage_enabel']:
        logger.info('from yolodet.api.test_two_stage import test')
        from yolodet.api.test_two_stage import test

    elif hasattr(opt, 'transformer') and opt.transformer['transformer_enabl']:
        logger.info('from yolodet.api.test import test')
        from yolodet.api.test_multi_head import test

    elif hasattr(opt, 'distiller') and opt.distiller['distill_enable']:
        logger.info('from yolodet.api.test_distiller import test')
        from yolodet.api.test_distiller import test

    elif hasattr(opt, 'multi_head') and opt.multi_head['multi_head_enable']:
        logger.info('from yolodet.api.test_multi_head import test')
        from yolodet.api.test_multi_head import test
    
    else:
        logger.info('from yolodet.api.test import test')
        from yolodet.api.test import test

    return create_dataloader, test