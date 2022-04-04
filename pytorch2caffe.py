import os
import yaml
import torch
import argparse
import logging

from yolodet.utils.general import check_file
from yolodet.utils.torch_utils import intersect_dicts

from yolodet.models.experimental import attempt_load
from convert.torch2caffe.pytoch_to_caffe import trans_net
from convert.torch2caffe.pytoch_to_caffe import trans_nets
logger = logging.getLogger(__name__)
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task','-t', type=str, default='oc', help='od, oc')
    parser.add_argument('--cfg','-c',type=str,default='configs/classify/test.yaml')#['configs/objectdet/test.yaml','configs/objectdet/test.yaml']
    parser.add_argument('--merge_bn',type=int,default=1)
    parser.add_argument('--weight', '-w', type=str, default='null.pth', help='weight path')
    parser.add_argument('--simple_name', type=int, default=1)
    parser.add_argument('--pre', type=str, default='detect')
    parser.add_argument('--draw', type=int, default=0)
    parser.add_argument('--has_two_stage',type=int, default=0)

    # cfg2opt
    parser.add_argument("--nnx_name",type=str, default='test', help='od oc')
    parser.add_argument("--image_name", type=str, default='', help='od oc')
    parser.add_argument("--is_profile", type=int, default=None, help='od oc')
    parser.add_argument("--rgb", type=int,default=None, help='od oc')
    parser.add_argument("--fixed_scale", type=int,default=None, help='od oc')
    parser.add_argument("--mean", type=list,default=None, help='od oc')
    parser.add_argument("--norm", type=list,default=None, help='od oc')

    parser.add_argument("--ot_feature", type=int,default=None, help='od oc')
    parser.add_argument("--multi_objects", type=int,default=None, help='od oc')
    parser.add_argument("--ivs_special_param", type=int,default=None, help='od oc')
    parser.add_argument("--struct_special_param", type=int,default=None, help='od oc')

    parser.add_argument("--platform", '-p', type=int,default=None, help='od oc')
    parser.add_argument("--extra", '-e', type=int,default=None, help='od oc')

    parser_,remaining = parser.parse_known_args()

    parser_cfg = [parser_.cfg] if isinstance(parser_.cfg,str) else parser_.cfg

    _args = []
    for _yaml in parser_cfg:
        if _yaml:
            with open(_yaml, 'r')as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
        args = parser.parse_args()
        assert args.cfg, f"config in null or weight is null"
        if args.task == 'od':
            if 'name_conf_cls_map' in args.data:
                nc = args.data['nc']
                name_conf_cls_map = args.data['name_conf_cls_map']
                assert len(name_conf_cls_map) == nc, "name_conf_cls_map is wrong"
                args.data['name'] = [name_conf_cls_map[i][0] for i in range(nc)]
                args.data['conf_thres'] = [name_conf_cls_map[i][1] for i in range(nc)]
                args.data['cls_map'] = [name_conf_cls_map[i][2] for i in range(nc)]
        _args.append(args)
    return _args

def trans_od(args_list):
    #create torch model
    model_list = []
    for args in args_list:
        version     = args.train_cfg['version']
        model       = args.train_cfg['model']
        anchors     = args.train_cfg['anchors']
        height      = args.train_cfg['height']
        width       = args.train_cfg['width']
        nc          = args.data['nc'] if not isinstance(args.data, list) else None
        input_size = (1,3,height,width)
        print(f">> input_size:{input_size}\n")
        
        #model set
        print('')
        if model.endswith('.yaml'):
            from yolodet.models.yolo_yaml import Model
            logger.info('Loading model from yolodet.models.yolo_yaml import Model')
        elif 'vid' in model:#视频检测模型
            pass
        else:# opt.train_cfg['cfg'].lower().endswith('.py'):
            from yolodet.models.yolo_py import Model
            logger.info('Loading model from yolodet.models.yolo_py import Model')

        if args.has_two_stage:
            pass
        else:
            if os.path.isfile(args.weight):
                ckpt = torch.load(args.weight, map_location='cpu')
                if 'model' not in ckpt:
                    ckpt['model'] = ckpt.pop('model_state_dict')
                model = Model(args, ch=3, nc=nc)
                exclude = ['anchor'] if args.train_cfg['model'] else []

                try:
                    state_dict = ckpt['model'].float().state_dict()
                except:
                    state_dict = ckpt['model']
                state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
                model.load_state_dict(state_dict, strict=False)  # load
                logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), args.weights))  # report
            else:
                model = Model(args, ch=3, nc=nc)

            if 'vid' not in args.train_cfg['version']:
                if args.merge_bn:
                    model.fuse()
                if hasattr(args, 'multi_head') and args.multi_head['multi_head_enable']:
                    pass
                else:
                    try:
                        model.model[-1].export = True
                    except:
                        model.model.detect.export = True
            else:
                model.export()
        
        model.eval()
        model_list.append(model)
        print()
    if len(model_list) == 1:
        prototxt,caffemodel, output_blobs = trans_net(model_list[0], input_size, ckpt=args.weight, merge_bn=args.merge_bn,
                                                    out_blob_pre=args.pre, draw=args.draw)
    else:
        prototxt,caffemodel, output_blobs = trans_nets(model_list, input_size, ckpt=args.weight, merge_bn=args.merge_bn,
                                                    out_blob_pre=args.pre, draw=args.draw)

    #to nnx json

    if args.platform:
        pass
    else:
        prototxt_nnx = f'./caffemodel/{args.nnx_name}.prototxt'
        caffemodel_nnx = f'./caffemodel/{args.nnx_name}.caffemodel'

    os.system('mv {} {}'.format(prototxt, prototxt_nnx))
    os.system('mv {} {}'.format(caffemodel, caffemodel_nnx)) 
    

def trans_oc(args):
   #create torch model
    model       = args.model
    model_cfg   = args.model_cfg
    num_classes = args.num_classes
    height      = args.input_size[1]
    width       = args.input_size[2]
    input_size = (1,3,height,width)
    print(f">> input_size:{input_size}\n")
    kwargs = {}
    kwargs['model_name'] = model
    kwargs['model_cfg'] = model_cfg
    kwargs['num_classes'] = num_classes

    #model set
    print('')
    from classify.timm.models.torch_model import Model
    logger.info('Loading model classify.timm.models.torch_model import Model')
    model = Model(kwargs) #如果在torch时进行bn融合就进行model.fuse()

    model.eval()
    print()

    prototxt, caffemodel, output_blobs = trans_net(model, input_size, ckpt=args.weight, merge_bn=args.merge_bn,
                                                  out_blob_pre='', draw=args.draw)
    
    #to nnx json

    if args.platform:
        pass
    else:
        prototxt_nnx = f'./caffemodel/{args.nnx_name}.prototxt'
        caffemodel_nnx = f'./caffemodel/{args.nnx_name}.caffemodel'

    os.system('mv {} {}'.format(prototxt, prototxt_nnx))
    os.system('mv {} {}'.format(caffemodel, caffemodel_nnx)) 


def cfg_2_opt(args, weight_name=None):
    pass

if __name__ == "__main__":
    args = get_args()
    if args[0].task == "oc": #oc仅支持但模型转换
        trans_oc(args[0])
    else:
        trans_od(args)






