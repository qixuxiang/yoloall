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
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from yolodet.api import test  # import test.py to get mAP after each epoch
from yolodet.models.experimental import attempt_load
from yolodet.dataset.autoanchor import check_anchors
# from yolodet.dataset.datasets import create_dataloader
from yolodet.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    print_mutation, set_logging, xyxy2xywh

from yolodet.loss.loss_base import ComputeLoss
from yolodet.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from yolodet.utils.torch_utils import ModelEMA
import pdb
from . import show_opts, create_model, create_optimizer, choose_dataloader_and_test
logger = logging.getLogger(__name__)


def train(opt, device, tb_writer=None):
    # Show parmter
    show_opts(opt)

    # Train_cfg
    epochs                  = opt.train_cfg['epochs']
    weights                 = opt.train_cfg['weights_one']
    loss_version            = opt.train_cfg['version']
    plots                   = not opt.train_cfg['evolve']  # create plots
    opt.data['version']     = opt.train_cfg['version']
    nbs                     = opt.train_cfg['nbs']
    nl                      = len(opt.train_cfg['anchors'])
    # System
    batch_size              = opt.batch_size
    total_batch_size        = opt.total_batch_size
    cuda                    = device.type != 'cpu'
    rank                    = opt.global_rank
    # Data
    train_path              = opt.data['train']
    test_path               = opt.data['val']
    nc                      = opt.data['nc']
    names                   = opt.data['names']
    # Directories
    save_dir                = Path(opt.save_dir)
    wdir                    = save_dir / 'weights'
    last                    = wdir / 'last.pth'
    best                    = wdir / 'best.pth'
    results_file            = save_dir / 'results.txt'
    results_file_per_class  = save_dir / 'results_per_class.txt'

    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    init_seeds(2 + rank)
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    with open(results_file_per_class,'w') as f:
        f.write(('%20s' + '%12s' * 6) % ('class', 'seen', 'nt', 'mp', 'mr', 'map50', 'map') + '\n')
    with open(results_file, 'w') as f:
        f.write('%10s' * 17 % ('epoch', 'mem', 'iterm', 'box', 'obj', 'cls', ' total', 'targets','lr', 'img_size', 'mp', 'mr', 'map50', 'map', 'box', 'obj', 'cls') + '\n')
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    # Create model
    model, ckpt = create_model(opt,device)
    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples img_size-> [h,w]

    # Optimizer parm
    accumulate              = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    quantize                = opt.train_cfg['quantize']
    opt.hyp['weight_decay'] = opt.hyp['weight_decay'] * accumulate # / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {opt.hyp['weight_decay']}")
    # mmdet ??????????????????
    if 'mmdet' in loss_version:
        max_norm        = opt.hyp.get('max_norm',0)
        max_one_norm    = opt.hyp.get('max_one_norm',0)
        if max_norm == 0:
            if max_one_norm > 0:
                max_norm = max_one_norm * batch_size
        logger.info(f"Scaled mmdet max_norm = {max_norm}")
        logger.info(f"Scaled mmdet max_norm = {max_one_norm}")
    # yolov3/yolov5 opitimizer??????
    if loss_version in ['yolov3','yolov5','yolov3-gaussian']:
        if opt.yolov3v5['loss_mean']:
            opt.yolov3v5['box'] *= 3 / nl  # scale to layers
            opt.yolov3v5['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
            opt.yolov3v5['obj'] *= (max(imgsz) / 640) ** 2 * 3 / nl  # scale to image size and layers
            #opt.hyp['warmup_bias_lr'] = opt.yolov3v5['warmup_bias_lr']
        logger.info(f"Scaled yolov3v5 box = {opt.yolov3v5['box']}")
        logger.info(f"Scaled yolov3v5 cls = {opt.yolov3v5['cls']}")
        logger.info(f"Scaled yolov3v5 obj = {opt.yolov3v5['obj']}")

    # Create optimizer
    start_epoch, best_fitness, optimizer, lf, scheduler = create_optimizer(opt, model, epochs, ckpt, results_file)

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.train_cfg['sync_bn'] and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Trainloader
    create_dataloader, test = choose_dataloader_and_test(opt)
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, augment=True, rank=rank)
    dataloader_norm = create_dataloader(train_path, imgsz, batch_size, gs, opt, augment=True, rank=rank, norm=True)[0]
    nb = len(dataloader)  # number of batches

    # Process 0
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt, rank=-1, pad=0)[0]
                                    #    hyp=opt.hyp, cache=opt.train_cfg['cache_images'] and not opt.train_cfg['notest'], rect=opt.train_cfg['rect'],#True
                                    #    rank=-1, world_size=opt.world_size, workers=opt.train_cfg['workers'],pad=0.5,version=loss_version)[0]

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights ?????????
    model.names = names
    if loss_version in ['yolov3','yolov5','yolov3-gaussian']:
        model.hyp = {**opt.hyp, **opt.yolov3v5}  # attach hyperparameters to model
    else:
        model.hyp = opt.hyp  # attach hyperparameters to model
    compute_loss = ComputeLoss(model, opt)
    # Start training
    t0 = time.time()
    t1 = time.time()
    nw = max(round(opt.hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    
    logger.info('Image sizes: (%g,%g) train, (%g,%g) test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (imgsz[0],imgsz[1], imgsz_test[0],imgsz_test[1], dataloader.num_workers, save_dir, epochs))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.train_cfg.get('image_weights',None):
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        #change - dataloader
        if epochs - epoch <= 10:
            dataloader = dataloader_norm
        if rank != -1:#??????dataloader?????????GPU?????????????????????????????????
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        #logger.info(('\n' + '%10s' * 10) % ('Epoch','gpu_mem','batch', 'box', 'obj', 'cls', 'total', 'targets','lr', 'img_size'))
        # if rank in [-1, 0]:
        #     pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        item_num = 0
        for i, (imgs, targets, paths, metas) in pbar:  # batch -------------------------------------------------------------
            item_num += 1
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0\ ???FCOS????????????????????????
            
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [opt.hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [opt.hyp['warmup_momentum'], opt.hyp['momentum']])

            # Multi-scale
            if opt.train_cfg['multi_scale']:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if 'mmdet' in loss_version:
                    gt_labels, gt_boxes = targets
                    img_metas = metas
                    for idx in range(len(img_metas)):
                        gt_labels[idx] = gt_labels[idx].to(device).long() #???????????????label??????????????????long?????????????????????assassin??????????????????
                        gt_boxes[idx] = gt_boxes[idx].to(device)
                    loss, loss_items = compute_loss(pred, [gt_boxes,gt_labels], img_metas)
                else:
                    loss, loss_items = compute_loss(pred, targets if isinstance(targets[0], list) else targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            if "mmdet" in loss_version and max_norm > 0:
                total_norm = nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)
                clip_coef = max_norm / (total_norm + 1e-6)
                if rank in [-1, 0]:
                    print("max_norm:{:6d},total_norm:{:8.2f},clip_coef:{}".format(int(max_norm),float(total_norm.cpu()),float(clip_coef.cpu())))


            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0] and item_num % 5 == 0:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3g' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                if 'mmdet' in loss_version:
                    targets_num = int(sum([temp.shape[0] for temp in targets[1]]))
                elif 'yolov3-gaussian' in loss_version:
                    targets_num = targets[-1].shape[0]
                else:
                    targets_num = targets.shape[0]
                s = ('%10s' * 3 + '%10.4g' * 7) % (
                    '%g/%g' % (epoch, epochs - 1), mem, '%g/%g' % (item_num,len(dataloader)), *mloss, targets_num, optimizer.param_groups[0]['lr'], imgs.shape[-1])
                logger.info(('Epoch: %-8s' + 'batch: %-10s' + 'mem: %-6s' + 'box: %-9.6g' + ' obj:%-9.6g' + ' cls:%-9.6g' + ' total:%-9.6g' + 'tgs:%-5g' + 'lr:%-9.6g' + '  size:%-3g %-3g' + '  t: %-4.3gs')
                    % ('%g/%g' % (epoch+1,epochs), '%g/%g' % (item_num,len(dataloader)), mem, *mloss, targets_num, optimizer.param_groups[0]['lr'], imgs.shape[-1], imgs.shape[-2], time.time()-t1))
                #pbar.set_description(s)

                # Plot
                if plots and ni < 20:
                    if 'mmdet' in loss_version:
                        targets_total = []
                        img_num = 0
                        for labels_gt,boxes_gt in zip(targets[0],targets[1]):
                            idx_batch = torch.tensor([img_num for res in range(len(labels_gt))])[None].T
                            targets_total.append(torch.cat((idx_batch, labels_gt[None].T.cpu(), boxes_gt.cpu()),1))
                            img_num += 1
                        targets = torch.cat(targets_total)
                        targets = torch.cat((targets[:,:2],xyxy2xywh(targets[:,2:]/torch.Tensor([imgsz[1],imgsz[0],imgsz[1],imgsz[0]]))),1)#.to(device)
                    elif 'yolov3-gaussian' in loss_version:
                        targets = targets[-1]
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
            t1 = time.time()
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
            
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            tables = []
            if opt.train_cfg['test'] or final_epoch:  # Calculate mAP
                results, maps, times, tables = test(opt,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.train_cfg.get('single_cls',False),
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 plots=plots and final_epoch,
                                                 log_imgs= 0)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            # if len(opt.train_cfg['name']) and opt.train_cfg['bucket']:
            #     os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.train_cfg['bucket'], opt.train_cfg['name']))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            #Log every_class
            tags_class = [['type/precision_{}'.format(pre_class_name), 'type/recall_{}'.format(pre_class_name), 'type/mAP_0.5_{}'.format(pre_class_name), 'type/mAP_0.5:0.95_{}'.format(pre_class_name)] for pre_class_name in opt.data['names']]
            # tags_per_class = []
            # for res in tags_class:
            #     tags_per_class.extend(res)
            if opt.train_cfg['test']:
                for cls_num in range(nc):
                    #??????1???????????????0??????total???
                    draw_data = np.array(' '.join(tables[cls_num+1].split()).split(' '),dtype=np.float32)[-4:]
                    # if sum(draw_data) == 0:
                    #     pass
                    for x, tag in zip(list(draw_data), tags_class[cls_num]):
                        if tb_writer:
                            tb_writer.add_scalar(tag, x, epoch)  # tensorboard

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            #save = (not opt.train_cfg['nosave']) or (final_epoch and not opt.train_cfg['evolve'])
            if not opt.train_cfg['evolve']:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
            # Write
            with open(results_file_per_class, 'a') as f:
                #f.write(('%20s' + '%12s' * 6) % ('class', 'seen', 'nt', 'mp', 'mr', 'map50', 'map') + '\n')
                [f.write(table + '\n') for table in tables]
                #f.write('\n')
                    # Plots
            if plots and (epoch+1) % 10 == 0: #???5??????????????????????????????????????????
                plot_results(save_dir=save_dir,nc=nc,cls_name=opt.data['names'])  # save as results.png
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    try:
        if rank in [-1, 0]:
            # Strip optimizers
            # final = best if best.exists() else last  # final model
            # for f in [last, best]:#best???last???????????????????????????<??????>
            #     if f.exists(): 
            #         strip_optimizer(f)  # strip optimizers
            # if opt.train_cfg['bucket']:
            #     res = opt.train_cfg['bucket']
            #     os.system(f'gsutil cp {final} gs://{res}/weights')  # upload

            # Plots
            if plots:
                plot_results(save_dir=save_dir,nc=nc,cls_name=opt.data['names'])  # save as results.png

            # Test best.pt
            logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        #     #if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
        #     if opt.data['data_type'].lower() == 'coco' and nc == 80:
        #         for conf, iou, save_json in ([0.25, 0.45, False], [0.001, 0.65, True]):  # speed, mAP tests
        #             results, _, _, ta = test.test(opt.data,
        #                                       batch_size=total_batch_size,
        #                                       imgsz=imgsz_test,
        #                                       conf_thres=conf,
        #                                       iou_thres=iou,
        #                                       model=attempt_load(final, device).half(),
        #                                       single_cls=opt.train_cfg['single_cls'],
        #                                       dataloader=testloader,
        #                                       save_dir=save_dir,
        #                                       save_json=save_json,
        #                                       plots=False)

        else:
            dist.destroy_process_group()
    except:
        logger.info('\nEnd training')

    torch.cuda.empty_cache()
    return results