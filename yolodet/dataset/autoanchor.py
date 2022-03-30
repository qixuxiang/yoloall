# Auto-anchor utils

import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm
import os
import copy
import random
import shutil
# import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
import multiprocessing
# import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, cls_map = None, single_cls=False, stride=32, pad=0.0, rank=-1,version="v5",debug=False,kmean_cls=None):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size[0] // 2, -img_size[1] // 2] if isinstance(img_size,list) else [-img_size // 2, -img_size // 2]
        self.stride = stride
        # self.cls_maps = {}
        self.version = version
        self.debug = debug
        self.kmean_cls = kmean_cls
        # if self.debug:
            
        # for keys,values in cls_map.items():
        #     for i in values:
        #         self.cls_maps[i] = keys

        #assert os.path.exists(path)
        self.image_path = []
        self.label_path = []
        lines = open(path).read().splitlines()
        for i in lines:
            path_a,path_b = i.split(';')
            self.image_path.append(path_a)
            self.label_path.append(path_b)

        cache_path = Path(path).with_suffix('.cache')  # cached labels
        # if cache_path.is_file():
        #     cache = torch.load(cache_path)  # load
        #     if cache['hash'] != get_hash(self.label_path + self.image_path) or 'results' not in cache:  # changed
        #         cache = self.cache_labels(cache_path)  # re-cache
        # else:
        cache = self.cache_labels(cache_path)  # cache

        # Display cache
        [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=desc, total=n, initial=n)
        assert nf > 0 or not augment, f'No labels found in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(cache.values()) #zip(*cache.values())
        # for array in labels:
        #     array[:,0] = [self.cat2label[i] for i in array[:,0]]

        self.labels = labels[0]
        self.shapes = np.array(shapes[0], dtype=np.float64)
        #self.img_files = list(cache.keys())  # update
        #self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes[0])  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path=Path('./labels.cache')):
        # Cache dataset labels, check images and read shapes
        x = {'lable':[],'shape':[]}  # dict
        nm,nf,ne,nc = 0,0,0,0
        def worker(pid_num,start,ends):
            result = {'nm':0, 'nf':0, 'ne':0, 'nc':0,'lable_postion':[],'shape':[]} # number missing, found, empty, duplicate
            image_paths = self.image_path[start:ends] 
            label_paths = self.label_path[start:ends] 
            pbar = tqdm(zip(image_paths, label_paths), desc='Scanning images', total=len(image_paths))
            for i, (im_file, lb_file) in enumerate(pbar):
                try:
                    # verify images
                    im = Image.open(im_file)
                    im.verify()  # PIL verify
                    shape = exif_size(im)  # image size
                    assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                    # verify labels
                    if os.path.isfile(lb_file):
                        result['nf'] += 1  # label found
                        with open(lb_file, 'r') as f:
                            l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                        if len(l):
                            assert l.shape[1] == 5, 'labels require 5 columns each'
                            assert (l >= 0).all(), 'negative labels'
                            assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                            assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                            if self.kmean_cls != None and len(self.kmean_cls) > 0:
                                l = l[[int(class_type) in self.kmean_cls for class_type in l[:,0]]]
                        else:
                            result['ne'] += 1  # label empty
                            l = np.zeros((0, 5), dtype=np.float32)
                    else:
                        result['nm'] += 1  # label missing
                        l = np.zeros((0, 5), dtype=np.float32)
                    # result['img_path'].append(im_file)
                    result['lable_postion'].append(l)
                    result['shape'].append(shape)
                except Exception as e:
                    result['nc'] += 1
                    print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

                pbar.desc = f"{result['nf']} found, {result['nm']} missing, {result['ne']} empty, {result['nc']} corrupted"
            # if nf == 0:
            #     print(f'WARNING: No labels found in {path}. See {help_url}')
            dic[pid_num] = result

        manager = multiprocessing.Manager()
        dic = manager.dict()
        num_worker = 1
        print(f"Use {num_worker} process loading data...")
        one_pid_nums = len(self.label_path) // num_worker
        #jobs = multiprocessing.Process(target=worker,[])#, args=(dic, i, i*2)
        job = []
        for pid in range(num_worker):
            if pid == num_worker-1:
                start = one_pid_nums*pid
                ends = len(self.label_path)
            else:
                start = one_pid_nums*pid
                ends = one_pid_nums*(pid+1)
            jobs = multiprocessing.Process(target=worker,args=(pid,start,ends))
            job.append(jobs)
            jobs.start()

        for p in job:
            p.join()

        for pid in range(num_worker):
            nm += dic[pid]['nm']
            nf += dic[pid]['nf']
            ne += dic[pid]['ne']
            nc += dic[pid]['nc']
            x['lable'].extend(dic[pid]['lable_postion'])
            x['shape'].extend(dic[pid]['shape'])

        x['hash'] = get_hash(self.image_path + self.label_path)
        x['results'] = [nf, nm, ne, nc, len(self.image_path)]
        #torch.save(x, path)  # save for next time
        print(f"New cache created: {path}")
        return x


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    print('\nAnalyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print('New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True, kmean_cls=None):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    # if isinstance(path, str):  # *.yaml file
    #     with open(path) as f:
    #         data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    #     from utils.datasets import LoadImagesAndLabels
    #     dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    # else:
    #     dataset = path  # dataset
    dataset = LoadImagesAndLabels(path, augment=False, rect=False, kmean_cls=kmean_cls)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)


if __name__ == "__main__":
    data_path = "/home/yu/data/dataset/coco/val_label.txt"
    kmean_cls = [0,1,2,3,4,5,6,7,8,9,10]
    kmean_anchors(path=data_path, n=9, img_size=[640,640], thr=4.0, gen=1000, verbose=True, kmean_cls=kmean_cls)