# Dataset utils and dataloaders

import glob
import logging
import math
import os
import copy
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
import multiprocessing
import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
import pdb
from yolodet.utils.general import xyxy2xywh, xywh2xyxy, clean_str
from yolodet.utils.torch_utils import torch_distributed_zero_first
from pycocotools.coco import COCO
from yolodet.dataset.augmentations_ssd import SSDAugmentation

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
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


# def create_dataloader(path, imgsz, batch_size, stride, cls_map, single_cls, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
#                       rank=-1, world_size=1, workers=8, image_weights=False,version='v5'):
def create_dataloader_mmdet_multi_head(path, imgsz, batch_size, stride, opt, augment=False, rank=-1, pad=0.0, rect=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels_multi_head_MMDET(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=opt.hyp,  # augmentation hyperparameters
                                      rect=opt.train_cfg.get('rect',None),  # rectangular training
                                      cache_images=opt.train_cfg.get('cache_images',None),
                                      cls_map = opt.data['cls_map'],
                                      single_cls=opt.train_cfg.get('single_cls',False),
                                      stride=int(stride) if isinstance(stride,int) else int(max(stride)),
                                      pad=pad,
                                      rank=rank,
                                      image_weights=opt.train_cfg.get('image_weights',None),
                                      version=opt.train_cfg['version'],
                                      debug = opt.train_cfg['debug'],
                                      multi_head = opt.multi_head)                                      

    workers=opt.train_cfg['workers']
    image_weights=opt.train_cfg.get('image_weights',None)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // opt.world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        shuffle=sampler is None,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels_multi_head_MMDET.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640):
        self.img_size = img_size

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'stream'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]


class LoadImagesAndLabels_multi_head_MMDET(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, cls_map = None, single_cls=False, stride=32, pad=0.0, rank=-1,version="v5",debug=False,
                 multi_head=None):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        #self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        #self.mosaic_border = [-img_size[0] // 2, -img_size[1] // 2] if isinstance(img_size,list) else [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.cls_maps = {}
        self.version = version
        self.debug = debug
        self.head = multi_head
            
        for keys,values in cls_map.items():
            for i in values:
                self.cls_maps[i] = keys

        assert os.path.exists(path)
        self.image_path = []
        self.label_path = []
        lines = open(path).read().splitlines()
        for i in lines:
            path_a,path_b = i.split(';')
            self.image_path.append(path_a)
            self.label_path.append(path_b)
        self.number = len(self.image_path)


        #cache_path = Path(path).with_suffix('.cache')  # cached labels
        cache_path = path + '.cache'
        logger.info('get cache_path: '+ cache_path + ' # ' + cache_path + '_' + str(self.number))
        if os.path.isfile(cache_path) and os.path.exists(cache_path):
            cache = torch.load(cache_path)  # load
            # if cache['hash'] != get_hash(self.label_path + self.image_path) or 'results' not in cache:  # changed
            #     cache = self.cache_labels(cache_path)  # re-cache
            if cache['name'] != cache_path + '_' + str(self.number):
                cache = self.cache_labels(cache_path)

        else:
            cache = self.cache_labels(cache_path)  # cache

        # Display cache
        if 'results' in cache.keys():
            [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
            desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=desc, total=n, initial=n)
            assert nf > 0 or not augment, f'No labels found in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        # if 'hash' in cache.keys():
        #     cache.pop('hash')  # remove hash
        #labels, shapes = zip(cache.values()) #zip(*cache.values())
        # for array in labels:
        #     array[:,0] = [self.cat2label[i] for i in array[:,0]]
        self.image_path = cache['imgs']
        self.labels = list(cache['labels'])

        self.shapes = np.array(cache['shapes'], dtype=np.float64)
        #self.shapes = np.array(shapes[0], dtype=np.float64)
        #self.img_files = list(cache.keys())  # update
        #self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(self.image_path)  # number of images
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
        self.Augmentation = SSDAugmentation(size=self.img_size, mean=(127, 127, 127), image_files=self.image_path, labels_info=self.labels, scaleup=self.augment,
                                            hyp=self.hyp, augments=self.augment)
        logger.info("Data enhance method: ")
        if isinstance(self.Augmentation.augment, list):
            for method_num in range(len(self.Augmentation.augment)):
                logger.info(f"Data enhance method_{method_num}: ")
                for enhance in self.Augmentation.augment[method_num].transforms:
                    logger.info("  " + str(enhance).split(" ")[0] + ">")
        else:
            for enhance in self.Augmentation.augment.transforms:
                logger.info("  " + str(enhance).split(" ")[0] + ">")
        
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
        x = {'imgs':[],'labels':[],'shapes':[],'name':''}  # dict
        nm,nf,ne,nc = 0,0,0,0
        def worker(pid_num,start,ends):
            result = {'nm':0, 'nf':0, 'ne':0, 'nc':0,'img_path':[],'lable_postion':[],'shape':[]} # number missing, found, empty, duplicate
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
                            #assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                        else:
                            result['ne'] += 1  # label empty
                            l = np.zeros((0, 5), dtype=np.float32)
                    else:
                        result['nm'] += 1  # label missing
                        l = np.zeros((0, 5), dtype=np.float32)
                    result['img_path'].append(im_file)
                    result['lable_postion'].append(l)
                    result['shape'].append(shape)
                except Exception as e:
                    result['nc'] += 1
                    print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

                pbar.desc = f"Scanning '{path}' for images and labels... " \
                            f"{result['nf']} found, {result['nm']} missing, {result['ne']} empty, {result['nc']} corrupted"
            # if nf == 0:
            #     print(f'WARNING: No labels found in {path}. See {help_url}')
            dic[pid_num] = result


        manager = multiprocessing.Manager()
        dic = manager.dict()
        job = []
        num_worker = 10
        one_pid_nums = len(self.label_path) // num_worker
        #jobs = multiprocessing.Process(target=worker,[])#, args=(dic, i, i*2)
        
        for pid in range(num_worker):
            if pid == num_worker-1:
                start = one_pid_nums*pid
                ends = len(self.label_path)
                #jobs = multiprocessing.Process(target=worker,args=(pid,start,ends))
                # job.append(jobs)
                # jobs.start()
                # continue
            else:
                start = one_pid_nums*pid
                ends = one_pid_nums*(pid+1)
            jobs = multiprocessing.Process(target=worker,args=(pid,start,ends))
            job.append(jobs)
            jobs.start()
            # jobs.join()

        for p in job:
            p.join()

        for pid in range(num_worker):
            nm += dic[pid]['nm']
            nf += dic[pid]['nf']
            ne += dic[pid]['ne']
            nc += dic[pid]['nc']
            x['imgs'].extend(dic[pid]['img_path'])
            x['labels'].extend(dic[pid]['lable_postion'])
            x['shapes'].extend(dic[pid]['shape'])

        #x['hash'] = get_hash(self.image_path + self.label_path)
        #x['results'] = [nf, nm, ne, nc, len(self.image_path)]
        x['name'] = path + '_' + str(self.number)
        torch.save(x, path)  # save for next time
        logging.info(f"New cache created: {path}:  {x['name']}")
        return x

    def __len__(self):
        return len(self.image_path)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        _img, _boxes, _labels, img_meta = self.Augmentation(index)
        if 0:
            debug_img = _img.astype(np.uint8)
            h,w = debug_img.shape[:2]
            cv2.namedWindow('Debug',cv2.WINDOW_AUTOSIZE)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #debug_img = copy.deepcopy(img)
            for target_num in range(len(_labels)):
                # if self.version == 'mmdet':
                right_point = (int(_boxes[target_num,0]),int(_boxes[target_num,1]))
                left_point = (int(_boxes[target_num,2]),int(_boxes[target_num,3]))
                cv2.rectangle(debug_img,right_point,left_point,(255,0,0),3)
                cv2.putText(debug_img,str(int(_labels[target_num])),right_point,font, 1.2, (255, 0, 0), 2)
            cv2.imshow('Debug',debug_img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()

        
        # hyp = self.hyp
        # mosaic = self.mosaic and random.random() < hyp['mosaic']
        # if 0:#mosaic:
        #     # Load mosaic
        #     img, labels = load_mosaic(self, index)
        #     shapes = None

        #     # MixUp https://arxiv.org/pdf/1710.09412.pdf
        #     if random.random() < hyp['mixup']:
        #         img2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
        #         r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        #         img = (img * r + img2 * (1 - r)).astype(np.uint8)
        #         labels = np.concatenate((labels, labels2), 0)

        # else:
        #     # Load image
        #     img, (h0, w0), (h, w) = load_image(self, index)
        #     # Letterbox
        #     shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        #     img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment) #shape [h,w]
        #     shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        #     # Load labels
        #     labels = []
        #     x = self.labels[index]
        #     if x.size > 0:
        #         # Normalized xywh to pixel xyxy format -> padding之后框也要进行相应的平移
        #         labels = x.copy()
        #         labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
        #         labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
        #         labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
        #         labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        # if self.augment:
        #     # Augment imagespace
        #     if not mosaic:
        #         img, labels = random_perspective(img, labels,
        #                                          degrees=hyp['Perspectives']['degrees'],
        #                                          translate=hyp['Perspectives']['translate'],
        #                                          scale=hyp['Perspectives']['scale'],
        #                                          shear=hyp['Perspectives']['shear'],
        #                                          perspective=hyp['Perspectives']['perspective'])

        #     # Augment colorspace
        #     augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        #     # Apply cutouts
        #     # if random.random() < 0.9:
        #     #     labels = cutout(img, labels)

        # nL = len(labels)  # number of labels

        # if self.augment:
        #     # flip up-down
        #     if random.random() < hyp['flipud']:
        #         img = np.flipud(img)
        #         if nL:
        #             #labels[:, 2] = 1 - labels[:, 2] # yolov5的写法
        #             labels[:, 2] = img.shape[0] - labels[:, 2]
        #             labels[:, 4] = img.shape[0] - labels[:, 4]
        #             labels[:, [2,4]] = labels[:, [4,2]]

        #     # flip left-right
        #     if random.random() < hyp['fliplr']:
        #         img = np.fliplr(img)
        #         if nL:
        #             #labels[:, 1] = 1 - labels[:, 1] # yolov5的写法
        #             labels[:, 1] = img.shape[1] - labels[:, 1]
        #             labels[:, 3] = img.shape[1] - labels[:, 3]
        #             labels[:, [1,3]] = labels[:, [3,1]]
        if _boxes.shape[0] > 0 and _labels.shape[0] > 0:
            labels = np.concatenate((_labels[None].T, _boxes),axis=1)
        else:
            labels = np.array([])
        if self.debug:
            cv2.namedWindow('Debug',cv2.WINDOW_AUTOSIZE)
            font = cv2.FONT_HERSHEY_SIMPLEX
            debug_img = copy.deepcopy(_img)
            debug_img = debug_img.astype(np.uint8)
            for target_num in range(len(labels)):
                # if self.version == 'mmdet':
                right_point = (int(labels[target_num,1]),int(labels[target_num,2]))
                left_point = (int(labels[target_num,3]),int(labels[target_num,4]))
                cv2.rectangle(debug_img,right_point,left_point,(255,0,0),3)
                cv2.putText(debug_img,str(int(labels[target_num,0])),right_point,font, 1.2, (255, 0, 0), 2)
            cv2.imshow('Debug',debug_img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()


        nL = len(labels)  # number of labels  
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
            if self.cls_maps:
                for box in labels_out:
                    cat = int(box[1])
                    if cat not in self.cls_maps.keys():
                        box[1] = -1
                    else:
                        box[1] = self.cls_maps[cat]
                labels_out = labels_out[labels_out[:,1] >= 0]

        gt_labels = labels_out[:,1]
        gt_boxes = labels_out[:,2:]
        # img_meta = {
        #     'filename':self.image_path[index],
        #     'ori_shape':(h0, w0, 3),
        #     'img_shape':(h, w, 3),
        #     'pad_shape':img.shape,
        #     'scale_factor':(w / w0, h / h0, w / w0, h / h0),
        #     'img_norm_cfg':{'mean': np.array([0., 0., 0.], dtype=np.float32), 'std': np.array([255., 255., 255.], dtype=np.float32), 'to_rgb': False}
        # }

        # Convert
        img = _img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        #multi head label
        gt_label_multi = []
        gt_bboxe_multi = []
        head_nums = self.head['num']
        for hi in range(head_nums):
            head_cls = self.head['head_cls'][hi]
            idx = [l in head_cls for l in gt_labels]

            gt_label_ = gt_labels[idx]
            gt_label_multi.append(gt_label_)

            gt_bboxe_ = gt_boxes[idx]
            gt_bboxe_multi.append(gt_bboxe_)
        return torch.from_numpy(img), gt_label_multi, gt_bboxe_multi, self.image_path[index], img_meta

    @staticmethod
    def collate_fn(batch):
        img, gt_labels, gt_boxes, path, img_metas = zip(*batch)  # transposed
        # for i, l in enumerate(label):
        #     l[:, 0] = i  # add target image index for build_targets() 按照batch分
        return torch.stack(img, 0), [gt_labels, gt_boxes], path, img_metas

