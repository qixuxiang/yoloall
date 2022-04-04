import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from copy import deepcopy
import random as py_random
import os
import math
from scipy.ndimage.filters import gaussian_filter

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


class LoadImageAndLabel(object):
    """return ervery image, bboxes, labels and image_mate 
    -> input boxes: x,y,w,h
    -> change boxes: x,y,x,y
    """
    def __init__(self, image_files, labels_info):
        self.image_files = image_files
        self.labels_info = labels_info

    def __call__(self, index, img=None, boxes=None, labels=None, image_mate=None):
        image_file = self.image_files[index]
        label_info = self.labels_info[index]
        assert os.path.exists(image_file),f'{image_file} not exist'
        img = cv2.imread(image_file)
        labels = label_info[:,0]
        boxes = np.zeros_like(label_info[:,1:])
        boxes[:, 0] = label_info[:, 1] - label_info[:, 3] / 2
        boxes[:, 1] = label_info[:, 2] - label_info[:, 4] / 2
        boxes[:, 2] = label_info[:, 1] + label_info[:, 3] / 2
        boxes[:, 3] = label_info[:, 2] + label_info[:, 4] / 2
        boxes = np.clip(boxes, 0.0, 1.0) # adjust box x1 y1 x2 y2
        image_mate = {
            "filename":image_file
        }
        return img, boxes, labels, image_mate

class ScaleImage(object):
    """
    """
    def __init__(self, img_size, scaleup):
        self.img_size = img_size
        self.augment = scaleup

    def __call__(self, img, boxes=None, labels=None, image_mate=None):
        
        h0, w0 = img.shape[:2]  # orig hw
        #r = self.img_size / max(h0, w0)  # resize image to img_size
        r = min(1.0 * self.img_size[0] / h0, 1.0 * self.img_size[1] / w0)
        if r != 1:  # always resize down, only resize up if training with augmentation
            #interp = cv2.INTER_AREA  if r < 1 and not self.augment else cv2.INTER_LINEAR #cv2.INTER_AREA对于图像采样更好点
            interp = cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        #return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        #     # Letterbox
        # shape = self.img_size  # final letterboxed shape
        img, ratio, pad = self.letterbox(img, self.img_size, auto=False, scaleup=self.augment) #shape [h,w]
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        
        # Load labels
        new_labels = np.zeros_like(boxes)

        if labels.size > 0:
            # Normalized xywh to pixel xyxy format -> padding之后框也要进行相应的平移
            new_labels[:, 0] = ratio[0] * w * boxes[:,0] + pad[0]  # pad width
            new_labels[:, 1] = ratio[1] * h * boxes[:,1] + pad[1]  # pad height
            new_labels[:, 2] = ratio[0] * w * boxes[:,2] + pad[0]
            new_labels[:, 3] = ratio[1] * h * boxes[:,3] + pad[1]

        meta = {
            'ori_shape':(h0, w0, 3),
            'img_shape':(h, w, 3),
            'pad_shape':img.shape,
            'scale_factor':(w / w0, h / h0, w / w0, h / h0),
            'shapes':shapes,
            'img_norm_cfg':{'mean': np.array([0., 0., 0.], dtype=np.float32), 'std': np.array([255., 255., 255.], dtype=np.float32), 'to_rgb': False}
        }

        image_mate = {**image_mate, **meta}
        return img, new_labels, labels, image_mate


    #@staticmethod
    def letterbox(self, img, new_shape=(640, 640), color=(127, 127, 127), auto=True, scaleFill=False, scaleup=True): #color=(114, 114, 114) 填写三通道的均值255/2
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) #w,h shape[0] -> h(new_unpad[1]), shape[1] -> w(new_unpad[0])
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

class GaussianBlur(object):
    """
    blur:中值滤波
    medianBlur：椒盐滤波
    Gaussian Blur:高斯滤波
    """
    def __init__(self, gaussianblur_cfg):
        self.ksize = (3, 5, 7, 9)
        self.prob = gaussianblur_cfg.get('prob', 0.3)

    def __call__(self, img, boxes=None, labels=None, image_mate=None):
        if random.random() <= self.prob:
            rand_num = random.randint(3)
            kernel_size = int(py_random.choice(self.ksize))
            if rand_num == 0:
                img = self.gaussianblur(img, kernel_size)
            # elif rand_num == 1:
            #     img = self.medianblur(img, kernel_size)
            elif  rand_num == 1:
                img = self.blur(img, kernel_size)
            else:
                img = self.bilateralFilter(img, kernel_size)
            
        else:
            pass
        return img, boxes, labels, image_mate
    
    def blur(self, img, ksize):
        return cv2.blur(img, (ksize,ksize))

    def medianblur(self, src, ksize):#src 这个参数是图像路径
        return cv2.medianBlur(src, ksize)

    def bilateralFilter(self, img, ksize):#双线性
        return cv2.bilateralFilter(img, d=ksize, sigmaColor = 255, sigmaSpace = 0)

    def gaussianblur(self, img, ksize):
        return cv2.GaussianBlur(img,(ksize,ksize), 0)


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, image_mate=None):
        for t in self.transforms:
            img, boxes, labels, image_mate = t(img, boxes, labels, image_mate)
        return img, boxes, labels, image_mate


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None, image_mate=None):
        return self.lambd(img, boxes, labels, image_mate)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        return image.astype(np.float32), boxes, labels, image_mate


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels, image_mate


class ToAbsoluteCoords(object): #已经转换为绝对坐标了
    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels, image_mate


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels, image_mate


class Resize(object):
    def __init__(self, size=[300,300]):
        self.size = size

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if boxes.size: #这边一定要确认boxes不是空的
            if boxes.max() > 1:#mmdet
                boxes[:,0] = boxes[:,0] * (self.size[1]/image.shape[1])
                boxes[:,2] = boxes[:,2] * (self.size[1]/image.shape[1])
                boxes[:,1] = boxes[:,1] * (self.size[0]/image.shape[0])
                boxes[:,3] = boxes[:,3] * (self.size[0]/image.shape[0])
            else:#yolo
                boxes[:,0] = boxes[:,0] * (self.size[1])
                boxes[:,2] = boxes[:,2] * (self.size[1])
                boxes[:,1] = boxes[:,1] * (self.size[0])
                boxes[:,3] = boxes[:,3] * (self.size[0])

        image = cv2.resize(image, (self.size[1],self.size[0]))
        
        # invalid_idx = ((boxes[:,2] - boxes[:,0]) < 10) | ((boxes[:,3] - boxes[:,1]) < 10)
        # boxes = boxes[~invalid_idx]
        # labels = labels[~invalid_idx]
        image_mate['pad_shape'] = (self.size[0],self.size[1],image_mate['pad_shape'][-1])
        return image, boxes, labels, image_mate



class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, prob=0.3):
        self.lower = lower
        self.upper = upper
        self.prob = prob
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.random() < self.prob:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels, image_mate


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, image_mate


class RandomLightingNoise(object):
    """
    # 随机光噪声
    """
    def __init__(self, prob=0.05):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.prob = prob

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.random() < self.prob:
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels, image_mate


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels, image_mate


class RandomContrast(object):
    """
    # 对比度
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels, image_mate


class RandomBrightness(object):
    """
    # 随机亮度 15%
    """
    def __init__(self, delta=32, prob=0.15):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.prob = prob

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.random() >= 1-self.prob:
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels, image_mate


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None, image_mate=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels, image_mate


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None, image_mate=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels, image_mate


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, prob=0.5):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

        self.prob = prob

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.random() < self.prob:
            return self.solution(image, boxes, labels, image_mate)
        else:
            return image, boxes, labels, image_mate

    def solution(self, image, boxes=None, labels=None, image_mate=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None or labels.size <= 0:
                return image, boxes, labels, image_mate

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels, image_mate


class Expand(object):
    def __init__(self, mean, prob=0.5):
        self.mean = mean
        self.prob = prob

    def __call__(self, image, boxes, labels, image_mate=None):
        if random.random() > self.prob:
            return image, boxes, labels, image_mate
        else:
            return self.solution(image, boxes, labels, image_mate)

    def solution(self, image, boxes, labels, image_mate):
        height, width, depth = image.shape
        ratio = random.uniform(1, 2)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        return image, boxes, labels, image_mate


class RandomMirror(object):
    """
    # 随机翻转
    """ 
    def __call__(self, image, boxes, classes, image_mate=None):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes, image_mate


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    """
    # 1、随机对比
    # 2、颜色空间变化
    # 3、随机饱和
    # 4、随机色调
    # 5、随机亮度
    # 6、随机光噪声
    """
    def __init__(self, prob=0.4):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        self.prob = prob

    def __call__(self, image, boxes, labels, image_mate=None):
        if random.random() < self.prob:
            im = image.copy()
            im, boxes, labels, image_mate = self.rand_brightness(im, boxes, labels, image_mate)
            if random.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            im, boxes, labels, image_mate = distort(im, boxes, labels, image_mate)
            return self.rand_light_noise(im, boxes, labels, image_mate)
        else:
            return image, boxes, labels, image_mate


class Perspective(object):
    """
    """
    def __init__(self, perspective_cfg):
        # Perspectives        = hyp['Perspectives']
        self.degrees        = perspective_cfg.get('degrees', 0)
        self.translate      = perspective_cfg.get('translate', 0)
        self.scale          = perspective_cfg.get('scale', 0)
        self.shear          = perspective_cfg.get('shear', 0)
        self.perspective    = perspective_cfg.get('perspective', 0)
        self.border         = perspective_cfg.get('border', (0,0))

    def __call__(self, image, boxes, labels, image_mate=None):
        if random.randint(2):
            return image, boxes, labels, image_mate
        targets = []
        if labels.size:
            assert boxes.shape[0] == labels.shape[0], f"{boxes.shape[0]},{labels.shape[0]}"
            targets = np.concatenate((labels[None].T, boxes),axis=1)
        height = image.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = image.shape[1] + self.border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(127, 127, 127))
            else:  # affine
                image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(127, 127, 127))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self.box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]

            labels = targets[:,0]
            boxes = targets[:,1:]
        return image, boxes, labels, image_mate


    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


class Mosaic(object):
    """
    load_mosaic4: choose 0~4 images to creat a new_image 0.5
    load_mosaic9: choose 5~9 images to creat a new_image 0.5
    """
    def __init__(self, img_size, image_files, labels_info, mosaic_cfg=None):
        self.img_size = img_size
        self.mosaic_border = [-img_size[0] // 1, -img_size[1] // 1] if isinstance(img_size,list) else [-img_size // 2, -img_size // 2]
        self.n = len(image_files)
        self.indices = range(self.n)
        self.image_files = image_files
        self.labels_info = labels_info
        self.mosaic_mode = mosaic_cfg.get('mode',0)

    def __call__(self, index, image=None, boxes=None, labels=None, image_mate=None):
        if random.randint(2):
            image, labels_mosaic, image_mate = self.load_mosaic9(index)
        else:
            image, labels_mosaic, image_mate = self.load_mosaic4(index)
        if labels_mosaic.size:
            labels = labels_mosaic[:,0]
            boxes = labels_mosaic[:,1:]
        else:
            boxes = np.array([[]])
            labels = np.array([])
        return image, boxes, labels, image_mate

    def load_image(self, index):
        path = self.image_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        #r = self.img_size / max(h0, w0)  # resize image to img_size
        r = min(1.0 * self.img_size[0] / h0, 1.0 * self.img_size[1] / w0)
        if r != 1:  # always resize down, only resize up if training with augmentation
            #interp = cv2.INTER_AREA  if r < 1 and not self.augment else cv2.INTER_LINEAR #cv2.INTER_AREA对于图像采样更好点
            interp = cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def load_mosaic4(self, index):
        #loads images in a mosaic
        labels4 = []
        img_list = []
        shape_list = []
        ori_shape_list = []
        sx = self.img_size[1] #960
        sy = self.img_size[0] #576
        assert isinstance(self.img_size,list),'img_size need to be list'
        #yc,xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        yc = int(random.uniform(-self.mosaic_border[0], 2 * sy + self.mosaic_border[0])) # -> [-288, 864]
        xc = int(random.uniform(-self.mosaic_border[1], 2 * sx + self.mosaic_border[1])) # -> [-480, 1440]mosaic center x, y
        #indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]  # 3 additional image indices
        if self.mosaic_mode:
            indices = [index] + py_random.choices(self.indices, k=py_random.randint(1,3))
        else:
            indices = [index] + py_random.choices(self.indices, k=3)#8 additional image indices
        py_random.shuffle(indices)
        #创建画布
        if len(indices) > 2:
            paint_y,paint_x = sy * 2, sx * 2
        else:
            paint_y,paint_x = sy * 1, sx * 2 # + self.mosaic_border[1]*(-1)

        for i, index in enumerate(indices):
            # Load image
            img_list.append(self.image_files[index])
            img, _, (h, w) = self.load_image(index) #_为原图尺寸
            ori_shape_list.append(_)
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((paint_y, paint_x, img.shape[2]), 127, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, sx * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(sy * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, sx * 2), min(sy * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            shape_list.append((padh,padw))

            # Labels
            x = self.labels_info[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1], 0, 2 * sx, out=labels4[:, 1])
            np.clip(labels4[:, 2], 0, 2 * sy, out=labels4[:, 2])
            np.clip(labels4[:, 3], 0, 2 * sx, out=labels4[:, 3])
            np.clip(labels4[:, 4], 0, 2 * sy, out=labels4[:, 4])
            #np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective
            # img4, labels4 = replicate(img4, labels4)  # replicate
        
        # mosaic_labels, mosaic_boxes, _ = np.split(labels4,[1,5],1)
        # mosaic_labels = mosaic_labels.squeeze()
        
        img_meta = {
            'filename':img_list,
            'ori_shape':ori_shape_list,
            'img_shape':shape_list,
            'pad_shape':img4.shape,
            'scale_factor':[[last[1]/ori[1],last[0]/ori[0],last[1]/ori[1],last[0]/ori[0]] for last,ori in zip(shape_list,ori_shape_list)],
            'img_norm_cfg':{'mean': np.array([0., 0., 0.], dtype=np.float32), 'std': np.array([255., 255., 255.], dtype=np.float32), 'to_rgb': False}
        }
        invalid_idx = ((labels4[:,3] - labels4[:,1]) < 10) | ((labels4[:,4] - labels4[:,2]) < 10)
        labels4 = labels4[~invalid_idx]
        return img4, labels4, img_meta# mosaic_boxes, mosaic_labels,

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9 = []
        img_list = []
        shape_list = []
        ori_shape_list = []
        #s = self.img_size
        sx = self.img_size[1]  #960
        sy = self.img_size[0]  #576

        if self.mosaic_mode:
            indices = [index] + py_random.choices(self.indices, k=py_random.randint(1,8))
        else:
            indices = [index] + py_random.choices(self.indices, k=8)#8 additional image indices
        py_random.shuffle(indices)
        x_det,y_det = 0,0
        #创建画布
        if len(indices) == 2:
            paint_y,paint_x = sy * 2, sx * 1
            x_det = self.img_size[1]
        elif len(indices) in [3,4]:
            paint_y,paint_x = sy * 2, sx * 2
            x_det = self.img_size[1]
        elif len(indices) in [5,6]:
            paint_y,paint_x = sy * 3, sx * 2
            x_det = self.img_size[1]
        else:
            paint_y,paint_x = sy * 3, sx * 3

        for i, index in enumerate(indices):
            # Load image
            img_list.append(self.image_files[index])
            img, _, (h, w) = self.load_image(index) #_为原图尺寸
            ori_shape_list.append(_)

            # place img in img9
            if i == 0:  # center 填充中心
                img9 = np.full((paint_y, paint_x, img.shape[2]), 127, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = sx, sy, sx + w, sy + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top 填充正上
                c = sx, sy - h, sx + w, sy
            elif i == 2:  # top right
                c = sx + wp, sy - h, sx + wp + w, sy
            elif i == 3:  # right
                c = sx + w0, sy, sx + w0 + w, sy + h
            elif i == 4:  # bottom right
                c = sx + w0, sy + hp, sx + w0 + w, sy + hp + h
            elif i == 5:  # bottom
                c = sx + w0 - w, sy + h0, sx + w0, sy + h0 + h
            elif i == 6:  # bottom left
                c = sx + w0 - wp - w, sy + h0, sx + w0 - wp, sy + h0 + h
            elif i == 7:  # left
                c = sx - w, sy + h0 - h, sx, sy + h0
            elif i == 8:  # top left
                c = sx - w, sy + h0 - hp - h, sx, sy + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            x = self.labels_info[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padx - x_det
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + pady - y_det
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padx - x_det
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + pady - y_det

            labels9.append(labels)
            #problem
            if self.mosaic_mode:
                pwd_x1 = x1 - x_det
                pwd_x2 = x2 - x_det if pwd_x1 >= 0 else x2 - x_det - pwd_x1
                # Image
                img9[y1 - y_det:y2 - y_det, max(pwd_x1, 0):pwd_x2] = img[y1 - pady:, x1 - padx:]
            else: 
                img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous
            shape_list.append((hp, wp))

        # Offset
        #yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        yc = int(random.uniform(0, sy))
        xc = int(random.uniform(0, sx))
        img9 = img9[yc:yc + 2 * sy, xc:xc + 2 * sx]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc

        if len(labels9):
            np.clip(labels9[:, 1], 0, 2 * sx, out=labels9[:, 1])
            np.clip(labels9[:, 2], 0, 2 * sy, out=labels9[:, 2])
            np.clip(labels9[:, 3], 0, 2 * sx, out=labels9[:, 3])
            np.clip(labels9[:, 4], 0, 2 * sy, out=labels9[:, 4])
        
        # mosaic_labels, mosaic_boxes, _ = np.split(labels9,[1,5],1)
        # mosaic_labels = mosaic_labels.squeeze()
        
        img_meta = {
            'filename':img_list,
            'ori_shape':ori_shape_list,
            'img_shape':shape_list,
            'pad_shape':img9.shape, #一个img_meta中不管融合几张图片只能有一个pad_shape(这个pad_shape是最后送入网络的尺寸)
            'scale_factor':[[last[1]/ori[1],last[0]/ori[0],last[1]/ori[1],last[0]/ori[0]] for last,ori in zip(shape_list,ori_shape_list)],
            'img_norm_cfg':{'mean': np.array([0., 0., 0.], dtype=np.float32), 'std': np.array([255., 255., 255.], dtype=np.float32), 'to_rgb': False}
        }
        invalid_idx = ((labels9[:,3] - labels9[:,1]) < 10) | ((labels9[:,4] - labels9[:,2]) < 10)
        labels9 = labels9[~invalid_idx]
        return img9, labels9, img_meta#mosaic_boxes, mosaic_labels,

class CutOut(object):
    """
    # cutout 用在目标框变更为绝对坐标后
    # 随机擦除 用不同颜色填充 默认擦除的框不与目标框iou不能超过30%
    """
    def __init__(self, cutout_cfg):#prob=0.01
        self.scales = [0.5, 0.25, 0.125, 0.0625, 0.03125] #重叠的百分比
        self.nums = [1, 2, 4, 8, 16] #每个重叠比例的个数
        self.prob = cutout_cfg.get('prob',0.01)

    def __call__(self, image, boxes, labels, image_mate=None):
        if random.random() <= self.prob:
            return self.solution(image, boxes, labels, image_mate)
        else:
            return image, boxes, labels, image_mate

    def solution(self, image, boxes, labels, image_mate):
        if labels.size:
            assert boxes.shape[0] == labels.shape[0], f"{boxes.shape[0]},{labels.shape[0]}"
            target = np.concatenate((labels[None].T, boxes), axis=1)
        h, w = image.shape[:2]
        scales = []  # image size fraction
        for i, j in zip(self.scales, self.nums):
            scales.extend([i]*j)
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            image[ymin:ymax, xmin:xmax] = [random.randint(0, 255) for _ in range(3)]

            # return unobscured labels
            if labels.size and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, target[:, 1:5])  # intersection over area
                target = target[ioa < 0.30]  # remove >30% obscured labels
        
        if labels.size:
            labels = target[:,0]
            boxes = target[:,1:]
        return image, boxes, labels, image_mate
    

class ColorJittering(object):
    """
    """
    def __init__(self):
        self.transform2 = transforms.Compose([
                                    #transforms.ToTensor(),
                                    transforms.ColorJitter(brightness=0.5, contrast=0., saturation=0., hue=0.5)
                                    ]
                    )

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        #transforms.RandomInvert(p=0.5) #以给定的概率随机地插入给定图像的颜色
        #transforms.RandomPosterize(bits, p=0.5)(img) #通过减少每个颜色通道的比特数，以给定的概率随机地对图像进行贴图。
        #transforms.RandomSolarize(threshold,p=0.5)(img) #通过反转所有高于阈值的像素值，以给定的概率随机地晒出图像。
        #transforms.RandomAdjustSharpness(sharpness_factor,p=0.5)(img) #以给定的概率随机调整图像的锐度。
        #transforms.RandomAutocontrast(p=0.5)(img) 以一个给定的概率随机地对给定图像的像素进行最大化对比度操作。
        #transforms.RandomEqualize(p=0.5)(img) 以给定的概率随机对图像进行直方图均衡化。

        #以下考虑使用：
        #[torchvision.transforms.AutoAugmentPolicy.CIFAR10, torchvision.transforms.AutoAugmentPolicy.IMAGENET, torchvision.transforms.AutoAugmentPolicy.SVHN]
        # transforms.RandAugment()
        # transforms.TrivalAugmentWide()

        #图像由BGR转换为RGB、在转为tensor

        image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()
        #image = #transforms.ColorJitter(brightness=0.5, contrast=0., saturation=0., hue=0.).forward(image)

        image = image.numpy().transpose((1, 2, 0))
        return image, boxes, labels, image_mate
        

class MixUp(object):
    """
    # 图片混合:两张相同大小的图片进行混合[不建议或者少用]
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    """
    def __init__(self, image_files, labels_info, img_size, scaleup, mixup_cfg):
        self.indices = range(len(image_files))
        self.prob = mixup_cfg.get('prob', 0.01)
        self.add_img = Compose([
                LoadImageAndLabel(image_files, labels_info),
                ConvertFromInts(),
                ScaleImage(img_size, scaleup)])

    def __call__(self, image, boxes, labels, image_mate):
        if random.random() < self.prob:
            return self.solution(image, boxes, labels, image_mate)
        else:
            return image, boxes, labels, image_mate


    def solution(self, image, boxes, labels, image_mate):
        index = py_random.choices(self.indices, k=1)[0]
        image_add, boxes_add, labels_add, image_mate_add = self.add_img(index)
        r = random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        image = (image * r + image_add * (1 - r)).astype(np.uint8)
        if labels.size and labels_add.size:
            boxes = np.concatenate((boxes, boxes_add), 0)
            labels = np.concatenate((labels, labels_add), 0)
            new_mate = {}
            keys = image_mate.keys()
            for item in keys:
                if item == 'pad_shape': #pad_shape 为送入网络里的大小
                    new_mate['pad_shape'] = image_mate['pad_shape']
                else:
                    new_mate[item] = []
                    new_mate[item].extend([image_mate[item],image_mate_add[item]])
            image_mate = new_mate
        elif labels.size <= 0 and labels_add.size > 0:
            boxes = boxes_add
            labels = labels_add
            image_mate = image_mate_add
        else:
            pass
        return image, boxes, labels, image_mate

class HistEqualize(object):
    """直方图均衡<不与高斯模糊一起开>
    # 注意opencv读取的颜色通道为[BGR],而matplotlib的颜色通道顺序为[RGB]
    # cv2.createCLAHE() #限制对比度自适应直方图均衡
    # cv2.equalizeHist() #进行直方图均衡化
    """
    def __init__(self, histequal_cfg):
        self.clahe = histequal_cfg.get('clahe',True)
        self.bgr = histequal_cfg.get('bgr',True)
        self.prob = histequal_cfg.get('prob',0.3)

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.random() < self.prob:
            image = self.solution(image.astype(np.uint8))
            image = image.astype(np.float32)
        return image, boxes, labels, image_mate

    def solution(self, image):
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV if self.bgr else cv2.COLOR_RGB2YUV)# convert YUV image to RGB
        if self.clahe:
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if self.bgr else cv2.COLOR_YUV2RGB)
        
# class CopyPaste(object):
#     """
#     """
#     def __init__(self):
#         pass

#     def __call__(self):
#         pass

class CopyPaste(object):
    """
    # 功能 对指定的label的图像从其他图片中抠出来贴到该图片中
    """
    def __init__(self, image_files, labels_info, img_size, scaleup, copypaste_cfg):#prob=0.3, past_class=[121,123,124,126,127]
        self.indices = range(len(image_files))
        self.image_files = image_files
        self.labels_info = labels_info
        self.past_class = copypaste_cfg.get('past_class',[121,123,124,126,127])
        self.past_times = copypaste_cfg.get('past_times',5)
        self.prob = copypaste_cfg.get('prob',0.3)
        self.add_img = Compose([
                LoadImageAndLabel(image_files, labels_info),
                ConvertFromInts(),
                ScaleImage(img_size, scaleup)])
        self.past_class_info1 = copypaste_cfg.get('past_class_info1',[123,124,126,127])
        self.past_number = copypaste_cfg.get('past_number',5)

    def __call__(self, image, boxes, labels, image_mate=None):
        if random.random() >= 1-self.prob:
            return self.solution(image, boxes, labels, image_mate)
        else:
            return image, boxes, labels, image_mate
        
    def solution(self, image, boxes, labels, image_mate):
        total, new_image_mate = self.select_info(image_mate)
        image_past = total['image_past']
        boxes_past = total['boxes_pasts']
        labels_past = total['labels_pasts']
        if len(labels_past) > 0:
            for time in range(self.past_times):
                idx = random.randint(labels_past.shape[0])
                past_box_info = boxes_past[idx]
                past_label_info = labels_past[idx]

                past_w = int(past_box_info[2]) - int(past_box_info[0])
                past_h = int(past_box_info[3]) - int(past_box_info[1])
                #限制贴图的大小
                if len(self.past_class_info1) and past_label_info in self.past_class_info1:
                    if past_w < 60 and past_h < 60:
                        continue
                else:
                    if past_w < 30 and past_h < 30:
                        continue

                # point [只在图像下半部分进行贴图]
                if image.shape[1] - past_w <= 0 or image.shape[0]-past_h <= image.shape[0]//4:
                    continue
                past_x0 = random.randint(0,image.shape[1] - past_w)
                past_y0 = random.randint(image.shape[0]//4, image.shape[0]-past_h)
                past_x1 = past_x0 + past_w
                past_y1 = past_y0 + past_h
                past_box = np.array([past_x0, past_y0, past_x1, past_y1], dtype=np.float32)
                past_label = np.array([past_label_info])
                if labels.size <= 0:
                    boxes = np.array([[past_x0, past_y0, past_x1, past_y1]], dtype=np.float32)
                    labels = past_label
                else:
                    ioa = bbox_ioa(past_box, boxes)  # intersection over area
                    if np.all(ioa <= 0.001):
                        image[past_y0:past_y1, past_x0:past_x1] = image_past[idx]#[int(past_box_info[1]):int(past_box_info[3]), int(past_box_info[0]):int(past_box_info[2])]
                        if len(boxes.shape) == 1:
                            boxes = boxes[None]
                        boxes = np.concatenate([boxes,past_box[None]],0)
                        labels = np.concatenate([labels, past_label],0)
        
        image_mate = new_image_mate
        return image, boxes, labels, image_mate

        # index = py_random.choices(self.indices, k=1)[0]
        # image_past, boxes_past, labels_past, image_mate_past = self.add_img(index)

        # if labels_past.size <= 0:#如果抽取的是背景图就过
        #     return image, boxes, labels, image_mate
        # if len(self.past_class) > 0:
        #     flg = np.ones_like(labels_past) < 0
        #     for class_type in self.past_class:
        #         flg = flg | (labels_past == class_type)
        #     boxes_past = boxes_past[flg]
        #     labels_past = labels_past[flg]

        # if len(labels_past) > 0:
        #     for time in range(self.past_times):
        #         idx = random.randint(labels_past.shape[0])
        #         past_box_info = boxes_past[idx]
        #         past_label_info = labels_past[idx]

        #         past_w = int(past_box_info[2]) - int(past_box_info[0])
        #         past_h = int(past_box_info[3]) - int(past_box_info[1])
        #         #限制贴图的大小
        #         if len(self.past_class_info1) and past_label_info in self.past_class_info1:
        #             if past_w < 60 and past_h < 60:
        #                 continue
        #         else:
        #             if past_w < 30 and past_h < 30:
        #                 continue

        #         # point [只在图像下半部分进行贴图]
        #         if image.shape[1] - past_w <= 0 or image.shape[0]-past_h <= image.shape[0]//4:
        #             continue
        #         past_x0 = random.randint(0,image.shape[1] - past_w)
        #         past_y0 = random.randint(image.shape[0]//4, image.shape[0]-past_h)
        #         past_x1 = past_x0 + past_w
        #         past_y1 = past_y0 + past_h
        #         past_box = np.array([past_x0, past_y0, past_x1, past_y1], dtype=np.float32)
        #         past_label = np.array([past_label_info])
        #         if labels.size <= 0:
        #             boxes = np.array([[past_x0, past_y0, past_x1, past_y1]], dtype=np.float32)
        #             labels = past_label
        #         else:
        #             ioa = bbox_ioa(past_box, boxes)  # intersection over area
        #             if np.all(ioa <= 0.001):
        #                 image[past_y0:past_y1, past_x0:past_x1] = image_past[int(past_box_info[1]):int(past_box_info[3]), int(past_box_info[0]):int(past_box_info[2])]
        #                 if len(boxes.shape) == 1:
        #                     boxes = boxes[None]
        #                 boxes = np.concatenate([boxes,past_box[None]],0)
        #                 labels = np.concatenate([labels, past_label],0)
        
        # return image, boxes, labels, image_mate


    def select_info(self,image_mate):
        total = {"image_past":[]}
        index = py_random.choices(self.indices, k=self.past_number)
        image_pasts = []
        boxes_pasts = []
        labels_pasts = []

        image_mate_pasts = {}
        keys = image_mate.keys()
        for item in keys:
            if item == 'pad_shape':
                image_mate_pasts['pad_shape'] = image_mate['pad_shape']
            else:
                image_mate_pasts[item] = []
                image_mate_pasts[item].append(image_mate[item])

        for idx in index:
            image_past, boxes_past, labels_past, image_mate_past = self.add_img(idx)
            if labels_past.size <= 0:#如果抽取的是背景图就过
                continue

            img_past = []
            if len(self.past_class) > 0:
                flg = np.ones_like(labels_past) < 0
                for class_type in self.past_class:
                    flg = flg | (labels_past == class_type)
                boxes_past = boxes_past[flg]
                labels_past = labels_past[flg]
                if len(boxes_past) and len(labels_past) and len(boxes_past)==len(labels_past):
                    img_past = [image_past[int(past_box_info[1]):int(past_box_info[3]), int(past_box_info[0]):int(past_box_info[2])] for past_box_info in boxes_past]
                    image_pasts.extend(img_past)
                    boxes_pasts.append(boxes_past)
                    labels_pasts.append(labels_past)
                    for item in keys:
                        if item == 'pad_shape': #pad_shape 为送入网络里的大小
                            image_mate_pasts['pad_shape'] = image_mate['pad_shape']
                        else:
                            image_mate_pasts[item].append(image_mate_past[item])

        total['image_past'] = image_pasts
        total['boxes_pasts'] = np.concatenate(boxes_pasts,0) if len(boxes_pasts) > 0 else []
        total['labels_pasts'] = np.concatenate(labels_pasts,0) if len(boxes_pasts) > 0 else []
        return total, image_mate_pasts


class CutMix(object):
    """
    # 贴图
    # 与MixUp不能同开 -> right_top_pach：第一张图的左上角被第二张图的左上角替换
    #                 -> right_bottom_pach: 第一张图的左下角被第二张图的左下角替换
    #                 -> left_top_pach: 第一张图的右上角被第二张图的右上角替换 
    #                 -> left_bottom_pach: 第一张图的右下角被第二张图的右下角替换
    """
    def __init__(self, image_files, labels_info, img_size=None, scaleup=None, cutmix_cfg=None):
        self.indices = range(len(image_files))
        self.prob = cutmix_cfg.get('prob', 0.6)
        self.size = img_size
        self.point_range = [[img_size[0]//4, 3 * img_size[0]//4], [img_size[1]//4, 3 * img_size[1]//4]]
        self.add_img = Compose([
                LoadImageAndLabel(image_files, labels_info),
                ConvertFromInts(),
                ScaleImage(img_size, scaleup)])

    def __call__(self, image, boxes, labels, image_mate):
        if random.random() < self.prob:
            index = py_random.choices(self.indices, k=1)[0]
            image_add, boxes_add, labels_add, image_mate_add = self.add_img(index)
            rand_num = random.randint(4)
            if rand_num == 4:
                image, boxes, labels = self.right_top_pach(image, boxes, labels, image_add, boxes_add, labels_add)
            elif rand_num == 3:
                image, boxes, labels = self.left_bottom_pach(image, boxes, labels, image_add, boxes_add, labels_add)
            elif rand_num == 2:
                image, boxes, labels = self.left_top_pach(image, boxes, labels, image_add, boxes_add, labels_add)
            else:
                image, boxes, labels = self.right_bottom_pach(image, boxes, labels, image_add, boxes_add, labels_add)
            if boxes.size and labels.size:
                invalid_idx = ((boxes[:,2] - boxes[:,0]) < 10) | ((boxes[:,3] - boxes[:,1]) < 10)
                boxes = boxes[~invalid_idx]
                labels = labels[~invalid_idx]
            else:
                boxes = np.array([[]])
                labels = np.array([])
            if len(image_mate_add) and len(image_mate):
                new_mate = {}
                keys = image_mate.keys()
                for item in keys:
                    if item == 'pad_shape': #pad_shape 为送入网络里的大小
                        new_mate['pad_shape'] = image_mate['pad_shape']
                    else:
                        new_mate[item] = []
                        new_mate[item].extend([image_mate[item],image_mate_add[item]])
                image_mate = new_mate
            elif len(image_mate_add) > 0 and len(image_mate) <= 0:
                image_mate = image_mate_add
            else:
                pass
        return image, boxes, labels, image_mate

    def right_top_pach(self, image, boxes, labels, image_add, boxes_add, labels_add):
        #right_top_pach process
        if labels.size:
            assert boxes.shape[0] == labels.shape[0], f"{boxes.shape[0]},{labels.shape[0]}"
            target = np.concatenate((labels[None].T, boxes), axis=1)

        if labels_add.size:
            assert boxes_add.shape[0] == labels_add.shape[0], f"{boxes_add.shape[0]},{labels_add.shape[0]}"
            target_add = np.concatenate((labels_add[None].T, boxes_add), axis=1)

        y_c = random.randint(self.point_range[0][0], self.point_range[0][1])
        x_c = random.randint(self.point_range[1][0], self.point_range[1][1])
        # h = self.size[0]
        # w = self.size[1]
        image[0:y_c, 0:x_c] = image_add[0:y_c, 0:x_c] #(0,0,0)
        vaild_targe = []
        if labels.size:
            delet_flag = (target[:, 3] < x_c) & (target[:, 4] < y_c) #先选出右下点在删除区域的框 剔掉
            target = target[~delet_flag]
            if target.shape[0] > 0:
                vaild_targe = np.zeros_like(target)
                vaild_targe[:,0] = target[:,0] #注意将lable合并
                vaild_targe[:,3:] = target[:,3:]
                #image1 procecs
                for every_box, save_box in zip(target, vaild_targe):
                    if every_box[1] < x_c and every_box[2] < y_c:
                        #交界点以1/2的目标框大小为界限
                        if (every_box[3] - every_box[1] > x_c - every_box[1]) and (every_box[4] - every_box[2] > y_c - every_box[2]):
                            if  x_c - every_box[1] < (every_box[3] - every_box[1])/2 or y_c - every_box[2] < (every_box[4] - every_box[2])/2:
                                save_box[1] = every_box[1]
                                save_box[2] = every_box[2]
                                continue
                        
                        if every_box[3] > x_c:
                            save_box[1] = x_c
                        else:
                            save_box[1] = every_box[1]

                        if every_box[4] > y_c:
                            save_box[2] = y_c
                        else:
                            save_box[2] = every_box[2]
                    else:
                        save_box[1] = every_box[1]
                        save_box[2] = every_box[2]

        vaild_targe_add = []
        if labels_add.size:
            add_flag = (target_add[:,1] < x_c) & (target_add[:,2] < y_c) #先选出左上的点粘贴区域的框 保留
            target_add = target_add[add_flag]
            if target_add.shape[0] > 0:
                vaild_targe_add = np.zeros_like(target_add)
                vaild_targe_add[:,:3] = target_add[:,:3]
                #image2 procecs
                for every_box, save_box in zip(target_add, vaild_targe_add):
                    if every_box[3] > x_c or every_box[4] > y_c:
                        if every_box[3] > x_c:
                            save_box[3] = x_c
                        else:
                            save_box[3] = every_box[3]

                        if every_box[4] > y_c:
                            save_box[4] = y_c
                        else:
                            save_box[4] = every_box[4]
                    else:
                        save_box[3] = every_box[3]
                        save_box[4] = every_box[4]

        if len(vaild_targe) and len(vaild_targe_add):
            target_cat = np.concatenate((vaild_targe, vaild_targe_add), 0)
        elif len(vaild_targe) > 0 and len(vaild_targe_add) <= 0:
            target_cat = vaild_targe
        elif len(vaild_targe) <= 0 and len(vaild_targe_add) > 0:
            target_cat = vaild_targe_add
        else:
            return image, boxes, labels

        boxes = target_cat[:, 1:]
        labels = target_cat[:, 0]
        return image, boxes, labels
     
    def right_bottom_pach(self, image, boxes, labels, image_add, boxes_add, labels_add):
        #right_bottom_pach process
        if labels.size:
            assert boxes.shape[0] == labels.shape[0], f"{boxes.shape[0]},{labels.shape[0]}"
            target = np.concatenate((labels[None].T, boxes), axis=1)

        if labels_add.size:
            assert boxes_add.shape[0] == labels_add.shape[0], f"{boxes_add.shape[0]},{labels_add.shape[0]}"
            target_add = np.concatenate((labels_add[None].T, boxes_add), axis=1)

        y_c = random.randint(self.point_range[0][0], self.point_range[0][1])
        x_c = random.randint(self.point_range[1][0], self.point_range[1][1])
        h = self.size[0]
        w = self.size[1]
        image[y_c:h, 0:x_c] = image_add[y_c:h, 0:x_c] #(0,0,0)
        if labels.size and labels_add.size:
            target = self.change2leftpoints(target)
            target_add = self.change2leftpoints(target_add)

        vaild_targe = []
        if labels.size:
            delet_flag = (target[:, 1] < x_c) & (target[:, 2] > y_c) #先选出右下点在删除区域的框 剔掉
            target = target[~delet_flag]

            if target.shape[0] > 0:
                vaild_targe = np.zeros_like(target)
                vaild_targe[:,:3] = target[:,:3]
                #image1 procecs
                for every_box, save_box in zip(target, vaild_targe):
                    if every_box[3] < x_c and every_box[4] > y_c:
                        #交界点以1/2的目标框大小为界限
                        if (every_box[1] - every_box[3] > x_c - every_box[3]) and (every_box[4] - every_box[2] > every_box[4] - y_c):
                            if  x_c - every_box[3] < (every_box[1] - every_box[3])/2 or every_box[4] - y_c < (every_box[4] - every_box[2])/2:
                                save_box[3] = every_box[3]
                                save_box[4] = every_box[4]
                                continue
                        
                        if every_box[1] > x_c:
                            save_box[3] = x_c
                        else:
                            save_box[3] = every_box[1]

                        if every_box[2] > y_c:
                            save_box[4] = every_box[2]
                        else:
                            save_box[4] = y_c
                    else:
                        save_box[3] = every_box[3]
                        save_box[4] = every_box[4]

        vaild_targe_add = []
        if labels_add.size:
            add_flag = (target_add[:,3] < x_c) & (target_add[:,4] > y_c) #先选出左上的点粘贴区域的框 保留
            target_add = target_add[add_flag]
            if target_add.shape[0] > 0:
                vaild_targe_add = np.zeros_like(target_add)
                vaild_targe_add[:,0] = target_add[:,0]
                vaild_targe_add[:,3:] = target_add[:,3:]
                #image2 procecs
                for every_box, save_box in zip(target_add, vaild_targe_add):
                    if every_box[1] > x_c or every_box[2] < y_c:
                        if every_box[1] > x_c:
                            save_box[1] = x_c
                        else:
                            save_box[1] = every_box[1]

                        if every_box[2] < y_c:
                            save_box[2] = y_c
                        else:
                            save_box[2] = every_box[2]
                    else:
                        save_box[1] = every_box[1]
                        save_box[2] = every_box[2]

        if len(vaild_targe) and len(vaild_targe_add):
            target_cat = np.concatenate((vaild_targe, vaild_targe_add), 0)
        elif len(vaild_targe) > 0 and len(vaild_targe_add) <= 0:
            target_cat = vaild_targe
        elif len(vaild_targe) <= 0 and len(vaild_targe_add) > 0:
            target_cat = vaild_targe_add
        else:
            return image, boxes, labels
        
        target_cat = self.change2leftpoints(target_cat)
        boxes = target_cat[:, 1:]
        labels = target_cat[:, 0]
        return image, boxes, labels
        

    def left_bottom_pach(self, image, boxes, labels, image_add, boxes_add, labels_add):
        #left_bottom_pach process
        if labels.size:
            assert boxes.shape[0] == labels.shape[0], f"{boxes.shape[0]},{labels.shape[0]}"
            target = np.concatenate((labels[None].T, boxes), axis=1)

        if labels_add.size:
            assert boxes_add.shape[0] == labels_add.shape[0], f"{boxes_add.shape[0]},{labels_add.shape[0]}"
            target_add = np.concatenate((labels_add[None].T, boxes_add), axis=1)

        y_c = random.randint(self.point_range[0][0], self.point_range[0][1])
        x_c = random.randint(self.point_range[1][0], self.point_range[1][1])
        h = self.size[0]
        w = self.size[1]
        # crop_w = w - x_c
        # crop_h = h - y_c
        image[y_c:h, x_c:w] = image_add[y_c:h, x_c:w] #(0,0,0)
        vaild_targe = []
        if labels.size:
            delet_flag = (target[:, 1] > x_c) & (target[:, 2] > y_c) #先选出左上点在删除区域的框 剔掉
            target = target[~delet_flag]
            if target.shape[0] > 0:
                vaild_targe = np.zeros_like(target)
                vaild_targe[:,:3] = target[:,:3]
                #image1 procecs
                for every_box, save_box in zip(target, vaild_targe):
                    if every_box[3] > x_c and every_box[4] > y_c:
                        #交界点以1/2的目标框大小为界限
                        if (every_box[3] - every_box[1] > every_box[3] - x_c) and (every_box[4] - every_box[2] > every_box[4] - y_c):
                            if  every_box[3] - x_c <  (every_box[3] - every_box[1])/2 or every_box[4] - y_c < (every_box[4] - every_box[2])/2:
                                save_box[3] = every_box[3]
                                save_box[4] = every_box[4]
                                continue
                        
                        if every_box[1] > x_c:
                            save_box[3] = every_box[3]
                        else:
                            save_box[3] = x_c
                        if every_box[2] > y_c:
                            save_box[4] = every_box[4]
                        else:
                            save_box[4] = y_c
                    else:
                        save_box[3] = every_box[3]
                        save_box[4] = every_box[4]

        vaild_targe_add = []
        if labels_add.size:
            add_flag = (target_add[:,3] > x_c) & (target_add[:,4] > y_c) #先选出左下的点粘贴区域的框 保留
            target_add = target_add[add_flag]
            if target_add.shape[0] > 0:
                vaild_targe_add = np.zeros_like(target_add)
                vaild_targe_add[:,0] = target_add[:,0]
                vaild_targe_add[:,3:] = target_add[:,3:]
                #image2 procecs
                for every_box, save_box in zip(target_add, vaild_targe_add):
                    if every_box[1] < x_c or every_box[2] < y_c:
                        if every_box[1] < x_c:
                            save_box[1] = x_c
                        else:
                            save_box[1] = every_box[1]

                        if every_box[2] < y_c:
                            save_box[2] = y_c
                        else:
                            save_box[2] = every_box[2]
                    else:
                        save_box[1] = every_box[1]
                        save_box[2] = every_box[2]

        if len(vaild_targe) and len(vaild_targe_add):
            target_cat = np.concatenate((vaild_targe, vaild_targe_add), 0)
        elif len(vaild_targe) > 0 and len(vaild_targe_add) <= 0:
            target_cat = vaild_targe
        elif len(vaild_targe) <= 0 and len(vaild_targe_add) > 0:
            target_cat = vaild_targe_add
        else:
            return image, boxes, labels

        boxes = target_cat[:,1:]
        labels = target_cat[:,0]
        return image, boxes, labels

    def change2leftpoints(self,boxes):
        w_det = boxes[:,1].copy()
        boxes[:,1] = boxes[:,3]
        boxes[:,3] = w_det
        return boxes


    def left_top_pach(self, image, boxes, labels, image_add, boxes_add, labels_add):
        #left_top_pach process
        if labels.size:
            assert boxes.shape[0] == labels.shape[0], f"{boxes.shape[0]},{labels.shape[0]}"
            target = np.concatenate((labels[None].T, boxes), axis=1)

        if labels_add.size:
            assert boxes_add.shape[0] == labels_add.shape[0], f"{boxes_add.shape[0]},{labels_add.shape[0]}"
            target_add = np.concatenate((labels_add[None].T, boxes_add), axis=1)

        y_c = random.randint(self.point_range[0][0], self.point_range[0][1])
        x_c = random.randint(self.point_range[1][0], self.point_range[1][1])
        h = self.size[0]
        w = self.size[1]
        # crop_w = w - x_c
        # crop_h = h - y_c
        image[0:y_c, x_c:w] = image_add[0:y_c, x_c:w] #(0,0,0)
        if labels.size and labels_add.size:      
            target = self.change2leftpoints(target)
            target_add = self.change2leftpoints(target_add)

        vaild_targe = []
        if labels.size:
            delet_flag = (target[:, 3] > x_c) & (target[:, 4] < y_c) #先选出右上点在删除区域的框 剔掉
            target = target[~delet_flag]
            if target.shape[0] > 0:
                vaild_targe = np.zeros_like(target)
                vaild_targe[:,0] = target[:,0]
                vaild_targe[:,3:] = target[:,3:]
                #image1 procecs
                for every_box, save_box in zip(target, vaild_targe):
                    if every_box[1] > x_c and every_box[2] < y_c:
                        #交界点以1/2的目标框大小为界限
                        if (every_box[1] - every_box[3] > every_box[1] - x_c) and (every_box[4] - every_box[2] > y_c - every_box[2]):
                            if  every_box[1] - x_c <  (every_box[1] - every_box[3])/2 or y_c - every_box[2]  < (every_box[4] - every_box[2])/2:
                                save_box[1] = every_box[1]
                                save_box[2] = every_box[2]
                                continue
                        
                        if every_box[3] > x_c:
                            save_box[1] = every_box[1]
                        else:
                            save_box[1] = x_c
                        if every_box[4] > y_c:
                            save_box[2] = y_c
                        else:
                            save_box[2] = every_box[2]
                    else:
                        save_box[1] = every_box[1]
                        save_box[2] = every_box[2]


        vaild_targe_add = []
        if labels_add.size:
            add_flag = (target_add[:,1] > x_c) & (target_add[:,2] < y_c) #先选出左下的点粘贴区域的框 保留
            target_add = target_add[add_flag]
            if target_add.shape[0] > 0:
                vaild_targe_add = np.zeros_like(target_add)
                vaild_targe_add[:,:3] = target_add[:,:3]
                #image2 procecs
                for every_box, save_box in zip(target_add, vaild_targe_add):
                    if every_box[3] < x_c or every_box[4] > y_c:
                        if every_box[3] > x_c:
                            save_box[3] = every_box[3]
                        else:
                            save_box[3] = x_c

                        if every_box[4] > y_c:
                            save_box[4] = y_c
                        else:
                            save_box[4] = every_box[4]
                    else:
                        save_box[3] = every_box[3]
                        save_box[4] = every_box[4]

        if len(vaild_targe) and len(vaild_targe_add):
            target_cat = np.concatenate((vaild_targe, vaild_targe_add), 0)
        elif len(vaild_targe) > 0 and len(vaild_targe_add) <= 0:
            target_cat = vaild_targe
        elif len(vaild_targe) <= 0 and len(vaild_targe_add) > 0:
            target_cat = vaild_targe_add
        else:
            return image, boxes, labels

        target_cat = self.change2leftpoints(target_cat)
        boxes = target_cat[:,1:]
        labels = target_cat[:,0]
        return image, boxes, labels

class Noise(object):
    """
    # sp_noise 椒盐噪点
    # gasuss_noise 高斯噪点
    # motion_blur 运动噪点
    """
    def __init__(self, noise_cfg):#prob, thres=0.001, mean=0, var=0.001
        self.thres = noise_cfg.get('thres', 0.001)
        self.mean = noise_cfg.get('mean', 0)
        self.var = noise_cfg.get('var', 0.001)
        self.prob = noise_cfg.get('prob', 0.1)

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.random() >= 1-self.prob: 
            rand_num = random.randint(3)
            if rand_num == 0:
                image = self.sp_noise(image)
            elif rand_num == 1:
                image = self.gasuss_noise(image)
            else:
                image = self.motion_blur(image)
        return image, boxes, labels, image_mate

    def sp_noise(self, image):
        output = np.zeros(image.shape,np.uint8)
        th = 1 - self.thres
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < self.thres:
                    output[i][j] = 0
                elif rdn > th:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def gasuss_noise(self, image):
        image = np.array(image/255, dtype=float)
        noise = random.normal(self.mean, self.var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        return out
 
    def motion_blur(self, image, degree_max=10, angle_max=40):
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        degree = random.randint(1, degree_max)
        angle = random.randint(1, angle_max)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

class RandomGamma(object):
    """
    # gamma变化 0.- 0.5变亮  1-2为暗
    """
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.random() <= self.prob: 
            return self.adjust_gamma(image),  boxes, labels, image_mate
        else:
            return image,  boxes, labels, image_mate

    def adjust_gamma(self, image):
        gamma = random.uniform(0.2, 2)#曝光度的范围
        gamma_img2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        image = image.clip(0)
        gamma_img2 = np.power(image, gamma)
        cv2.normalize(gamma_img2, gamma_img2, 0, 255, cv2.NORM_MINMAX)
        gamma_img2 = cv2.convertScaleAbs(gamma_img2)
        return gamma_img2.astype(np.float32)

class ElasticTransform(object):
    def __init__(self, prob=0.05):
        self.alpha = 5
        self.sigma = 1.0
        self.alpha_affine = 0.0
        self.interpolation=cv2.INTER_LINEAR
        self.border_mode=cv2.BORDER_REFLECT_101
        self.value=None
        self.random_state=None
        self.approximate=False
        self.same_dxdy=True
        self.prob = prob

    def __call__(self, image, boxes=None, labels=None, image_mate=None):
        if random.random() < self.prob:
            image = self.solution(image)
        return image, boxes, labels, image_mate
    
    def solution(self, image):
        height, width = image.shape[:2]
        # Random affine
        center_square = np.float32((height, width)) // 2
        square_size = min((height, width)) // 3
        alpha = float(self.alpha)
        sigma = float(self.sigma)
        alpha_affine = float(self.alpha_affine)

        pts1 = np.float32(
            [
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size,
            ]
        )
        pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        matrix = cv2.getAffineTransform(pts1, pts2)

        image = cv2.warpAffine(image, M=matrix, dsize=(width, height), flags=self.interpolation, borderMode=self.border_mode, borderValue=self.value)
        #img = warp_fn(img)

        if self.approximate:
            # Approximate computation smooth displacement map with a large enough kernel.
            # On large images (512+) this is approximately 2X times faster
            dx = np.random.rand(height, width).astype(np.float32) * 2 - 1
            cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
            dx *= alpha
            if self.same_dxdy:
                # Speed up even more
                dy = dx
            else:
                dy = np.random.rand(height, width).astype(np.float32) * 2 - 1
                cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
                dy *= alpha
        else:
            dx = np.float32(gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha)
            if self.same_dxdy:
                # Speed up
                dy = dx
            else:
                dy = np.float32(gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma) * alpha)

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        image = cv2.remap(image, map1=map_x, map2=map_y, interpolation=self.interpolation, borderMode=self.border_mode, borderValue=self.value)
        return image



class SSDAugmentation(object):
    def __init__(self, size=300, mean=(127, 127, 127), image_files=None, labels_info=None, scaleup=False, hyp=None, mosaic_border=None, augments=False, norm=False):
        self.mean = mean
        self.size = size
        self.scaleup = scaleup
        self.hyp = hyp
        self.mosaic_border = mosaic_border
        self.image_files = image_files
        self.labels_info = labels_info
        self.augment_train = augments
        # 功能0马赛克
        self.augment0 = Compose([
                Mosaic(img_size = self.size,
                        image_files = self.image_files, 
                        labels_info = self.labels_info,
                        mosaic_cfg = self.hyp['mosaic']),
                #Perspective(self.hyp),
                Resize(self.size)
            ])
        # self.augment = Compose([
        #     LoadImageAndLabel(self.image_files, self.labels_info),
        #     ConvertFromInts(),
        #     ScaleImage(self.size, self.scaleup),# ToAbsoluteCoords(), ToPercentCoords(),#两者的功能一样     
        #     ColorJittering()
        # ])
        # 功能1 基础
        self.augment1 = Compose([
            LoadImageAndLabel(image_files = self.image_files, labels_info = self.labels_info),
            ConvertFromInts(),
            ScaleImage(img_size = self.size, scaleup = self.scaleup),# ToAbsoluteCoords(), ToPercentCoords(),#两者的功能一样                       
            Perspective(perspective_cfg = self.hyp['perspectives']),
            GaussianBlur(gaussianblur_cfg = self.hyp['gaussianblur']),
            Noise(noise_cfg = self.hyp['noise']),
            CutOut(cutout_cfg = self.hyp['cutout']),
            MixUp(image_files = self.image_files, labels_info = self.labels_info, img_size = self.size, scaleup = self.scaleup, mixup_cfg = self.hyp['mixup']),
            #CopyPaste(image_files = self.image_files, labels_info = self.labels_info, img_size = self.size, scaleup = self.scaleup, copypaste_cfg = self.hyp['copypaste']),
            
            #CutMix(self.image_files, self.labels_info,self.size,self.scaleup),
            # PhotometricDistort(),
            # Expand(self.mean),
            # RandomSampleCrop(),
            RandomMirror(),
            # ToPercentCoords(),
            # Resize(self.size),
            # SubtractMeans(self.mean)
        ])
        # 功能2 目标框贴图
        self.augment2 = Compose([
            LoadImageAndLabel(image_files = self.image_files, labels_info = self.labels_info),
            ConvertFromInts(),
            ScaleImage(img_size = self.size, scaleup = self.scaleup),# ToAbsoluteCoords(), ToPercentCoords(),#两者的功能一样                       
            Perspective(perspective_cfg = self.hyp['perspectives']),
            # GaussianBlur(gaussianblur_cfg = self.hyp['gaussianblur']),
            # Noise(noise_cfg = self.hyp['noise']),
            # CutOut(cutout_cfg = self.hyp['cutout']),
            # MixUp(image_files = self.image_files, labels_info = self.labels_info, img_size = self.size, scaleup = self.scaleup, mixup_cfg = self.hyp['mixup']),
            CopyPaste(image_files = self.image_files, labels_info = self.labels_info, img_size = self.size, scaleup = self.scaleup, copypaste_cfg = self.hyp['copypaste']),
            RandomMirror(),
        ])
        # 功能3 图片贴图
        self.augment3 = Compose([
            LoadImageAndLabel(image_files = self.image_files, labels_info = self.labels_info),
            ConvertFromInts(),
            ScaleImage(img_size = self.size, scaleup = self.scaleup),# ToAbsoluteCoords(), ToPercentCoords(),#两者的功能一样                       
            Perspective(perspective_cfg = self.hyp['perspectives']),
            HistEqualize(histequal_cfg = self.hyp['histequal']),
            # GaussianBlur(gaussianblur_cfg = self.hyp['gaussianblur']),
            # Noise(noise_cfg = self.hyp['noise']),
            # CutOut(cutout_cfg = self.hyp['cutout']),
            # MixUp(image_files = self.image_files, labels_info = self.labels_info, img_size = self.size, scaleup = self.scaleup, mixup_cfg = self.hyp['mixup']),
            #CopyPaste(image_files = self.image_files, labels_info = self.labels_info, img_size = self.size, scaleup = self.scaleup, copypaste_cfg = self.hyp['copypaste']),
            CutMix(image_files = self.image_files, labels_info = self.labels_info, img_size = self.size, scaleup = self.scaleup, cutmix_cfg=self.hyp['cutmix']),
            # PhotometricDistort(),
            # Expand(self.mean),
            # RandomSampleCrop(),
            RandomMirror()
            # ToPercentCoords(),
            # Resize(self.size),
            # SubtractMeans(self.mean)
        ])
        # 功能4 图像质量【直方图均衡、随机光线变化、】
        self.augment4 = Compose([
            LoadImageAndLabel(image_files = self.image_files, labels_info = self.labels_info),
            ConvertFromInts(),
            ScaleImage(img_size = self.size, scaleup = self.scaleup),# ToAbsoluteCoords(), ToPercentCoords(),#两者的功能一样                       
            Perspective(perspective_cfg = self.hyp['perspectives']),
            CutMix(image_files = self.image_files, labels_info = self.labels_info, img_size = self.size, scaleup = self.scaleup, cutmix_cfg=self.hyp['cutmix']),
            HistEqualize(histequal_cfg = self.hyp['histequal']),
            RandomBrightness(),
            RandomGamma(),
            RandomLightingNoise(),
            RandomSaturation(),
            RandomMirror()])
        
        # 功能5 图像干扰【模糊、噪点、弹性变换、（烟雾）】
        self.augment5 = Compose([
            LoadImageAndLabel(image_files = self.image_files, labels_info = self.labels_info),
            ConvertFromInts(),
            ScaleImage(img_size = self.size, scaleup = self.scaleup),# ToAbsoluteCoords(), ToPercentCoords(),#两者的功能一样                       
            Perspective(perspective_cfg = self.hyp['perspectives']),
            GaussianBlur(gaussianblur_cfg = self.hyp['gaussianblur']),
            Noise(noise_cfg = self.hyp['noise']),
            ElasticTransform(),
            RandomMirror(),
        ])

        # 功能5 扣图
        self.augment6 = Compose([
            LoadImageAndLabel(image_files = self.image_files, labels_info = self.labels_info),
            ConvertFromInts(),
            ScaleImage(img_size = self.size, scaleup = self.scaleup),# ToAbsoluteCoords(), ToPercentCoords(),#两者的功能一样                       
            Perspective(perspective_cfg = self.hyp['perspectives']),
            GaussianBlur(gaussianblur_cfg = self.hyp['gaussianblur']),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            Resize(self.size),
        ])

        self.augment_normal = Compose([
                LoadImageAndLabel(image_files = self.image_files, labels_info = self.labels_info),
                ConvertFromInts(),
                ScaleImage(img_size = self.size, scaleup = self.scaleup),
                Perspective(perspective_cfg = self.hyp['perspectives_norm']),
                RandomMirror(),
                ])

        self.augment_base = Compose([
                LoadImageAndLabel(image_files = self.image_files, labels_info = self.labels_info),
                ConvertFromInts(),
                ScaleImage(img_size = self.size, scaleup = self.scaleup)])

        if self.augment_train:
            if norm:
                self.augment = [self.augment_normal]
            else:
                self.augment = [self.augment0, self.augment1, self.augment2, self.augment3, self.augment4, self.augment5, self.augment6]#
        else:
            self.augment = [self.augment_base]

    def __call__(self, index):#img, boxes, labels
        num = random.randint(len(self.augment))
        return self.augment[num](index)
        # return self.augment(index)


