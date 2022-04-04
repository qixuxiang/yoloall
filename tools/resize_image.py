# coding=utf-8
import os
import cv2
import math
import glob
import tqdm
import random
import argparse
import numpy as np
import multiprocessing

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--nworks", type=int, default=1)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--src", type=str, default='')
    parser.add_argument("--dst", type=str, default='')
    opt = parser.parse_args()

    return opt


def worker(return_dict, num, one, lines_one):

    result = []

    for n, line in enumerate(lines_one):
        print(str(n)+'/'+str(one), end='\r')

        line = line.strip()
        pic = line

        if pic.split('.')[-1].lower() not in img_formats:
            continue

        img = cv2.imread(pic)

        h, w = img.shape[:2]

        if w > 3000:
            # print(pic)
            h = h // 3
            w = w // 3
            img = cv2.resize(img, (w , h))
            cv2.imwrite(pic, img)

        elif w >=1920 :
            # print(pic)
            h = h // 2
            w = w // 2
            img = cv2.resize(img, (w , h))
            cv2.imwrite(pic, img)

        
        result.append([])

    return_dict[num] = result


if __name__ == '__main__':

    opt = get_args()
    opt.src = opt.src.strip()
    opt.dst = opt.dst.strip()

    # find file
    if os.path.isfile(opt.src):
        lines = open(opt.src, 'r').readlines()
    else:
        cmd = f'find {opt.src} -name "*.*" > tmp_lines.txt'
        os.system(cmd)
        lines = open('tmp_lines.txt', 'r').readlines()
        os.remove('tmp_lines.txt')

    # shuffle
    if opt.shuffle:
        random.shuffle(lines)

    # worker lines
    one = math.ceil(1.0 * len(lines) / opt.nworks)

    # worker return 
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # run
    job = []
    for i in range(opt.nworks):
        if i == opt.nworks - 1:
            p = multiprocessing.Process(target=worker, args=(return_dict, i, one, lines[one * i:]))
            job.append(p)
            p.start()
            continue
        p = multiprocessing.Process(target=worker, args=(return_dict, i, one, lines[one * i:one * (i + 1)]))
        job.append(p)
        p.start()

    for p in job:
        p.join()

    for idx in return_dict:
        return_dict[idx]