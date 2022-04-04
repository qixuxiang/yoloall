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
import logging

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv', 'dav']  # acceptable video suffixes

logging.basicConfig(
    format="%(message)s",
    filemode = 'w',
    filename="/home/yu/workspace/yoloall/yoloall/tools/video2image_log/cut_video_20220330.txt",
    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--nworks", type=int, default=1)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--resize", type=int, default=1)

    parser.add_argument("--src", type=str, default='') #')
    parser.add_argument("--dst", type=str, default='') #保存路径
    opt = parser.parse_args()

    return opt


def worker(opt, return_dict, num, one, lines_one):

    result = []

    for n, line in enumerate(lines_one):
        # print(str(n)+'/'+str(one), end='\r')

        line = line.strip()
        video_path = line

        if video_path.split('.')[-1].lower() not in vid_formats:
            continue


        if not os.path.exists(video_path):
            print(video_path,' not find')
            continue

        # print(video_path)
        video_dst = os.path.join(opt.dst, '/'.join(video_path.split('/')[10:]))
        video_dst = video_dst[:-4]

        os.makedirs(video_dst, exist_ok=True)

        capture = cv2.VideoCapture(video_path)
        cap_width = int(capture.get(3))
        cap_height = int(capture.get(4))

        framerate = int(capture.get(5))
        framenum = int(capture.get(7))

        video_len = 1.0 * framenum / framerate /60 #min

        #print(framenum)
        print(video_len)
        middle_number = random.randint(3,8)
        small_number = random.randint(3,8)
        #sp_number = random.randint(5,8)
        tin_number = random.randint(1,3)
        if video_len > 10:
            skip = middle_number
        elif video_len > 3:
            skip = small_number
        # elif video_len > 0.5:
        #     skip = sp_number
        else:
            skip = tin_number
        print("vedio:{} | image_len:{:2f} | save image nums:{}".format(video_path,video_len,int(framenum / (framerate*skip))))
        logger.info("vedio:{} | image_len:{:2f} | save image nums:{}".format(video_path,video_len,int(framenum / (framerate*skip))))
        pos = 0
        while pos < framenum:
            #print(pos)
            ret, image = capture.read()
            if not ret:
                break
            
            pos += framerate*skip
            capture.set(cv2.CAP_PROP_POS_FRAMES, pos)

            if opt.resize:
                if cap_width > 3000:
                    height = cap_height // 3
                    width = cap_width // 3
                    image = cv2.resize(image, (width , height))

                elif cap_width >=1920 :
                    height = cap_height // 2
                    width = cap_width // 2
                    image = cv2.resize(image, (width , height))

            cv2.imwrite(os.path.join(video_dst, '{:07}.jpg'.format(pos)), image)

            # if pos > 500 * framerate*skip:
            #     break

        
        result.append([])

    return_dict[num] = result


if __name__ == '__main__':


    opt = get_args()
    opt.src = opt.src.strip()
    opt.dst = opt.dst.strip()
    # print(opt)

    assert opt.dst,'{} path erro!'.format(opt.dst)
    os.makedirs(opt.dst, exist_ok=True)
    
    # # find file
    if os.path.isfile(opt.src):
        lines = open(opt.src,'r').read().splitlines()
    else:
        # lines = glob.glob(os.path.join(opt.src.strip(), '*.*'))
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
            p = multiprocessing.Process(target=worker, args=(opt, return_dict, i, one, lines[one * i:]))
            job.append(p)
            p.start()
            continue
        p = multiprocessing.Process(target=worker, args=(opt, return_dict, i, one, lines[one * i:one * (i + 1)]))
        job.append(p)
        p.start()

    for p in job:
        p.join()

    for idx in return_dict:
        return_dict[idx]
    
    os.system(f'chmod -R 777 {opt.dst}')