# import os
# import numpy as np
# path = "./val_data.txt"
# line = open(path).read().splitlines()
# datas = []
# for i in line:
#     labels = i.split(';')[-1]
#     data = np.loadtxt(labels)
#     if len(data) <= 0:
#         continue
#     elif len(data.shape) == 1:
#         datas.append(data[None])
#     else:
#         datas.append(data)
# print(np.stack(datas,-1))
# print(type(datas[0]))

# print(datas[0].shape)
# print(len(datas[1].shape))
# print(datas[1][None].shape)
# xx = np.concatenate(datas,0)
# print(min(xx[:,0]))
# dic = {}
# for i in range(80):
#     dic[i] = [i]
# print(dic)

# import os
# path = "/home/yu/data/dataset/coco/train.txt"
# save_path = "/home/yu/data/dataset/coco/train_label.txt"
# line = open(path).read().splitlines()
# with open(save_path,'w',encoding='utf-8')as f:
#     for i in line:
#         img_path = i.replace('/labels/','/images/').replace('.txt','.jpg')
#         assert os.path.exists(img_path),"{} not exist".format(img_path)
#         string = img_path+';'+i
#         f.writelines(string+"\n")

# import os
# path = "/home/yu/data/dataset/coco/val.txt"
# save_path = "/home/yu/data/dataset/coco/val_label.txt"
# line = open(path).read().splitlines()
# with open(save_path,'w',encoding='utf-8')as f:
#     for i in line:
#         img_path = i.replace('/images/','/labels/').replace('.jpg','.txt')
#         assert os.path.exists(img_path),"{} not exist".format(img_path)
#         string = i+';'+img_path
#         f.writelines(string+"\n")

# import numpy as np
# anchors = [[10,13, 16,30, 33,23],[30,61, 62,45, 59,119],[116,90, 156,198, 373,326]]
# # b = [[(10, 13), (16, 30), (33, 23)],
# #     [(30, 61), (62, 45), (59, 119)],
# #     [(116, 90), (156, 198), (373, 326)]]
# a = np.array(anchors)
# print(a[0])
data_demo = {
    'data_type':"COCO",
    'train_path':'/home/yu/data/dataset/coco/val_label.txt',
    'val_path':'/home/yu/data/dataset/coco/test_label.txt',
    'test_path':'',
    'nc': 80,
    'names':[  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ],
    'cls_map':
    # {
    #     0:[1], 1:[2], 2:[3], 3:[4], 4:[5], 5:[6], 6:[7], 7:[8], 8:[9], 9:[10], 10:[11], 11:[13], 12:[14], 13:[15], 14:[16], 15:[17], 16:[18], 17:[19], 18:[20], 19:[21], 20:[22], 21:[23], 22:[24], 23:[25], 24:[27], 25:[28], 26:[31], 27:[32], 28:[33], 29:[34],
    #     30:[35], 31:[36], 32:[37], 33:[38], 34:[39], 35:[40], 36:[41], 37:[42], 38:[43], 39:[44], 40:[46], 41:[47], 42:[48], 43:[49], 44:[50], 45:[51], 46:[52], 47:[53], 48:[54], 49:[55], 50:[56], 51:[57], 52:[58], 53:[59], 54:[60], 55:[61], 56:[62], 57:[63],
    #     58:[64], 59:[65], 60:[67], 61:[70], 62:[72], 63:[73], 64:[74], 65:[75], 66:[76], 67:[77], 68:[78], 69:[79], 70:[80], 71:[81], 72:[82], 73:[84], 74:[85], 75:[86], 76:[87], 77:[88], 78:[89], 79:[90]
    #     },
    
    {
        0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9], 10: [10], 11: [11], 12: [12], 13: [13], 14: [14], 15: [15], 16: [16], 17: [17], 18: [18], 19: [19], 20: [20], 21: [21], 22: [22], 23: [23], 24: [24], 25: [25], 26: [26], 
        27: [27], 28: [28], 29: [29], 30: [30], 31: [31], 32: [32], 33: [33], 34: [34], 35: [35], 36: [36], 37: [37], 38: [38], 39: [39], 40: [40], 41: [41], 42: [42], 43: [43], 44: [44], 45: [45], 46: [46], 47: [47], 48: [48], 49: [49], 50: [50], 51: [51], 
        52: [52], 53: [53], 54: [54], 55: [55], 56: [56], 57: [57], 58: [58], 59: [59], 60: [60], 61: [61], 62: [62], 63: [63], 64: [64], 65: [65], 66: [66], 67: [67], 68: [68], 69: [69], 70: [70], 71: [71], 72: [72], 73: [73], 74: [74], 75: [75], 76: [76], 
        77: [77], 78: [78], 79: [79]
    },
    'iou_thres': [0.3],
    'nms_thres': [0.3],
}

res = []
for i in range(data_demo['nc']):
    res.extend([[data_demo['names'][i],0.3,data_demo['cls_map'][i]]])
print(res)