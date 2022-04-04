# coding=utf-8
import os
import math
import glob
import tqdm
import random
import cv2
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif']='SimHei'
data_output = "./output/"
print(matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())
cnames = {
    #无效颜色
    # 'aliceblue':'#F0F8FF','antiquewhite':'#FAEBD7','azure':'#F0FFFF','beige':'#F5F5DC','blanchedalmond':'#FFEBCD','cornsilk':'#FFF8DC','floralwhite':'#FFFAF0','ghostwhite':'#F8F8FF','honeydew':'#F0FFF0',   'ivory':'#FFFFF0','lavender':'#E6E6FA','lavenderblush':'#FFF0F5','lemonchiffon':'#FFFACD','lightcyan':'#E0FFFF','lightgoldenrodyellow': '#FAFAD2','lightyellow':'#FFFFE0','linen':'#FAF0E6','mintcream':'#F5FFFA','mistyrose':'#FFE4E1','oldlace':'#FDF5E6','papayawhip':'#FFEFD5','seashell':'#FFF5EE','snow':'#FFFAFA','wheat':'#F5DEB3','white':'#FFFFFF','whitesmoke':'#F5F5F5',
    #有效颜色  
    'aqua':'#00FFFF','aquamarine':'#7FFFD4','bisque':'#FFE4C4','black':'#000000','blue':'#0000FF','blueviolet':'#8A2BE2','brown':'#A52A2A','burlywood':'#DEB887','cadetblue':'#5F9EA0','chartreuse':'#7FFF00','chocolate':'#D2691E','coral':'#FF7F50','cornflowerblue':'#6495ED','crimson':'#DC143C','cyan':'#00FFFF','darkblue':'#00008B','darkcyan':'#008B8B','darkgoldenrod':'#B8860B','darkgray':'#A9A9A9','darkgreen':'#006400','darkkhaki':'#BDB76B','darkmagenta':'#8B008B','darkolivegreen':'#556B2F','darkorange':'#FF8C00','darkorchid':'#9932CC','darkred':'#8B0000','darksalmon':'#E9967A',
    'darkseagreen':'#8FBC8F','darkslateblue':'#483D8B','darkslategray':'#2F4F4F','darkturquoise':'#00CED1','darkviolet':'#9400D3','deeppink':'#FF1493','deepskyblue':'#00BFFF','dimgray':'#696969','dodgerblue':'#1E90FF','firebrick':'#B22222','forestgreen':'#228B22','fuchsia':'#FF00FF','gainsboro':'#DCDCDC','gold':'#FFD700','goldenrod':'#DAA520','gray':'#808080','green':'#008000','greenyellow':'#ADFF2F','hotpink':'#FF69B4','indianred':'#CD5C5C','indigo':'#4B0082','khaki':'#F0E68C','lawngreen':'#7CFC00','lightblue':'#ADD8E6','lightcoral':'#F08080','lightgreen':'#90EE90',
    'lightgray':'#D3D3D3','lightpink':'#FFB6C1','lightsalmon':'#FFA07A','lightseagreen':'#20B2AA','lightskyblue':'#87CEFA','lightslategray':'#778899','lightsteelblue':'#B0C4DE','lime':'#00FF00','limegreen':'#32CD32','magenta':'#FF00FF','maroon':'#800000','mediumaquamarine':'#66CDAA','mediumblue':'#0000CD','mediumorchid':'#BA55D3','mediumpurple':'#9370DB','mediumseagreen':'#3CB371','mediumslateblue':'#7B68EE','mediumspringgreen':'#00FA9A','mediumturquoise':'#48D1CC','mediumvioletred':'#C71585','midnightblue':'#191970','moccasin':'#FFE4B5','navajowhite':'#FFDEAD','navy':'#000080',
    'olive':'#808000','olivedrab':'#6B8E23','orange':'#FFA500','orangered':'#FF4500','orchid':'#DA70D6','palegoldenrod':'#EEE8AA','palegreen':'#98FB98','paleturquoise':'#AFEEEE','palevioletred':'#DB7093','peachpuff':'#FFDAB9','peru':'#CD853F','pink':'#FFC0CB','plum':'#DDA0DD','powderblue':'#B0E0E6','purple':'#800080','red':'#FF0000','rosybrown':'#BC8F8F','royalblue':'#4169E1','saddlebrown':'#8B4513','salmon':'#FA8072','sandybrown':'#FAA460','seagreen':'#2E8B57','sienna':'#A0522D','silver':'#C0C0C0','skyblue':'#87CEEB','slateblue':'#6A5ACD','slategray':'#708090',
    'springgreen':'#00FF7F','steelblue':'#4682B4','tan':'#D2B48C','teal':'#008080','thistle':'#D8BFD8','tomato':'#FF6347','turquoise':'#40E0D0','violet':'#EE82EE','yellow':'#FFFF00','yellowgreen':'#9ACD32'
    }

dic_data = {'person': '0', 'bicycle': '1', 'car': '2', 'motorcycle': '3', 'airplane': '4', 'bus': '5', 'train': '6', 'truck': '7', 'boat': '8', 'traffic light': '9', 
            'fire hydrant': '10', 'stop sign': '11', 'parking meter': '12', 'bench': '13', 'bird': '14', 'cat': '15', 'dog': '16', 'horse': '17', 'sheep': '18', 'cow': '19', 
            'elephant': '20', 'bear': '21', 'zebra': '22', 'giraffe': '23', 'backpack': '24', 'umbrella': '25', 'handbag': '26', 'tie': '27', 'suitcase': '28', 'frisbee': '29', 
            'skis': '30', 'snowboard': '31', 'sports ball': '32', 'kite': '33', 'baseball bat': '34', 'baseball glove': '35', 'skateboard': '36', 'surfboard': '37', 'tennis racket': '38', 'bottle': '39', 
            'wine glass': '40', 'cup': '41', 'fork': '42', 'knife': '43', 'spoon': '44', 'bowl': '45', 'banana': '46', 'apple': '47', 'sandwich': '48', 'orange': '49', 
            'broccoli': '50', 'carrot': '51', 'hot dog': '52', 'pizza': '53', 'donut': '54', 'cake': '55', 'chair': '56', 'couch': '57', 'potted plant': '58', 'bed': '59', 
            'dining table': '60', 'toilet': '61', 'tv': '62', 'laptop': '63', 'mouse': '64', 'remote': '65', 'keyboard': '66', 'cell phone': '67', 'microwave': '68', 'oven': '69', 
            'toaster': '70', 'sink': '71', 'refrigerator': '72', 'book': '73', 'clock': '74', 'vase': '75', 'scissors': '76', 'teddy bear': '77', 'hair drier': '78', 'toothbrush': '79'}


ch_name = {'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车', 'airplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车', 'boat': '船', 
          'traffic light': '红绿灯', 'fire hydrant': '消防栓', 'stop sign': '停止标志', 'parking meter': '停车收费表', 'bench': '长凳', 'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 
          'sheep': '羊', 'cow': '牛', 'elephant': '象', 'bear': '熊', 'zebra': '斑马', 'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞', 'handbag': '手提包', 'tie': '领带', 
          'suitcase': '手提箱', 'frisbee': '飞盘', 'skis': '滑雪板', 'snowboard': '单板滑雪', 'sports ball': '运动球', 'kite': '风筝', 'baseball bat': '棒球棒', 'baseball glove': '棒球手套', 
          'skateboard': '滑板', 'surfboard': '冲浪板', 'tennis racket': '网球拍', 'bottle': '瓶子', 'wine glass': '红酒杯', 'cup': '杯子', 'fork': '叉子', 'knife': '刀', 'spoon': '勺', 
          'bowl': '碗', 'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子', 'broccoli': '西兰花', 'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '比萨', 'donut': '甜甜圈', 
          'cake': '蛋糕', 'chair': '椅子', 'couch': '长椅', 'potted plant': '盆栽', 'bed': '床', 'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视', 'laptop': '笔记本电脑', 'mouse': '鼠标', 
          'remote': '遥控器', 'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉', 'oven': '烤箱', 'toaster': '烤面包机', 'sink': '洗碗槽', 'refrigerator': '冰箱', 'book': '书', 
          'clock': '时钟', 'vase': '花瓶', 'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机', 'toothbrush': '牙刷'}

image_wide = 640
image_high = 640
usefull_list = [1,2,3,4]#i for i in range(len(dic_data))
#usefull_list = [52,53,121, 122, 123, 125]
# class_dic = {47:0,48:0,50:0,51:0,57:0,118:0,119:0,120:0,121:0,122:0,123:0,124:0,125:0,126:0,127:0,52:0,53:0,128:0,129:0,130:0}
label_enhance = 1 #数据扩充标志位
colors = [list(cnames.keys())[i] for i in random.sample([i for i in range(len(cnames))],len(usefull_list))]
chose_list = []

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nworks", type=int, default=1)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--src", type=str, default='/home/yu/data/dataset/coco/val_label.txt')
    parser.add_argument("--dst", type=str, default='')
    opt = parser.parse_args()
    return opt


def worker(return_dict, num, one, lines_one):

    result = {}
    counts = []
    erro = []
    FP_pic = []
    backgroud_pic = []
    number = 0
    label_count = []
    chosce_class = []
    
    for n, line in enumerate(lines_one):
        print(str(n)+'/'+str(one), end='\r')
        line = line.strip()
        
        if ";" in line:
            pic, txt = line.split(';')
        else:
            txt = line
        #print(txt)
        lines = open(txt,'r').read().splitlines()
        l = np.array([i.split() for i in lines if all(i.split(' '))],dtype=np.float32)
        if len(l) == 0:
            FP_pic.append(line)
            continue
        #过滤小于10像素的目标框
        l= l[(l[:,3]*image_wide >= 5) & (l[:,4]*image_high >= 5)]
        #仅仅统计usefull_list中的类别:
        l= l[[int(cls_t) in usefull_list for cls_t in l[:,0]]] 

        flg = 0
        for res in l[:,0]:
            if res in usefull_list:
                flg = 1
                break
        if flg == 0:
            backgroud_pic.append(line)
        # try:
        #     data = np.loadtxt(txt).reshape(-1, 5)
        # except:
        #     import pdb
        #     pdb.set_trace()
        for d in l:
            cat = int(d[0])
            # if cat == 36 or cat == 49 or cat == 56:
            #     erro.append(line)
            result[cat] = result.get(cat, 0) + 1
        #提取某个类别
        if len(chose_list) > 0:
            commonEle=[val for val in l[:,0] if val in chose_list]
            if len(commonEle) > 0:
                chosce_class.append(line)

        #数据配比double -> 只要有用样本，过滤掉无用背景图片
        if flg == 1 and label_enhance:#只含有有用样本
            label_count.append(line)#先对增加的数据进行统计

            x_count = l[:,0].tolist()
            #设置规则扩充数据
        counts.append(list(set(sorted(l[:,0]))))
        number += 1
    return_dict[num] = (result,counts,erro,FP_pic,backgroud_pic,number,label_count,chosce_class)

def display(L):
    for path in L:
        if ";" in path:
            image_path,all_path = path.split(';')
        else:
            image_path = path.replace('/labels_dir/','/')[:-4]
            all_path = path
        print(image_path)
        img = cv2.imread(image_path)
        h,w = img.shape[0],img.shape[1]

        #all_path = path
        #all_path = path+'.txt'
        print(all_path)
        assert os.path.exists(all_path),'{} path erro!'.format(all_path)
        with open(all_path,'r')as f:
            lines = f.readlines()
            for string in lines:
                info = string.strip().split(' ')
                type_name = info[0]
                if type_name not in ['36','49','56']:
                    continue
                else:
                    center_w = float(info[3]) * w
                    center_h = float(info[4]) * h
                    x1 = int(float(info[1]) * w - center_w/2)
                    y1 = int(float(info[2]) * h - center_h/2)
                    x2 = int(float(info[1]) * w + center_w/2)
                    y2 = int(float(info[2]) * h + center_h/2)
                    cv2.putText(img, '{}'.format(type_name), (x1,y1-3), 0, float(6) / 4, [0,255,0], thickness=4, lineType=cv2.FONT_HERSHEY_SIMPLEX)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imshow('img', img)
        k = cv2.waitKey(0)
        if k == ord('q'):# or k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    opt = get_args()
    opt.src = opt.src.strip()
    opt.dst = opt.dst.strip()

    # find file
    if os.path.isfile(opt.src):
        lines = open(opt.src, 'r').readlines()
    else:
        cmd = f'find {opt.src} -name "*.txt" > tmp_lines.txt'
        os.system(cmd)
        lines = open('tmp_lines.txt', 'r').readlines()
        os.remove('tmp_lines.txt')

    # mkdir
    if not os.path.exists(data_output):
        os.mkdir(data_output)

    print(len(lines))
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

    dic_change = {}
    dic_num = {}
    for key,value in dic_data.items():
        dic_change[value] = key
        dic_num[key] = 0
    dic_nums = dic_num.copy()


    cat_counts = {}
    pic_counts = []
    erro_count = []
    FP_count = []
    bg_count = []
    total_num = 0
    label_enhance_list = []
    chosce_class_list = []
    for idx in return_dict:
        return_i,return_j,return_e,return_f,return_b,return_num,return_l,return_c = return_dict[idx]
        for cat in return_i:
            cat_counts[cat] = cat_counts.get(cat, 0) + return_i[cat]
        for clss in return_j:
            pic_counts.extend(clss)
        for erro in return_e:
            erro_count.append(erro)
        for fp in return_f:
            FP_count.append(fp)
        for bg in return_b:
            bg_count.append(bg)
        for labels_ in return_l:
            label_enhance_list.append(labels_)
        for chose_ in return_c:
            chosce_class_list.append(chose_)
        total_num += return_num
    
    print(cat_counts)
    for num in list(set(sorted(pic_counts))):
        dic_num[dic_change[str(int(num))]] = pic_counts.count(num)
    
    #将图片张数进行可视化绘制柱状图
    # Histogram_dic = {}
    # usefull_class_name = [dic_change[str(i)] for i in usefull_list]
    # dic_num_list = sorted([dic_num[i] for i in usefull_class_name],reverse=True)
    # for keys,values in dic_num.items():
    #     if keys in usefull_class_name:
    #         Histogram_dic[values] = ch_name[keys]
    # y = dic_num_list
    # x = [Histogram_dic[i] for i in y]
    # plt.figure(figsize=(15,10))
    # plt.bar(x,y,color = colors)
    # plt.xticks(rotation=45,size = 15)
    # for data_x,data_y in zip(x,y):
    #     plt.text(data_x,data_y+10,str(data_y),ha='center',va= 'bottom',fontsize=12)
    # plt.savefig('./result.png',dpi=600)
    # plt.show()

    print('>>>>>>>>>>>>>>>>>picture numbers>>>>>>>>>>>>>>>>>')
    print(dic_num)
    print('>>>>>>>>>>>>>>>>>>>>负样本数量>>>>>>>>>>>>>>>>>>>>>>>')
    print(len(FP_count))
    with open(data_output + '负样本.txt','w')as f:
        for i in FP_count:
            f.writelines(i+'\n')

    print('>>>>>>>>>>>>>>>>>>>>背景样本数量>>>>>>>>>>>>>>>>>>>>>>>')
    print(len(bg_count))
    with open(data_output + '负样本.txt','w')as f:
        for i in bg_count:
            f.writelines(i+'\n')

    print('>>>>>>>>>>>>>>>>>>>>样本数量>>>>>>>>>>>>>>>>>>>>>>>')
    print(total_num)

    print('>>>>>>>>>>>>>>>>>>>>样本扩充>>>>>>>>>>>>>>>>>>>>>>>')
    print(len(label_enhance_list))
    with open(data_output + 'datas.txt','w')as f:
        for i in label_enhance_list:
            f.writelines(i+'\n')

    print('>>>>>>>>>>>>>>>>>>>选择保留的样本数量>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(len(chosce_class_list))
    with open(data_output + 'chose_save.txt','w')as f:
        for i in chosce_class_list:
            f.writelines(i+'\n')


    for key,value in cat_counts.items():
        dic_nums[dic_change[str(key)]] = value
    

    # dic_nums['GarbageCan'] = dic_nums['GarbageCan_Overflow'] + dic_nums['GarbageCan_NotFull']
    # dic_nums['FP'] = len(FP_count)
    # dic_nums['BG'] = len(bg_count)
    # dic_nums['TP'] = len(lines) - len(FP_count) - len(bg_count)
    # print()
    # print('>>>>>>>>>>>>>>>>>picture class numbers>>>>>>>>>>>>>>>>>')
    # print(dic_nums)

    #将类别框数进行可视化绘制柱状图
    Histogram_dic = {}
    usefull_class_name = [dic_change[str(i)] for i in usefull_list]
    dic_num_list = sorted([dic_nums[i] for i in usefull_class_name],reverse=True)
    for keys,values in dic_nums.items():
        if keys in usefull_class_name:
            Histogram_dic[values] = ch_name[keys]
    y = dic_num_list
    x = [Histogram_dic[i] for i in y]
    plt.figure(figsize=(15,10))
    plt.bar(x,y,color = colors)
    plt.xticks(rotation=45,size = 15)
    for data_x,data_y in zip(x,y):
        plt.text(data_x,data_y+10,str(data_y),ha='center',va= 'bottom',fontsize=12)
    plt.savefig(data_output + 'result1.png',dpi=600)

    print()
    print('>>>>>>>>>>>>>>>>>问题图片>>>>>>>>>>>>>>>>>')
    print("错误图片数量：",len(erro_count))
    #print(erro_count)
    #错误图片进行显示
    if len(erro_count) > 0:
        display(erro_count)
    # with open('./erro.txt','w',encoding='UTF-8')as f:
    #     for i in erro_count:
    #         f.writelines(i+'\n')
    # if len(bg_count) > 0: #查看背景图片
    #     display(bg_count)
    list_key = []
    list_value = []
    save_count_data = pd.DataFrame()
    cloum = 0
    for key,value in dic_nums.items():
        if value > 0:
            temp = [ch_name[key],value]
            save_count_data[cloum] = pd.Series(temp)
            cloum += 1

    save_count_data.to_csv(data_output + 'count_20211022.csv',index=False,encoding='gbk')
    print('Done!')

