import os 
import pandas as pd
import random
path = "/home/yu/data/kaggle_data/cvpr2020-plant-pathology-master/data/"
img_path = os.path.join(path,'images')
csv_path = os.path.join(path,'train.csv')

data = pd.read_csv(csv_path)

class_type = {'healthy':0, 'multiple_diseases':1, 'rust':2, 'scab':3}
print(data.iloc[0]['healthy'])
train = []
test = []
for num in range(len(data)):
    for key in class_type.keys():
        if data.iloc[num][key] == 1:
            all_path = os.path.join(img_path,data.iloc[num]["image_id"]+'.jpg')
            assert os.path.exists(all_path)
            label_style = all_path + ';' + str(class_type[key]) + '\n'
            if random.random() > 0.1:
                train.append(label_style)
            else:
                test.append(label_style)

with open(path+'train.txt','w')as f:
    for i in train:
        f.writelines(i)


with open(path+'test.txt','w')as f:
    for i in test:
        f.writelines(i)



