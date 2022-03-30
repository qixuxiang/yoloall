# def load_classes(path):
#     with open(path, "r") as fp:
#         names = fp.read().split("\n")[:-1]
#     return names

# if __name__ == '__main__':
#     class_path = 'data/coco/coco.names'
#     class_list = load_classes(class_path)
#     img_path = 'data/coco/images/000000581886.jpg'
#     img = np.array(Image.open(img_path))
#     H, W, C = img.shape
#     label_path = 'data/coco/labels/000000581886.txt'
#     boxes = np.loadtxt(label_path, dtype=np.float).reshape(-1, 5)
#     # xywh to xxyy
#     boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * W
#     boxes[:, 2] = (boxes[:, 2] - boxes[:, 4] / 2) * H
#     boxes[:, 3] *= W
#     boxes[:, 4] *= H
#     fig = plt.figure()
#     ax = fig.subplots(1)
#     for box in boxes:
#         bbox = patches.Rectangle((box[1], box[2]), box[3], box[4], linewidth=2,
#                                  edgecolor='r', facecolor="none")
#         label = class_list[int(box[0])]
#         # Add the bbox to the plot
#         ax.add_patch(bbox)
#         # Add label
#         plt.text(
#             box[1],
#             box[2],
#             s=label,
#             color="white",
#             verticalalignment="top",
#             bbox={"color": 'g', "pad": 0},
#         )
#         ax.imshow(img)
#     plt.show()


import os
path = '/home/yu/data/dataset/coco/images/val2017'
label_dir = os.listdir(path)
image_path = [os.path.join(path,i) for i in label_dir]
print(image_path[0])
with open('val.txt','w',encoding='utf-8')as f:
    for paths in image_path:
        assert os.path.exists(paths),'path erro!'
        f.writelines(paths+'\n')