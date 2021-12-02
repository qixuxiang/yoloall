# YOLOALL-PyTorch

YOLOALL-PyTorch是一阶段YOLO系列算法的目标检测开发套件的集合，旨在帮助开发者更快更好的完成检测模型的训练、模型转换及部署到相关平台的全部流程。YOLOALL-PyTorch包含yolov3、yolov4、darknet、yolov5、yolof、mmdetection—yolov3等多种轻量化目标检测算法，其中包括各种模型、数据增强方法、损失函数等。后续会持续加入新的轻量化算法及新的想法。

# 简介

### **代码更新记录**：

- 2021/11/04完成yolov3、yolov4、yolov5代码的集成，代码未进行优化。

- 2021/12/02完成yolo-mmdet版本的代码集成(增加了梯度裁剪功能)

  

### **支持Backone类型**：

- yaml
- py
- cfg

### 数据增强方法：

- Mosaic
- Mixup
- Resize
- LetterBox
- RandomCrop
- RandomFlip
- RandomHSV
- RandomBlur
- RandomNoise
- RandomAffine
- RandomTranslation
- Normalize
- ImageToTensor

### 损失函数：

- bbox loss (IOU,GIOU,DIOU,CIOU) [需要添加]
- confidence loss(YOLOv4,YOLOv5,PP-YOLO) [需要添加]
- IOU_Aware_Loss(PP-YOLO) [需要添加]
- FocalLoss [需要添加]

### 训练方法：

- 指数移动平均 [需要添加]

- 预热 

- 梯度剪切 [需要添加]

- 梯度累计更新 [需要添加]

- 多尺度训练 [需要添加]

- 学习率调整：Fixed，Step，Exp，Poly，Inv，Consine [需要添加]

- Label Smooth

### 模型组键扩展：

-  **Group Norm**
-  **Modulated Deformable Convolution**
-  **Focus**
-  **Spatial Pyramid Pooling**
-  **FPN-PAN**
-  **coord conv**
- **drop block**
- **SAM**

### 模型在coco上效果

略

### 预训练模型

略

### 模型训练

略

# 参考

略

# 感谢

略

