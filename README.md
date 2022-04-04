# YOLOALL-PyTorch

YOLOALL-PyTorch不仅是YOLO系列算法的目标检测的集合，同样也在以YOLO的思想对其他一阶段、二阶段不限于anchor_base和anchor_free方法的改写，旨在能够更快更好的完成检测模型的训练、模型转换及部署到相关平台的一体化流程，该项目有以下特点：

- 能够快速完成项目中的检测/识别模型的训练及迭代：其中包括数据可视化分析、数据格式转换、不同模型格式的转换、anchor生成、数据增强、模型转换等
- 快速复现论文中新检测和识别方法或者新的idear：项目中嵌入Timm库、mmdetection目标检测框架一些核心算法进行集成，方便自定义和设计算法
- 对标工程代码，可以更灵活的设计网络、预处理和后处理
- 参数更改均在config的yaml文件中，仅仅需要更改该文件的参数即可

# 简介

### 支持算法

###### 目标检测算法(objectdet)：

| 目标检测算法       | loss-version      | 后处理参数字段      | anchor设置                 |
| ------------------ | ----------------- | ------------------- | -------------------------- |
| yolov3             | yolov3            | yolov3v5            | 根据检测头数量设置         |
| yolov4             | yolov4            | yolov4              | 根据检测头数量设置         |
| yolov5             | yolov5            | yolov3v5            | 根据检测头数量设置         |
| yolov3-mmdet       | yolo-mmdet        | yolommdet           | 根据检测头数量设置         |
| yolov3-gaussian    | yolov3-gaussian   | yolov3gaussian      | 根据检测头数量设置         |
| yolo-multi_head    | multi_head_enable | multi_head          | 根据multi_head参数设置     |
| yolo-poly          | poly_yolo         | ———                 | 一组(单检测头)             |
| yolof              | yolof-mmdet       | yolofmmdet          | 一组(单检测头)             |
| yoloatss           | yolo-atss-mmdet   | yoloatss            | 根据检测头数量设置         |
| yolox              | yolox-mmdet       | yoloxmmdet          | [[8,8], [16,16], [32, 32]] |
| yolo-transformer   | transformer_enabl | transformer_cfg     | 根据检测头数量设置         |
| yolo-roi【二阶段】 | two_stage_enabel  | two_stage           | 根据检测头数量设置         |
| yolo-fcos          | yolo-fcos-mmdet   | yolofcos            | [[8,8], [16,16], [32, 32]] |
| yolo-centernet     | yolo-center-mmdet | yolocenter          | 不需要                     |
| yolo-gfocal_v1     | yolo-gfl-mmdet    | yologfl/use_sigmoid | [[8,8], [16,16], [32, 32]] |
| yolo-gfocal_v2     | yolo-gfl-mmdet    | yologfl/use_sigmoid | [[8,8], [16,16], [32, 32]] |

###### 分类算法支持(classify)：

- 支持Timm中集成的所有目标识别算法，只需要在configs/classify/test.yaml 配置文件中更改data_dir训练和测试路径、model 框架支持模型的名称
- 支持自定义网络，classify/timm/models/torch_models中添加自己的模型，并将模型文件的名字更改至configs/classify/test.yaml中的model名
- 支持自定义数据形式、自持模型剪枝、蒸馏

###### 蒸馏算法(distiller)：

- 支持离线蒸馏、在线蒸馏

- 目前只支持一阶段的蒸馏算法

  | 蒸馏算法 | OD算法                           | 蒸馏方式    |
  | -------- | -------------------------------- | ----------- |
  | FGD      | yolo-mmdet (anchor_base方法)     | feature-map |
  | LD       | yolo-gfl-mmdet (anchor_free方法) | soft-label  |

### **支持Backone类型**：

- 支持yaml、py、cfg相互转换

### 数据增强方法：

- RandomMosaic
- Mixup
- CutMix
- CopyPaste
- ScaleImage
- LetterBox
- HistEqualize
- RandomCrop
- RandomMirror
- RandomHSV
- RandomBlur（blur、medianblur、bilateralFilter、gaussianblur、motion_blur）
- RandomNoise（sp_noise、gasuss_noise）
- RandomLighting（RandomBrightness、RandomGamma、RandomLightingNoise、RandomContrast、RandomSaturation、RandomHue）
- RandomAffine
- Perspective
- ElasticTransform
- ImageToTensor

### 模型转换

- 支持一阶段和二阶段检测模型转caffemodle、prototxt文件

- 支持多模型转换为一个caffemodel和prototxt

  模型转换支持的层操作和算子操作：

  | layer               | torch-operation  | tensor-operation | torchvision_operation |
  | ------------------- | ---------------- | ---------------- | --------------------- |
  | conv2d              | torch.split      | view             | torchvision.roi_pool  |
  | linear              | torch.cat        | mean             | torchvision.roi_align |
  | relu                | torch.max        | __add__          |                       |
  | prelu               | torch.div        | __sub__          |                       |
  | tanh                | torch.ones       | __mul__          |                       |
  | hardtanh            | torch.zeros      | permute          |                       |
  | leaky_relu          | torch.ones_like  | contiguous       |                       |
  | sigmoid             | torch.zeros_like | pow              |                       |
  | softmax             | torch.sigmoid    | sum              |                       |
  | max_pool2d          | torch.tanh       | sqrt             |                       |
  | avg_pool2d          |                  | unsqueeze        |                       |
  | adaptive_avg_pool2d |                  | expand_as        |                       |
  | dropout             |                  | flatten          |                       |
  | threshold           |                  |                  |                       |
  | batch_norm          |                  |                  |                       |
  | instance_norm       |                  |                  |                       |
  | conv_transpose2d    |                  |                  |                       |
  | interpolate         |                  |                  |                       |

  

# 参考

略



