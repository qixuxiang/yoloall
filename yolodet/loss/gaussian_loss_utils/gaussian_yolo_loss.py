# Loss functions
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import cv2
from yolodet.utils.general import bbox_iou,box_iou_2
from yolodet.utils.torch_utils import is_parallel
import copy
from yolodet.loss.mmdet_loss_utils import CrossEntropyLoss #, MSELoss, CIoULoss
import torch.nn.functional as F

def box_iou_graph(b1, b2):
    """
    Return iou tensor

    Args:
        b1 (tensor): (fh, fw, num_anchors_this_layer, 4)
        b2 (tensor): (num_gt_boxes, 4)

    Returns:
        iou (tensor): shape=(num_b1_boxes, num_b2_boxes)
    """
    # Expand dim to apply broadcasting.
    # (fh, fw, num_anchors_this_layer, 1, 4)
    b1 = b1.unsqueeze(-2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    # (1, num_gt_boxes, 4)
    b2 = b2.unsqueeze(0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # (fh, fw, num_anchors_this_layer, num_b2_boxes, 2)
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes - intersect_mins))
    # (fh, fw, num_anchors_this_layer, num_b2_boxes)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # (fh, fw, num_anchors_this_layer, 1)
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    # (1, num_gt_boxes)
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def nll_loss(x, mu, sigma, sigma_const=0.3):
    pi = np.pi
    Z = (2 * pi * (sigma + sigma_const) ** 2) ** 0.5 # （2 * pi * (sigma + 0.3)**2)**0.5
    probability_density = torch.exp(-0.5 * (x - mu) ** 2 / ((sigma + sigma_const) ** 2)) / Z
    nll = -torch.log(probability_density + 1e-7)
    return nll


def y_pred_graph(raw_y_pred, anchors, input_shape, device):
    num_anchors_this_layer = len(anchors)
    anchors_tensor = anchors.reshape([1, 1, 1, num_anchors_this_layer, 2])
    grid_shape = torch.tensor(raw_y_pred.shape[1:3]).tolist()
    grid_y, grid_x = torch.meshgrid([torch.arange(grid_shape[0]), torch.arange(grid_shape[1])])
    grid = torch.stack((grid_x, grid_y), 2).view((1, grid_shape[0], grid_shape[1], 1, 2)).float().to(device)

    y_pred_xy = (raw_y_pred[..., :2].sigmoid() + grid) / torch.tensor(grid_shape[::-1]).to(device)
    y_pred_wh = raw_y_pred[..., 2:4].exp() * (anchors_tensor / torch.tensor(input_shape.tolist()[::-1]).to(device))
    y_pred_box = torch.cat([y_pred_xy, y_pred_wh],-1)

    y_pred_delta_xy = raw_y_pred[..., :2].sigmoid() #xy的偏移
    y_pred_log_wh = raw_y_pred[..., 2:4] #wh的系数
    y_pred_sigma = raw_y_pred[..., 4:8].sigmoid() #xywh的方差
    y_pred_confidence = raw_y_pred[..., 8:9].sigmoid() #置信度
    y_pred_class_probs = raw_y_pred[..., 9:].sigmoid() #分类

    return grid, y_pred_box, y_pred_delta_xy, y_pred_log_wh, y_pred_sigma, y_pred_confidence, y_pred_class_probs


def compute_gaussian_yolo_loss(self,p,target,imgs):
    object_scale = 2
    noobject_scale = 0.5
    loss_cls = CrossEntropyLoss(use_sigmoid=True,reduction='sum')#1.0
    loss_conf = CrossEntropyLoss(use_sigmoid=True,reduction='sum')#1.0

    targets = target[0]
    img_metas = target[1]
    label_trues = target[-1]
    device = p[0].device
    yolo_outputs = [p[i].permute(0, 2, 3, 1, 4).contiguous() for i in range(len(p))][::-1]
    input_shape = torch.tensor(yolo_outputs[0].shape[1:3]) * self.stride[-1]
    grid_shapes = [torch.tensor(yolo_outputs[i].shape[1:3]) for i in range(len(p))]
    loss = 0
    bbox_loss = 0
    obj_loss = 0
    cls_loss = 0
    batch_size = len(targets)
    anchors = (self.anchors.cpu() * torch.tensor(self.stride).reshape(len(self.stride),1,1)).reshape(-1,2).to(device)
    anchor_masks = np.array([i for i in range(anchors.shape[0])]).reshape(-1, anchors.shape[0] // len(self.stride)).tolist()[::-1]
    raw_y_trues = []
    for level in range(len(targets[0])):
        res = []
        for batch in range(batch_size):
            res.append(torch.tensor(targets[batch][level]).to(device))
        raw_y_trues.append(torch.cat(res,0))
    
    if 0:
        for i in range(batch_size):
            per_image = (imgs[i] * 255.0).cpu().int().numpy().astype(np.int8)
            img = per_image.transpose(1, 2, 0)  # BGR to RGB, to 3x416x416 #[:, :, ::-1].
            img = np.ascontiguousarray(img)
            info = targets[i]
            bboxes = []
            labels = []
            for level_map in range(len(info)):
                object_mask = info[level_map][..., 4:5]
                print(f'stride {level_map*8}: ',object_mask.sum())
                if object_mask.sum() > 0:
                    gt_bboxes = info[level_map][object_mask[...,0]>0][...,:4]
                    gt_label = torch.tensor(info[level_map][object_mask[...,0]>0][...,5:]).max(1)[1]
                    gt_bbox = np.zeros_like(gt_bboxes)
                    gt_bbox[:,:2] = (gt_bboxes[:,:2] - gt_bboxes[:,2:] / 2) * np.array([img.shape[1],img.shape[0]])
                    gt_bbox[:,2:] = (gt_bboxes[:,:2] + gt_bboxes[:,2:] / 2) * np.array([img.shape[1],img.shape[0]])
                    bboxes.append(gt_bbox)
                    labels.extend(gt_label.tolist())
            bboxes = np.concatenate(bboxes,0)

            # labels = label_trues[label_trues[:,0] == i][:, 1:]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # debug_img = copy.deepcopy(img)
            # for target_num in range(len(labels)):
            #     right_point = (int((labels[target_num,1] - labels[target_num,3]/2)*img.shape[1]), int((labels[target_num,2] - labels[target_num,4]/2)*img.shape[0]))
            #     left_point = (int((labels[target_num,1] + labels[target_num,3]/2)*img.shape[1]), int((labels[target_num,2] + labels[target_num,4]/2)*img.shape[0]))
            #     cv2.rectangle(debug_img,right_point,left_point,(255,0,0),3)
            #     cv2.putText(debug_img,str(int(labels[target_num,0])),right_point,font, 1.2, (255, 0, 0), 2)

            for bbox, label in zip(bboxes.tolist(), labels):
                cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),1)
                cv2.putText(img, str(label), (int(bbox[0]),int(bbox[1])), 0, 1, (0,0,255), 1)
            cv2.imshow("pos_bbox",img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                raise Exception


    if False:
        for img_meta, info, idx in zip(img_metas,targets,[i for i in range(batch_size)]):
            label_true = label_trues[label_trues[:,0] == idx][:, 1:]
            img_shape = img_meta['img_shape'] #(h,w)
            pad_shape = img_meta['pad_shape']
            t_pad = (pad_shape[0] - img_shape[0]) // 2 #h
            l_pad = (pad_shape[1] - img_shape[1]) // 2 #w
            print('t_pad: ',t_pad)
            print('l_pad: ',l_pad)
            img = cv2.imread(img_meta['filename'])
            #img = cv2.resize(img,(img_shape[1],img_shape[0]))#
            top, bottom = int(round(t_pad - 0.1)), int(round(t_pad + 0.1))
            left, right = int(round(l_pad - 0.1)), int(round(l_pad + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(127,127,127))  # add border
            # bboxes = []
            # labels = []
            # for level_map in range(len(info)):
            #     object_mask = info[level_map][..., 4:5]
            #     print(f'stride {level_map*8}: ',object_mask.sum())
            #     if object_mask.sum() > 0:
            #         gt_bboxes = info[level_map][object_mask[...,0]>0][...,:4]
            #         gt_label = torch.tensor(info[level_map][object_mask[...,0]>0][...,5:]).max(1)[1]
            #         gt_bbox = np.zeros_like(gt_bboxes)
            #         gt_bbox[:,:2] = (gt_bboxes[:,:2] - gt_bboxes[:,2:] / 2) * np.array([img_shape[1],img_shape[0]])
            #         gt_bbox[:,2:] = (gt_bboxes[:,:2] + gt_bboxes[:,2:] / 2) * np.array([img_shape[1],img_shape[0]])
            #         bboxes.append(gt_bbox)
            #         labels.extend(gt_label.tolist())
            #bboxes = np.concatenate(bboxes,0)
            
            #label_true的
            # labels = [int(i) for i in label_true[:,0].tolist()]
            # gt_bboxes = label_true[:,1:]
            # bboxes = np.zeros_like(gt_bboxes)
            # # bboxes[:,:2] = (gt_bboxes[:,:2] - gt_bboxes[:,2:] / 2) * np.array([img_shape[1],img_shape[0]]) + l_pad
            # # bboxes[:,2:] = (gt_bboxes[:,:2] + gt_bboxes[:,2:] / 2) * np.array([img_shape[1],img_shape[0]]) + t_pad
            # bboxes[:,0] = (gt_bboxes[:,0] - gt_bboxes[:,2] / 2) * img_shape[1] + l_pad
            # bboxes[:,1] = (gt_bboxes[:,1] - gt_bboxes[:,3] / 2) * img_shape[0] + t_pad
            # bboxes[:,2] = (gt_bboxes[:,0] + gt_bboxes[:,2] / 2) * img_shape[1] + l_pad
            # bboxes[:,3] = (gt_bboxes[:,1] + gt_bboxes[:,3] / 2) * img_shape[0] + t_pad

            
            # for bbox, label in zip(bboxes.tolist(), labels):
            #     cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),1)
            #     cv2.putText(img, str(label), (int(bbox[0]),int(bbox[1])), 0, 1, (0,0,255), 1)

            labels = label_true
            font = cv2.FONT_HERSHEY_SIMPLEX
            debug_img = copy.deepcopy(img)
            for target_num in range(len(labels)):
                right_point = (int((labels[target_num,1] - labels[target_num,3]/2)*img_shape[1] + l_pad), int((labels[target_num,2] - labels[target_num,4]/2)*img_shape[0]) + t_pad)
                left_point = (int((labels[target_num,1] + labels[target_num,3]/2)*img_shape[1] + l_pad), int((labels[target_num,2] + labels[target_num,4]/2)*img_shape[0]) + t_pad)
                cv2.rectangle(debug_img,right_point,left_point,(255,0,0),3)
                cv2.putText(debug_img,str(int(labels[target_num,0])),right_point,font, 1.2, (255, 0, 0), 2)

            cv2.imshow("pos_bbox",debug_img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                raise Exception

    for l in range(len(p)):
        grid_shape = grid_shapes[l]
        raw_y_pred = yolo_outputs[l]
        raw_y_true = raw_y_trues[l]
        anchor_mask = anchor_masks[l]
        object_mask = raw_y_true[..., 4:5]
        y_true_class_probs = raw_y_true[..., 5:]
               
        grid, y_pred_box, y_pred_delta_xy, y_pred_log_wh, y_pred_sigma, y_pred_confidence, y_pred_class_probs = \
            y_pred_graph(raw_y_pred, anchors[anchor_mask], input_shape, device)
        
        y_true_delta_xy = raw_y_true[..., :2] * torch.tensor(grid_shapes[l].tolist()[::-1]).to(p[l].device) - grid #计算gt框内x y（中心点）的当前层feature_map上的偏置
        y_true_log_wh = torch.log(raw_y_true[..., 2:4] * torch.tensor(input_shape.tolist()[::-1]).to(device) / anchors[anchor_mask].to(device) + 1e-18) 
        y_true_log_wh[...,0] = object_mask[...,0] * y_true_log_wh[...,0]
        y_true_log_wh[...,1] = object_mask[...,0] * y_true_log_wh[...,1]
        #y_true_log_wh = K.switch(object_mask, y_true_log_wh, K.zeros_like(y_true_log_wh)) #k.switch 根据一个标量值在两个操作之间切换。
        box_loss_scale = 2 - raw_y_true[..., 2:3] * raw_y_true[..., 3:4] #???
        # ignore_mask = tf.TensorArray(K.dtype(raw_y_trues[0]), size=1, dynamic_size=True)
        object_mask_bool = object_mask.bool()
        ignore_mask = torch.zeros_like(object_mask[...,-1]).bool()
        for b in range(batch_size):
            # (num_gt_boxes, 4)
            gt_box = raw_y_true[b, ..., 0:4][object_mask_bool[b,...,0]]
            # (grid_height, grid_width, num_anchors_this_layer, num_gt_boxes)
            iou = box_iou_graph(y_pred_box[b], gt_box)
            # (grid_height, grid_width, num_anchors_this_layer)
            if iou.numel() > 0: #背景样本略过
                best_iou = torch.max(iou, axis=-1)[0]
                ignore_mask[b] = (best_iou < self.hyp['gaussian_thresh']).float()
        # (batch_size, grid_height, grid_width, num_anchors_this_layer)
        ignore_mask = ignore_mask.unsqueeze(-1).float()
        y_true = torch.cat([y_true_delta_xy, y_true_log_wh], axis=-1) #真实x,y的偏移，w,h的系数
        y_pred_mu = torch.cat([y_pred_delta_xy, y_pred_log_wh], axis=-1) #预测的x,y的偏移， w,h的系数
        x_loss = nll_loss(y_true[..., 0:1], y_pred_mu[..., 0:1], y_pred_sigma[..., 0:1]) #y_pred_sigma 为方差
        x_loss = object_mask * box_loss_scale * x_loss
        y_loss = nll_loss(y_true[..., 1:2], y_pred_mu[..., 1:2], y_pred_sigma[..., 1:2]) #x,y,w,h的损失loss
        y_loss = object_mask * box_loss_scale * y_loss
        w_loss = nll_loss(y_true[..., 2:3], y_pred_mu[..., 2:3], y_pred_sigma[..., 2:3])
        w_loss = object_mask * box_loss_scale * w_loss
        h_loss = nll_loss(y_true[..., 3:4], y_pred_mu[..., 3:4], y_pred_sigma[..., 3:4])
        h_loss = object_mask * box_loss_scale * h_loss #框的置信度通过交叉验正熵进行
        pos_and_neg_mask = ignore_mask * noobject_scale + object_mask * object_scale
        confidence_loss = loss_conf(y_pred_confidence, object_mask, weight=pos_and_neg_mask)#, weight=pos_and_neg_mask
        class_loss = loss_cls(y_pred_class_probs, y_true_class_probs, weight=object_mask)#

        # confidence_loss = object_mask * F.binary_cross_entropy_with_logits(object_mask, y_pred_confidence) + (1 - object_mask) * self.BCEobj(object_mask, y_pred_confidence) * ignore_mask
        # class_loss = object_mask * self.BCEcls(y_true_class_probs, y_pred_class_probs) # 分类损失
        x_loss = torch.sum(x_loss) / batch_size
        y_loss = torch.sum(y_loss) / batch_size
        w_loss = torch.sum(w_loss) / batch_size
        h_loss = torch.sum(h_loss) / batch_size
        confidence_loss = torch.sum(confidence_loss) / batch_size
        class_loss = torch.sum(class_loss) / batch_size
        bbox_loss += x_loss + y_loss + w_loss + h_loss
        obj_loss += confidence_loss
        cls_loss += class_loss
        loss += bbox_loss + obj_loss + cls_loss
    return loss, torch.stack((bbox_loss, obj_loss, cls_loss, loss)).detach()
    