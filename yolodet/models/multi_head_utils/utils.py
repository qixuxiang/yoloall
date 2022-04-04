import torch
from yolodet.models.common_py import Conv, Focus, Catneck, Incept, Detect

def build_detect(self, input_channel, nc, anchors, heads):

    if heads is None:
        self.detect = Detect(nc, anchors, input_channel)
    else:
        if isinstance(heads, list):
            for di, head in enumerate(heads):
                head_num = head['num']
                head_stride_idx = head['head_s']
                head_anchor_mask = head['head_mask']
                nnx_enable = head['nnx_enable'] if 'nnx_enable' in head else [True for _ in range(head_num)]

                for hi in range(head_num):
                    tmp_ch = [input_channel[idx] for idx in head_stride_idx[hi]]
                    tmp_anchors = [anchors[di][idx] for idx in head_anchor_mask[hi]]

                    name = f'detect_{di}_{hi}'
                    self.add_module(name, Detect(nc, tmp_anchors, tmp_ch, nnx_enable=nnx_enable[hi]))
                    m = eval('self.'+ name)
                    m.coupling_init_detect_layers()
        else:
            head_num = heads['num']
            head_stride_idx = heads['head_s']
            head_anchor_mask = heads['head_mask']
            nnx_enable = heads['nnx_enable'] if 'nnx_enable' in heads else [True for _ in range(head_num)]
           

            for hi in range(head_num):
                tmp_ch = [input_channel[idx] for idx in head_stride_idx[hi]]
                tmp_anchors = [anchors[idx] for idx in head_anchor_mask[hi]]
                #detect_nc = len(head_class[hi])

                name = 'detect' + str(hi) #注意这里的每个检测头必须传进nc为总的nc，否则在做one-hot时会出错
                self.add_module(name, Detect(nc, tmp_anchors, tmp_ch, nnx_enable=nnx_enable[hi]))
                m = getattr(self, name)
                m.coupling_init_detect_layers()

def forward_detect(self, hs):

    if self.heads is None:
        return self.detect(hs)
    else:
        if isinstance(self.heads, list):
            preds_multi_data = []
            heads_num = len(self.heads)
            heads_img_batch = hs[0].shape[0] // heads_num
            
            for di, head in enumerate(self.heads):
                head_num = head['num']
                head_stride_idx = head['head_s']

                preds = []
                for hi in range(head_num):
                    detect_layer = getattr(self, f'detect_{di}_{hi}')
                    if not self.training:
                        preds.append(detect_layer([hs[idx] for idx in head_stride_idx[hi]]))
                    else:
                        preds.append(detect_layer([hs[idx][di*heads_img_batch:(di+1)*heads_img_batch] for idx in head_stride_idx[hi]]))
                preds_multi_data.append(preds)
            
            return preds_multi_data
        else:
            head_num = self.heads['num']
            head_stride_idx = self.heads['head_s']

            preds = []
            for hi in range(head_num):
                detect_layer = getattr(self, 'detect' + str(hi))
                preds.append(detect_layer([hs[idx] for idx in head_stride_idx[hi]]))
                
            return preds