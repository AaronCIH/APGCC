import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class AnchorPoints(nn.Module):
    # Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved. 
    # Single Layer Version
    def __init__(self, pyramid_levels=None, stride=None, row=3, line=3):
        super(AnchorPoints, self).__init__()
        self.pyramid_level = pyramid_levels  # default: 3
        if stride is None:
            self.stride = 2 ** self.pyramid_level # default: 8
        else:
            self.stride = stride  # default 8
        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:] # image shape: h,w
        image_shape = np.array(image_shape)
        image_shapes = (image_shape + self.stride - 1) // self.stride # get downsample scale
        # get reference points for each level
        # each anchor block expand the number (row * line) of anchors
        anchor_points = self._generate_anchor_points(self.stride, row=self.row, line=self.line) # control the shift in the anchor block
        # anchor_map
        shifted_anchor_points = self._shift(image_shapes, self.stride, anchor_points)
        # get final anchor_map
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))

    def _generate_anchor_points(self, stride=16, row=3, line=3):
        # generate the reference points in grid layout
        row_step = stride / row
        line_step = stride / line

        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        anchor_points = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()
        return anchor_points
    
    def _shift(self, shape, stride, anchor_points):  # shape is feature map shape
        # shift the meta-anchor to get an acnhor points
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchor_points.shape[0]  # num_of_points
        K = shifts.shape[0]  # num_of_pixel 
        all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
        all_anchor_points = all_anchor_points.reshape((K * A, 2))
        return all_anchor_points

# Basic Decoder Network
class Basic_Decoder_Model(nn.Module):
    def __init__(self, in_planes, num_anchor_points=4, num_classes=2, line=2, row=2, inner_planes=256, feat_layers=[3,4], anchor_stride=None, **kwargs):
        super(Basic_Decoder_Model, self).__init__()
        self.in_planes = in_planes          # default: feat1,2,3,4's out_dims, default: vgg16 = (128, 256, 512, 512)
        self.inner_planes = inner_planes    # default: 256
        self.num_anchor_points = num_anchor_points
        self.num_classes = num_classes      # default: 2
        self.feat_layers = feat_layers      # default: [3,4]

        # build modules
        from .modules import RegressionModel, ClassificationModel, FPN
        self.fpn = FPN(self.in_planes[0], self.in_planes[1], self.in_planes[2], self.in_planes[3], self.inner_planes, self.feat_layers)
        self.offset_head = RegressionModel(num_features_in=self.inner_planes, num_anchor_points=num_anchor_points)
        self.confidence_head = ClassificationModel(num_features_in=self.inner_planes, \
                                                    num_classes=self.num_classes, \
                                                    num_anchor_points=num_anchor_points)
        self.anchor_points = AnchorPoints(pyramid_levels=self.feat_layers[0], stride=anchor_stride, row=row, line=line) # get featmap coords offset

        # aux
        self.aux_en = kwargs['AUX_EN']
        if self.aux_en:
            raise NotImplemented

    def forward(self, samples, features):
        # forward the feature pyramid
        batch_size = features[0].shape[0]
        # FPN will output different deep feature according to feat_layers; P5_x:/16(256), P4_x:/8(256), P3_x:/4(256), P2_x:/2(256)
        features_fpn = self.fpn([features[0], features[1], features[2], features[3]])
        # run the regression and classification branch
        offset = self.offset_head(features_fpn[-1]) * 100     # [b, h/d*w/d*line*row, 2(x,y)]
        confidence = self.confidence_head(features_fpn[-1])   # [b, h/d*w/d*line*row, 2(p/bg)]
        # get sample point map, transfer regression points with offset.
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = offset + anchor_points     # transfer feature coordinate moving to image coordinate. (correspond to head center)
        output_confid = confidence
        out = {'pred_logits': output_confid, 'pred_points': output_coord, 'offset': offset}
        return out

# IFI decoder
class IFI_Decoder_Model(nn.Module):
    def __init__(self, in_planes, feat_layers=[3,4], num_classes=2, num_anchor_points=4,line=2, row=2, anchor_stride=None, inner_planes=256, 
                 sync_bn=False, dilations=(2, 4, 8), require_grad=False, head_layers=[512,256,256], out_type='Normal',
                 pos_dim=24, ultra_pe=False, learn_pe=False, unfold=False, local=False, no_aspp=False, stride=1, **kwargs):
        """
        in_planes: encoder_outplanes; feat_layers: use_encoder_feats; inner_planes: trans_ens_outplanes;
        dilations: only for ASPP; out_type: type of output layer.
        num_anchor_points = line*row
        num_classes = num of classes

        -- For IfI modules:
        # feat_num: # of encoder feats; num_anchor_points: line*row, num of queries each pixel;
        # num_classes: 2 {'regression':2(momentum), 'classifier':confidence},
        # ultra_pe/learn_pe: additional position encoding, pos_dim: additional position encoding dimension
        # local/unfold: impose the local feature information, head_layers: head layers setting, defaults=[512,256,256]
        # feat_dim: input_feats_dims=inner_planes

        -- forward: {'pred_logits', 'pred_points', 'offset'}
        """
        super(IFI_Decoder_Model, self).__init__()
        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm2d
        self.in_planes = in_planes          # feat1,2,3,4's out_dims; default: VGG16, [128, 256, 512, 512]
        self.inner_planes = inner_planes    # default: 256
        self.num_anchor_points = num_anchor_points
        self.num_classes = num_classes      # default: 2

        self.feat_num = len(feat_layers)    # change the encoder feature num.
        self.feat_layers = feat_layers      # control the number of decoder features.
        self.no_aspp = no_aspp
        self.unfold = unfold
        self.out_type = out_type
        self.num_anchor_points = num_anchor_points
        self.num_classes = num_classes
        
        # build modules
        from .modules import ASPP, ifi_simfpn
        # Embedding Decoder.
        if 1 in self.feat_layers:
            self.enc1 = nn.Sequential(nn.Conv2d(self.in_planes[0], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        if 2 in self.feat_layers:
            self.enc2 = nn.Sequential(nn.Conv2d(self.in_planes[1], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        if 3 in self.feat_layers:
            self.enc3 = nn.Sequential(nn.Conv2d(self.in_planes[2], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
        if 4 in self.feat_layers:
            if self.no_aspp:
                self.head = nn.Sequential(nn.Conv2d(self.in_planes[-1], inner_planes, kernel_size=1), norm_layer(inner_planes), nn.ReLU(inplace=True))
            else:
                self.aspp = ASPP(self.in_planes[-1], inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
                self.head = nn.Sequential(
                    nn.Conv2d(self.aspp.get_outplanes(), inner_planes, kernel_size=3, padding=1, dilation=1, bias=False),
                    norm_layer(inner_planes),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1))   
        # IFI Module: position_encoding + regression + classifier
        self.ifi = ifi_simfpn(ultra_pe=ultra_pe, pos_dim=pos_dim, sync_bn=sync_bn, 
                              num_anchor_points=self.num_anchor_points, num_classes=self.num_classes, 
                              local=local, unfold=unfold, stride=stride, learn_pe=learn_pe, 
                              require_grad=require_grad, head_layers=head_layers, feat_num=self.feat_num, feat_dim=inner_planes)
        # Output Decoder.
        if self.out_type == 'Conv':        
            raise NotImplemented
        elif self.out_type == "Deconv":      
            raise NotImplemented

        # Align to real coords
        self.anchor_stride, self.row, self.line = anchor_stride, row, line
        self.anchor_points = AnchorPoints(pyramid_levels=self.feat_layers[0], stride=anchor_stride, row=row, line=line)
        
        # Auxiliary Anchors
        self.aux_en = kwargs['AUX_EN']
        self.aux_number = kwargs['AUX_NUMBER']
        self.aux_range = kwargs['AUX_RANGE']
        self.aux_kwargs = kwargs['AUX_kwargs']
        print("auxiliary anchors: (pos, neg, lambda, kwargs) ", self.aux_number, self.aux_range, self.aux_kwargs)

    def forward(self, samples, features):
        # feats is [feat1, feat2, feat3, feat4]
        # align to max_shape of input_feat by implicit function
        ht, wt = features[self.feat_layers[0]-1].shape[-2], features[self.feat_layers[0]-1].shape[-1]
        batch_size = features[0].shape[0]
        
        # Embedding encoding.
        target_feat = []
        if 1 in self.feat_layers:
            x1 = self.enc1(features[0])
            target_feat.append(x1)
        if 2 in self.feat_layers:
            x2 = self.enc2(features[1])
            target_feat.append(x2)
        if 3 in self.feat_layers:
            x3 = self.enc3(features[2])
            target_feat.append(x3)
        if 4 in self.feat_layers:
            if self.no_aspp:
                aspp_out = self.head(features[-1])
            else:
                aspp_out = self.aspp(features[-1])
                aspp_out = self.head(aspp_out)
            target_feat.append(aspp_out)

        # IFI module forward: position_encoding + offset_head + confidence_head
        context = []
        for i, feat in enumerate(target_feat):
            context.append(self.ifi(feat, size=[ht, wt], level=i+1))
        context = torch.cat(context, dim=-1).permute(0,2,1)
        offset, confidence = self.ifi(context, size=[ht, wt], after_cat=True)

        # output decoder.
        if self.out_type == 'Conv':        
            raise KeyError('{} is not finished'.format(self.out_type))
        elif self.out_type == 'Deconv':
            raise KeyError('{} is not finished'.format(self.out_type))

        # Transform output
        offset *= 100   
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)   # get sample point map
        output_coord = offset + anchor_points   # [b, h/d*w/d*line*row, 2(x,y)]  # transfer feature coordinate moving to image coordinate. (correspond to head center)
        output_confid = confidence              # [b, h/d*w/d*line*row, 2(confidence)]
        out = {'pred_logits': output_confid, 'pred_points': output_coord, 'offset': offset} 

        if not self.aux_en or not self.training:
            return out
        else:
            raise NotImplemented                # still refinement, will be announced ASAP
            out['aux'] = None
            return out
