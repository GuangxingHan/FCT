# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Wednesday, September 28, 2022

@author: Guangxing Han
"""
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from .FCT import make_stage
from functools import partial

from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from .fsod_fast_rcnn import FsodFastRCNNOutputLayers

import time
from detectron2.structures import Boxes, Instances

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)

def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

@ROI_HEADS_REGISTRY.register()
class FsodRes5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        self.freeze_roi_feature_extractor = cfg.MODEL.ROI_HEADS.FREEZE_ROI_FEATURE_EXTRACTOR
        self.only_train_norm = cfg.MODEL.ROI_HEADS.ONLY_TRAIN_NORM

        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.branch_embed4, self.patch_embed4, self.block4, self.norm4, out_channels = self._build_res5_block(cfg)
        if self.freeze_roi_feature_extractor:
            self._freeze_roi_feature_extractor()
        if self.only_train_norm:
            self._only_train_norm()

        self.box_predictor = FsodFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if isinstance(self.branch_embed4, nn.Parameter):
            self.branch_embed4.normal_(mean=0.0, std=0.02)
        elif isinstance(self.branch_embed4, nn.Embedding):
            self.branch_embed4.weight.data.normal_(mean=0.0, std=0.02)

    def _freeze_roi_feature_extractor(self):
        self.branch_embed4.eval()
        for param in self.branch_embed4.parameters():
            param.requires_grad = False

        self.patch_embed4.eval()
        for param in self.patch_embed4.parameters():
            param.requires_grad = False

        self.block4.eval()
        for param in self.block4.parameters():
            param.requires_grad = False

        self.norm4.eval()
        for param in self.norm4.parameters():
            param.requires_grad = False

    def _only_train_norm(self):
        for name, param in self.branch_embed4.named_parameters():
            if 'norm' not in name:
                param.requires_grad = False

        for name, param in self.patch_embed4.named_parameters():
            if 'norm' not in name:
                param.requires_grad = False

        for name, param in self.block4.named_parameters():
            if 'norm' not in name:
                param.requires_grad = False

        for name, param in self.norm4.named_parameters():
            if 'norm' not in name:
                param.requires_grad = False

    def _build_res5_block(self, cfg):
        backbone_type = cfg.MODEL.BACKBONE.TYPE
        if backbone_type == "pvt_v2_b2_li":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=True, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b5":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b4":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b3":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b2":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b1":
            branch_embed = nn.Embedding(2, 512)
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b0":
            branch_embed = nn.Embedding(2, 256)
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 256
        else:
            print("do not support backbone type ", backbone_type)
            return None

        return branch_embed, patch_embed, block, norm, out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        B = x.shape[0]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def _shared_roi_transform_mutual(self, box_features, support_box_features):
        x = box_features
        y = support_box_features
        x, H_x, W_x = self.patch_embed4(x)
        y, H_y, W_y = self.patch_embed4(y)

        x_branch_embed = torch.zeros(x.shape[:-1], dtype=torch.long).cuda()
        x = x + self.branch_embed4(x_branch_embed)

        y_branch_embed = torch.ones(y.shape[:-1], dtype=torch.long).cuda()
        y = y + self.branch_embed4(y_branch_embed)

        for blk in self.block4:
            x, y = blk(x, H_x, W_x, y, H_y, W_y)

        x = self.norm4(x)
        B_x = x.shape[0]
        x = x.reshape(B_x, H_x, W_x, -1).permute(0, 3, 1, 2).contiguous()

        y = self.norm4(y)
        B_y = y.shape[0]
        y = y.reshape(B_y, H_y, W_y, -1).permute(0, 3, 1, 2).contiguous()

        return x, y

    def roi_pooling(self, features, boxes):
        box_features = self.pooler(
            [features[f] for f in self.in_features], boxes
        )
        #feature_pooled = box_features.mean(dim=[2, 3], keepdim=True)  # pooled to 1x1

        return box_features #feature_pooled

    def forward(self, images, features, support_box_features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images
        
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.roi_pooling(features, proposal_boxes)

        support_box_features = support_box_features.mean(0, True)
        box_features, support_box_features = self._shared_roi_transform_mutual(box_features, support_box_features)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features, support_box_features)

        return pred_class_logits, pred_proposal_deltas, proposals

    @torch.no_grad()
    def eval_with_support(self, images, query_features_dict, support_proposals_dict, support_box_features_dict):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images
        
        full_proposals_ls = []
        full_scores_ls = []
        full_bboxes_ls = []
        full_cls_ls = []
        cnt = 0
        for cls_id, proposals in support_proposals_dict.items():
            support_box_features = support_box_features_dict[cls_id].mean(0, True)

            proposals_ls = [Instances.cat([proposals[0], ])]
            full_proposals_ls.append(proposals[0])
            proposal_boxes = [x.proposal_boxes for x in proposals_ls]

            box_features = self.roi_pooling(query_features_dict[cls_id], proposal_boxes)
            box_features, support_box_features = self._shared_roi_transform_mutual(box_features, support_box_features)

            pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features, support_box_features)
            full_scores_ls.append(pred_class_logits)
            full_bboxes_ls.append(pred_proposal_deltas)
            full_cls_ls.append(torch.full_like(pred_class_logits[:, 0].unsqueeze(-1), cls_id).to(torch.int8))
            del box_features
            del support_box_features

            cnt += 1
        
        class_logits = torch.cat(full_scores_ls, dim=0)
        proposal_deltas = torch.cat(full_bboxes_ls, dim=0)
        pred_cls = torch.cat(full_cls_ls, dim=0) #.unsqueeze(-1)
        
        predictions = class_logits, proposal_deltas
        proposals = [Instances.cat(full_proposals_ls)]
        pred_instances, _ = self.box_predictor.inference(pred_cls, predictions, proposals)
        pred_instances = self.forward_with_given_boxes(query_features_dict, pred_instances)

        return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances
