"""
Created on Wednesday, September 28, 2022

This script is a simplified version of the training script in detectron2/tools.

@author: Guangxing Han
"""

import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.registry import Registry

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.roi_heads import ROIHeads, ROI_HEADS_REGISTRY

from .pvt_v2 import make_stage
from functools import partial

@ROI_HEADS_REGISTRY.register()
class PVTROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        patch_embed4: nn.Module,
        block4: nn.Module,
        norm4: nn.Module,
        freeze_roi_feature_extractor: torch.bool,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        # if isinstance(res5, (list, tuple)):
        #     res5 = nn.Sequential(*res5)
        self.patch_embed4 = patch_embed4
        self.block4 = block4
        self.norm4 = norm4
        self.freeze_roi_feature_extractor = freeze_roi_feature_extractor
        if self.freeze_roi_feature_extractor:
            self._freeze_roi_feature_extractor()
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        freeze_roi_feature_extractor = cfg.MODEL.ROI_HEADS.FREEZE_ROI_FEATURE_EXTRACTOR
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["freeze_roi_feature_extractor"] = freeze_roi_feature_extractor
        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods."
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["patch_embed4"], ret["block4"], ret["norm4"], out_channels = cls._build_res5_block(cfg)

        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )
        return ret

    def _freeze_roi_feature_extractor(self):
        self.patch_embed4.eval()
        for param in self.patch_embed4.parameters():
            param.requires_grad = False

        self.block4.eval()
        for param in self.block4.parameters():
            param.requires_grad = False

        self.norm4.eval()
        for param in self.norm4.parameters():
            param.requires_grad = False

    @classmethod
    def _build_res5_block(self, cfg):
        backbone_type = cfg.MODEL.BACKBONE.TYPE
        if backbone_type == "pvt_v2_b2_li":
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=True, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b2":
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b1":
            patch_embed, block, norm = make_stage(
                 i=3,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1], num_stages=3, linear=False, pretrained=None
            )
            out_channels = 512
        elif backbone_type == "pvt_v2_b0":
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

        return patch_embed, block, norm, out_channels

    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        # return self.res5(x)
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        B = x.shape[0]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
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
            feature_list = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(feature_list, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances
