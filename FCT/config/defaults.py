"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""

from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER.HEAD_LR_FACTOR = 1.0

# ---------------------------------------------------------------------------- #
# Few shot setting
# ---------------------------------------------------------------------------- #
_C.INPUT.FS = CN()
_C.INPUT.FS.FEW_SHOT = False
_C.INPUT.FS.SUPPORT_WAY = 2
_C.INPUT.FS.SUPPORT_SHOT = 10
_C.INPUT.FS.SUPPORT_EXCLUDE_QUERY = False

# _C.DATASETS.TRAIN_KEEPCLASSES = 'all'
_C.DATASETS.TEST_KEEPCLASSES = ''
_C.DATASETS.TEST_SHOTS = (1,2,3,5,10,30)
_C.DATASETS.SEEDS = 0

_C.MODEL.BACKBONE.TYPE = "pvt_v2_b2_li"
_C.MODEL.BACKBONE.ONLY_TRAIN_NORM = False
_C.MODEL.BACKBONE.TRAIN_BRANCH_EMBED = True
_C.MODEL.RPN.FREEZE_RPN = False
_C.MODEL.ROI_HEADS.FREEZE_ROI_FEATURE_EXTRACTOR = False
_C.MODEL.ROI_HEADS.ONLY_TRAIN_NORM = False

_C.SOLVER.SOLVER_TYPE = "adamw"
