#!/usr/bin/env python3
"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""

import torch

model = torch.load("./pvt_v2_b2_li.pth")
# model = torch.load("./pvt_v2_b1.pth")
# model = torch.load("./pvt_v2_b0.pth")
# model = torch.load("./pvt_v2_b2.pth")
# model = torch.load("./pvt_v2_b3.pth")

for layer_ in list(model.keys()):
    if layer_.startswith("patch_embed1.") or layer_.startswith("block1.") or layer_.startswith("norm1.") or layer_.startswith("patch_embed2.") or layer_.startswith("block2.") or layer_.startswith("norm2.") or layer_.startswith("patch_embed3.") or layer_.startswith("block3.") or layer_.startswith("norm3."):
        model['backbone.'+layer_] = model[layer_]
        model.pop(layer_)
        print("replacing {} with {}".format(layer_, 'backbone.'+layer_))
    elif layer_.startswith("patch_embed4.") or layer_.startswith("block4.") or layer_.startswith("norm4."):
        model['roi_heads.'+layer_] = model[layer_]
        model.pop(layer_)
        print("replacing {} with {}".format(layer_, 'roi_heads.'+layer_))

torch.save(model, "pvt_v2_b2_li_C4.pth")
# torch.save(model, "pvt_v2_b1_C4.pth")
# torch.save(model, "pvt_v2_b0_C4.pth")
# torch.save(model, "pvt_v2_b2_C4.pth")
# torch.save(model, "pvt_v2_b3_C4.pth")
print(model.keys())
