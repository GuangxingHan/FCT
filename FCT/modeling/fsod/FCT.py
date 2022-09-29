"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from timm.models.layers import to_2tuple, trunc_normal_
# from mmdet.utils import get_root_logger
# from mmcv.runner import load_checkpoint

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec


def drop_path(x, y, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x, y
    keep_prob = 1 - drop_prob

    shape_x = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor_x = keep_prob + torch.rand(shape_x, dtype=x.dtype, device=x.device)
    random_tensor_x.floor_()  # binarize
    output_x = x.div(keep_prob) * random_tensor_x

    shape_y = (y.shape[0],) + (1,) * (y.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor_y = keep_prob + torch.rand(shape_y, dtype=y.dtype, device=y.device)
    random_tensor_y.floor_()  # binarize
    output_y = y.div(keep_prob) * random_tensor_y
    return output_x, output_y


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x[0], x[1], self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H_x, W_x, y, H_y, W_y):
        B_x, N_x, C_x = x.shape
        B_y, N_y, C_y = y.shape
        assert B_x == 1 or B_y == 1
        # assert B_y == 1
        q_x = self.q(x).reshape(B_x, N_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        q_y = self.q(y).reshape(B_y, N_y, self.num_heads, C_y // self.num_heads).permute(0, 2, 1, 3)
        # print("q_x.shape={}, q_y.shape={}".format(q_x.shape, q_y.shape))

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B_x, C_x, H_x, W_x)
                x_ = self.sr(x_).reshape(B_x, C_x, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv_x = self.kv(x_).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)

                y_ = y.permute(0, 2, 1).reshape(B_y, C_y, H_y, W_y)
                y_ = self.sr(y_).reshape(B_y, C_y, -1).permute(0, 2, 1)
                y_ = self.norm(y_)
                kv_y = self.kv(y_).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv_x = self.kv(x).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)
                kv_y = self.kv(y).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B_x, C_x, H_x, W_x)
            x_ = self.sr(self.pool(x_)).reshape(B_x, C_x, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv_x = self.kv(x_).reshape(B_x, -1, 2, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)

            y_ = y.permute(0, 2, 1).reshape(B_y, C_y, H_y, W_y)
            y_ = self.sr(self.pool(y_)).reshape(B_y, C_y, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            y_ = self.act(y_)
            kv_y = self.kv(y_).reshape(B_y, -1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)

        k_x, v_x = kv_x[0], kv_x[1]
        k_y, v_y = kv_y[0], kv_y[1]
        # print("k_x.shape={}, k_y.shape={}".format(k_x.shape, k_y.shape))
        # print("v_x.shape={}, v_y.shape={}".format(v_x.shape, v_y.shape))

        if B_x == 1:
            k_y_avg = k_y.mean(0, True)
            v_y_avg = v_y.mean(0, True)
            k_cat_x = torch.cat((k_x, k_y_avg), dim=2)
            v_cat_x = torch.cat((v_x, v_y_avg), dim=2)
        elif B_y == 1:
            k_y_ext = k_y.repeat(B_x, 1, 1, 1)
            v_y_ext = v_y.repeat(B_x, 1, 1, 1)
            k_cat_x = torch.cat((k_x, k_y_ext), dim=2)
            v_cat_x = torch.cat((v_x, v_y_ext), dim=2)

        # print("k_cat.shape={}, v_cat.shape={}".format(k_cat.shape, v_cat.shape))

        attn_x = (q_x @ k_cat_x.transpose(-2, -1)) * self.scale
        attn_x = attn_x.softmax(dim=-1)
        attn_x = self.attn_drop(attn_x)

        x = (attn_x @ v_cat_x).transpose(1, 2).reshape(B_x, N_x, C_x)
        x = self.proj(x)
        x = self.proj_drop(x)

        if B_x == 1:
            k_x_ext = k_x.repeat(B_y, 1, 1, 1)
            v_x_ext = v_x.repeat(B_y, 1, 1, 1)
            k_cat_y = torch.cat((k_x_ext, k_y), dim=2)
            v_cat_y = torch.cat((v_x_ext, v_y), dim=2)
        elif B_y == 1:
            k_x_avg = k_x.mean(0, True)
            v_x_avg = v_x.mean(0, True)
            k_cat_y = torch.cat((k_x_avg, k_y), dim=2)
            v_cat_y = torch.cat((v_x_avg, v_y), dim=2)

        attn_y = (q_y @ k_cat_y.transpose(-2, -1)) * self.scale
        attn_y = attn_y.softmax(dim=-1)
        attn_y = self.attn_drop(attn_y)

        y = (attn_y @ v_cat_y).transpose(1, 2).reshape(B_y, N_y, C_y)
        y = self.proj(y)
        y = self.proj_drop(y)

        return x, y #torch.cat((x, y), dim=1)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H_x, W_x, y, H_y, W_y):
        outs = self.drop_path(self.attn(self.norm1(x), H_x, W_x, self.norm1(y), H_y, W_y))
        x = x + outs[0]
        y = y + outs[1]
        outs = self.drop_path((self.mlp(self.norm2(x), H_x, W_x), self.mlp(self.norm2(y), H_y, W_y)))
        x = x + outs[0]
        y = y + outs[1]
        return x, y


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def make_stage(i,
             img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
             num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
             attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
             sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, pretrained=None):
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
    cur = 0
    for idx_ in range(i):
        cur += depths[idx_]

    patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                    patch_size=7 if i == 0 else 3,
                                    stride=4 if i == 0 else 2,
                                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                    embed_dim=embed_dims[i])

    block = nn.ModuleList([Block(
        dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
        sr_ratio=sr_ratios[i], linear=linear)
        for j in range(depths[i])])
    norm = norm_layer(embed_dims[i])

    return patch_embed, block, norm


class PyramidVisionTransformerV2(Backbone): #(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, pretrained=None, only_train_norm=False, train_branch_embed=True, frozen_stages=-1, multi_output=False):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.branch_embed_stage = 0
        for i in range(num_stages):
            patch_embed, block, norm = make_stage(i, 
                 img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer, depths,
                 sr_ratios, num_stages, linear, pretrained)

            if i >= self.branch_embed_stage:
                branch_embed = nn.Embedding(2, embed_dims[i])
                setattr(self, f"branch_embed{i + 1}", branch_embed)
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        # self.init_weights(pretrained)

        self.multi_output = multi_output
        self.embed_dims = embed_dims

        self.frozen_stages = frozen_stages
        self.only_train_norm = only_train_norm
        self.train_branch_embed = train_branch_embed

        self._freeze_stages()
        if not self.train_branch_embed:
            self._freeze_branch_embed()

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            m.normal_(mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        if self.multi_output:
            outputs = {"res2": x[0], "res3": x[1], "res4": x[2], "res5": x[3]}
            return outputs
        else:
            outputs = {"res4": x[-1]}
            return outputs


    def forward_features_with_two_branch(self, x, y):
        B_x = x.shape[0]
        B_y = y.shape[0]
        outs = []

        # print("Input x.shape={}".format(x.shape))
        # print("Input y.shape={}".format(y.shape))
        for i in range(self.num_stages):
            if i >= self.branch_embed_stage:
                branch_embed = getattr(self, f"branch_embed{i + 1}")
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H_x, W_x = patch_embed(x)
            y, H_y, W_y = patch_embed(y)

            if i < self.branch_embed_stage:
                for blk in block:
                    x, y = blk(x, H_x, W_x, y, H_y, W_y)
            else:
                x_branch_embed = torch.zeros(x.shape[:-1], dtype=torch.long).cuda()
                x = x + branch_embed(x_branch_embed)

                y_branch_embed = torch.ones(y.shape[:-1], dtype=torch.long).cuda()
                y = y + branch_embed(y_branch_embed)

                for blk in block:
                    x, y = blk(x, H_x, W_x, y, H_y, W_y)

            x = norm(x)
            x = x.reshape(B_x, H_x, W_x, -1).permute(0, 3, 1, 2).contiguous()
            y = norm(y)
            y = y.reshape(B_y, H_y, W_y, -1).permute(0, 3, 1, 2).contiguous()
            outs.append((x, y))

        return outs


    def forward_with_two_branch(self, x, y):
        outs = self.forward_features_with_two_branch(x, y)
        # x = self.head(x)

        if self.multi_output:
            outputs = {"res2": outs[0], "res3": outs[1], "res4": outs[2], "res5": outs[3]}
            return outputs
        else:
            outputs = {"res4": outs[-1]}
            return outputs


    def output_shape(self):
        if self.multi_output:
            return {
                "res2": ShapeSpec(
                    channels=self.embed_dims[0], stride=4
                ),
                "res3": ShapeSpec(
                    channels=self.embed_dims[1], stride=8
                ),
                "res4": ShapeSpec(
                    channels=self.embed_dims[2], stride=16
                ),
                "res5": ShapeSpec(
                    channels=self.embed_dims[3], stride=32
                )
            }
        else:
            return {
                "res4": ShapeSpec(
                    channels=self.embed_dims[2], stride=16
                )
            }

    def _freeze_branch_embed(self):
        for i in range(self.num_stages):
            if i >= self.branch_embed_stage:
                branch_embed = getattr(self, f"branch_embed{i + 1}")
                branch_embed.eval()
                for param in branch_embed.parameters():
                    param.requires_grad = False

    def _freeze_stages(self):
        print("===============frozen at ", self.frozen_stages)

        if self.only_train_norm:
            print("Only train the normalization layers")

            for i in range(2, 5):
                print("===============freezing stage ", i-1)
                patch_embed = getattr(self, f"patch_embed{i - 1}")
                block = getattr(self, f"block{i - 1}")
                norm = getattr(self, f"norm{i - 1}")

                patch_embed.eval()
                for name, param in patch_embed.named_parameters():
                    if 'norm' in name:
                        if i < self.frozen_stages + 1:
                            param.requires_grad = False
                        else:
                            pass
                    else:
                        param.requires_grad = False

                block.eval()
                for name, param in block.named_parameters():
                    if 'norm' in name:
                        if i < self.frozen_stages + 1:
                            param.requires_grad = False
                        else:
                            pass
                    else:
                        param.requires_grad = False

                norm.eval()
                for name, param in norm.named_parameters():
                    if i < self.frozen_stages + 1:
                        param.requires_grad = False
                    else:
                        pass
        else:
            for i in range(2, self.frozen_stages + 1):
                print("===============freezing stage ", i-1)
                patch_embed = getattr(self, f"patch_embed{i - 1}")
                block = getattr(self, f"block{i - 1}")
                norm = getattr(self, f"norm{i - 1}")

                patch_embed.eval()
                for param in patch_embed.parameters():
                    param.requires_grad = False

                block.eval()
                for param in block.parameters():
                    param.requires_grad = False

                norm.eval()
                for param in norm.parameters():
                    param.requires_grad = False


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


# @BACKBONE_REGISTRY.register()
class pvt_v2_b0(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b1(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b2(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b2_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b2_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2_li.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b3(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b3_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b3_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b4(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b4_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b5(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


# @BACKBONE_REGISTRY.register()
class pvt_v2_b5_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b5_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True, pretrained="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth", num_stages=kwargs['num_stages'], only_train_norm=kwargs['only_train_norm'], train_branch_embed=kwargs['train_branch_embed'], frozen_stages=kwargs['frozen_stages'], multi_output=kwargs['multi_output'])


@BACKBONE_REGISTRY.register()
def build_FCT_backbone(cfg, input_shape):
    backbone_type = cfg.MODEL.BACKBONE.TYPE
    if backbone_type == "pvt_v2_b2_li":
        return pvt_v2_b2_li(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM, train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED, frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b5":
        return pvt_v2_b5(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM, train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED, frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b4":
        return pvt_v2_b4(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM, train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED, frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b3":
        return pvt_v2_b3(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM, train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED, frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b2":
        return pvt_v2_b2(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM, train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED, frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b1":
        return pvt_v2_b1(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM, train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED, frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    elif backbone_type == "pvt_v2_b0":
        return pvt_v2_b0(only_train_norm=cfg.MODEL.BACKBONE.ONLY_TRAIN_NORM, train_branch_embed=cfg.MODEL.BACKBONE.TRAIN_BRANCH_EMBED, frozen_stages=cfg.MODEL.BACKBONE.FREEZE_AT, num_stages=3, multi_output=False)
    else:
        print("do not support backbone type ", backbone_type)
        return None
