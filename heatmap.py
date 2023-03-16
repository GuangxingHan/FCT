#!/usr/bin/env python3
"""
Created on Thursday, March 16, 2023
@author: Guangxing Han
"""

# model architecture: pvt_v2_b2_li
# mask_cur is the attention map of attn_x (https://github.com/GuangxingHan/FCT/blob/main/FCT/modeling/fsod/FCT.py#L197) at the head_ head and the idx_ row
mask_cur = attn_x[0][head_][idx_].squeeze()

mask_1 = mask_cur[:49]
mask_1 = (mask_1 - mask_1.min())/(mask_1.max() - mask_1.min())
mask_1 = mask_1.reshape(7,7)
heat_img_1 = cv2.resize(mask_1, (query_image.shape[1], query_image.shape[0]))
heat_img_1 = heat_img_1 * 255
heat_img_1 = heat_img_1.astype(np.uint8)
heat_img_1 = cv2.applyColorMap(heat_img_1, cv2.COLORMAP_JET)
# print("heat_img_1.shape=", heat_img_1.shape)
img_add_attn_1 = cv2.addWeighted(query_image, 0.7, heat_img_1, 0.3, 0)
cv2.imwrite("query_attention.jpg", img_add_attn_1)

mask_2 = mask_cur[49:]
mask_2 = (mask_2 - mask_2.min())/(mask_2.max() - mask_2.min())
mask_2 = mask_2.reshape(7,7)
heat_img_2 = cv2.resize(mask_2, (support_image.shape[1], support_image.shape[0]))
heat_img_2 = heat_img_2 * 255
heat_img_2 = heat_img_2.astype(np.uint8)
heat_img_2 = cv2.applyColorMap(heat_img_2, cv2.COLORMAP_JET)
# print("heat_img_2.shape=", heat_img_2.shape)
img_add_attn_2 = cv2.addWeighted(support_image, 0.7, heat_img_2, 0.3, 0)
cv2.imwrite("support_attention.jpg", img_add_attn_2)
