# import cv2
#
# img = cv2.imread('1.png',0)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#             if img[i][j]>220:
#                 img[i][j]=255
#             else:
#                 img[i][j]=0
#
# cv2.imshow('a',img)
# cv2.imwrite('2.png',img)
# # cv2.waitKey(0)

import torch
import torch.nn as nn

x=torch.zeros(1,1,6,6)
x[:,:,:,5:6]=1.0
print(x)
conv=nn.Conv2d(1,1,kernel_size=3,stride=2)
x=conv(x)
print(x)