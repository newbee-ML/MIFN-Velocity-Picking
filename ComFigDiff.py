import cv2
import numpy as np
import os
Root = "C:\\Users\\25509\\Desktop\\20220321"

"""
dq8
"""
AP1 = cv2.imread(os.path.join(Root, 'dq8_940_AI.png'))
MP1 = cv2.imread(os.path.join(Root, 'dq8_940_manu.png'))
Diff = AP1-MP1
cv2.imwrite(os.path.join(Root, 'dq8_940_diff.png'), Diff)
# 1
Ymin, Ymax, Xmin, Xmax = 1500, 2127, 1098, 1400
CAP1 = AP1[Xmin:Xmax, Ymin:Ymax]
CMP1 = MP1[Xmin:Xmax, Ymin:Ymax]
cv2.imwrite(os.path.join(Root, 'Compare', '1_dq8_940_AP.png'), CAP1)
cv2.imwrite(os.path.join(Root, 'Compare', '1_dq8_940_MP_1.png'), CMP1)
# 2
Ymin, Ymax, Xmin, Xmax = 7032, 7663, 348, 843
CAP1 = AP1[Xmin:Xmax, Ymin:Ymax]
CMP1 = MP1[Xmin:Xmax, Ymin:Ymax]
cv2.imwrite(os.path.join(Root, 'Compare', '2_dq8_940_AP.png'), CAP1)
cv2.imwrite(os.path.join(Root, 'Compare', '2_dq8_940_MP.png'), CMP1)
# print(np.max(AP1-MP1))
# print(123)

"""
hade
"""
AP1 = cv2.imread(os.path.join(Root, 'hade_2620_AI.png'))
MP1 = cv2.imread(os.path.join(Root, 'hade_2620_manu.png'))
Ymin, Ymax, Xmin, Xmax = 1000, 1500, 1098, 1400
CAP1 = AP1[Xmin:Xmax, Ymin:Ymax]
CMP1 = MP1[Xmin:Xmax, Ymin:Ymax]
# cv2.imshow('image', AP1-MP1)  # 显示图片
# cv2.resizeWindow('image', 600, 500)
# cv2.waitKey(0) 
cv2.imwrite(os.path.join(Root, 'hade_2620_diff.png'), Diff)
cv2.imwrite(os.path.join(Root, 'Compare', 'hade_2620_AP.png'), CAP1)
cv2.imwrite(os.path.join(Root, 'Compare', 'hade_2620_MP.png'), CMP1)