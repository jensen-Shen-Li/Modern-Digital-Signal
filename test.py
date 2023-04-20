import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取图像
img = cv2.imread('notebooks/images/wall.png')
print("img", type(img))
exit(0)

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用Sobel滤波器检测竖向边缘
sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)

# 对Sobel滤波器结果进行二值化处理
ret, binary = cv2.threshold(sobelx, 100, 255, cv2.THRESH_BINARY)

# 使用形态学闭运算来连接边缘
kernel = np.ones((5,5), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 使用膨胀操作增加边缘的宽度
kernel = np.ones((1,10), np.uint8)
dilated = cv2.dilate(closed, kernel, iterations=1)

# 将处理后的边缘与原始图像相加以增强竖线对比度
result = cv2.addWeighted(img, 0.7, cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR), 0.3, 0)

M = np.float32([[1, 0, 0], [0, 1, 50]])  # 10
resultss = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

# 显示结果图像
plt.figure(figsize=(20,20))
plt.imshow(resultss)
plt.axis('off')
plt.show()
cv2.imwrite('result.jpg', result)

# def drop_clo(x, y):
#     list_i = []
#     for i in range(len(x) - 1):
#         if abs(x[i + 1] - x[i]) >= 10:
#             continue
#         for j in range(1, len(x) - 1):
#             print("xxx", x[i])
#             print("xxxxx", x[i+j])
#             if abs(x[i+j] - x[i]) < 10 and abs(y[i+j] - y[i]) < 10:
#                 print('iiiiii', i)
#                 list_i.append(i)
#     return list_i
#
# list_x = [20,15,14,2]
# list_y = [100, 10, 9, 20]
#
# lis = drop_clo(list_x, list_y)
# print(lis)