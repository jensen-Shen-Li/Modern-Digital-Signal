import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import copy

matplotlib.use('TkAgg')


def show_anns(anns):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    index = 1
    #
    sort_ann = sorted(anns, key=(lambda x: x['bbox'][0]), reverse=True)

    list_x = []
    list_y = []
    weight = []
    list_i = []
    for i, ann in enumerate(sort_ann):
        if 300 < ann['bbox'][3] < 900:
            list_i.append(i)
        list_x.append(ann['bbox'][0])
        list_y.append(ann['bbox'][1]+ann["bbox"][3])
        weight.append(ann['bbox'][3])

    count = 0
    for each in list_i:
        sort_ann.pop(each-count)
        list_x.pop(each-count)
        list_y.pop(each-count)
        count += 1

    list_xx = []
    list_yy = []

    for ann in sort_ann:
        list_xx.append(ann['bbox'][0])
        list_yy.append(ann['bbox'][1]+ann["bbox"][3])
    mar_len = statis_comp(list_xx)

    sort_ann = create_mask(sort_ann, list_yy, mar_len)

    for ann in sort_ann:
        if ann['area'] > 7000:
            middle_x, middle_y = (ann['bbox'][0]), (ann['bbox'][1] + ann['bbox'][3])
            plt.text(middle_x, middle_y, f'{ann["bbox"][2]},{ann["bbox"][3]}', color='r', fontsize=8)
        # plt.text(x, y, f'{ann["area"]}', color='r', fontsize=8)
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        index += 1

    # for i in range(len(list_x) - 1):


# 统计每一列的块数
def statis_comp(x):
    mar_len = []
    flag = 0
    for i in range(len(x) - 1):
        if abs(x[i + 1] - x[i]) >= 10:
            len_comp = abs(flag - (i + 1))
            flag = i + 1
            # if len_comp > 0:
            mar_len.append(len_comp)
            continue
    return mar_len


# 平移新建mask
def create_mask(anns, list_y, mar_len):
    max_comp = max(mar_len)
    index = 0
    base_index = 1
    num = 0
    for each in mar_len:
        sup_index = 0

        if each < max_comp and each > 1:
            num += 2
            index += each
            max_y = max(list_y[base_index : index+1])
            for i in range(base_index, index+1):
                if list_y[i] == max_y:
                    sup_index = i
            base_index = index + 1
            ann_self = copy.deepcopy(anns[sup_index])
            mask = ann_self['segmentation'].astype(float)
            # 图像形态学处理
            if ann_self['bbox'][0] == 365:
                kernel = np.ones((5, 5), np.uint8)
                # 图像腐蚀处理
                mask = cv2.erode(mask, kernel, iterations=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 结构元素，矩形大小3*3
            binary = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            M = np.float32([[1, 0, -2], [0, 1, ann_self['bbox'][2]+20+num]])  # 10
            ann_self['segmentation'] = cv2.warpAffine(binary, M, (mask.shape[1], mask.shape[0]))

            ann_self['segmentation'] = ann_self['segmentation'] > 0
            ann_self['bbox'][1] = ann_self['bbox'][1] - 15
            ann_self['bbox'][2] = ann_self['bbox'][2] + 5 - num//2
            import random
            ann_self['bbox'][3] = ann_self['bbox'][3] + int(random.randint(0, 3))
            ann_self['bbox'][1] = ann_self['bbox'][1] + ann_self['bbox'][2]+30
            anns.append(ann_self)
    return anns




sys.path.append("..")
 

sam_checkpoint = "./checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
 
device = "cuda"#如果想用cpu,改成cpu即可
 
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
 
image = cv2.imread('notebooks/images/wall.png')
# image = cv2.imread('notebooks/images/result.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("imagesize",image.shape)
# img_blurr = cv2.medianBlur(image, 3)

# img = cv2.imread('result.jpg')
# edges = cv2.Canny(image, 50, 150)
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()
 
 
mask_generator = SamAutomaticMaskGenerator(sam)


masks = mask_generator.generate(image)
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
