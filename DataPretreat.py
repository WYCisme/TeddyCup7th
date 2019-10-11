'''
需要用到的一些工具函数
'''
import csv

import numpy as np
import cv2
import matplotlib.pyplot as plt


def getRangeImageDepth(image):
    z = np.any(image, axis=(1, 2))
    if len(np.where(z)[0]) > 0:
        startposition, endposition = np.where(z)[0][[0, -1]]
    else:
        startposition = endposition = 0
    return startposition, endposition


def show_src_seg(image_array, maskimg, index, rows=3, start_with=0, show_every=1):
    assert image_array.shape == maskimg.shape
    rows = image_array.shape[0]
    plan_rows = start_with + rows * show_every - 1
    print("rows=%d,planned_rows=%d" % (rows, plan_rows))
    cols = 2
    print("final rows=%d" % rows)
    fig, ax = plt.subplots(rows, cols, figsize=[5 * cols, 5 * rows])
    for i in range(rows):
        ind = start_with + i * show_every
        ax[i, 0].set_title('ct_img %d' % ind)
        ax[i, 0].imshow(image_array[ind], cmap='gray')
        ax[i, 0].axis('off')

        ax[i, 1].set_title('mask %d' % ind)
        ax[i, 1].imshow(maskimg[ind], cmap='gray')
        ax[i, 1].axis('off')
    name = "./ResultComp/" + str(index) + ".png"
    plt.savefig(name)


def transform_ctdata(image, windowWidth, windowCenter, normal=False):

    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


def clahe_equalized(imgs):
    assert (len(imgs.shape) == 3)  # 3D arrays
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(len(imgs)):
        imgs_equalized[i, :, :] = clahe.apply(np.array(imgs[i, :, :], dtype=np.uint8))
    return imgs_equalized


# 剪裁出直肠部分
def crop_images_tiqucaijian(images, maskimg):
    images = images[:, 250:450, :]
    images = images[:, :, 160:352]

    maskimg = maskimg[:, 250:450, :]
    maskimg = maskimg[:, :, 160:352]

    return images, maskimg


# 将宽高剪裁为192*200的输出mask还原成512*512
def crop_images_huanyuan(images, maskimg):
    fixed_buquanRight = np.zeros((200,160), dtype=np.uint8)  # 160*200用于给mask结果右补全图像
    fixed_buquanLeft = np.zeros((200, 160), dtype=np.uint8)  # 160*200用于给mask结果左补全图像
    images = np.concatenate((images, fixed_buquanRight), axis=1)  # 补全像素
    images = np.concatenate((fixed_buquanLeft, images), axis=1)  # 补全像素
    maskimg = np.concatenate((maskimg, fixed_buquanRight), axis=1)  # 补全像素
    maskimg = np.concatenate((fixed_buquanLeft, maskimg), axis=1)  # 补全像素

    fixed_buquanTop = np.zeros((250, 512), dtype=np.uint8)  # 250*512用于给mask结果上补全图像
    fixed_buquanDown = np.zeros((62, 512), dtype=np.uint8)  # 62*512用于给mask结果下补全图像
    images = np.concatenate((fixed_buquanTop, images), axis=0)  # 补全像素
    images = np.concatenate((images, fixed_buquanDown), axis=0)  # 补全像素
    maskimg = np.concatenate((fixed_buquanTop, maskimg), axis=0)  # 补全像素
    maskimg = np.concatenate((maskimg, fixed_buquanDown), axis=0)  # 补全像素

    return images, maskimg

def get_areaOfMask(image_mask):
    max_area = 0  # 统计面积最大的一张图片的面积
    all_area=0
    for img in image_mask:
        a = len(img[img != 0])
        all_area+=a
        if a > max_area:
            max_area = a

    return max_area, all_area
# 读取csv文件
def read_csv():
    csv_file = csv.reader(open('临床数据.csv'))
    # print(csv_file)
    # 添加newline可以避免一行之后的空格,这样需要在python3环境下运行
    ccc = []
    for item in csv_file:
        # print item
        item[1] = item[1][0:2]
        ccc.append(item)
        # print(item)

    src = ccc.copy()
    for x in range(0, len(ccc)):  # 获取的顺序倒了，需要反转顺序
        ccc[len(ccc) - x - 1] = src[x]
    return ccc


