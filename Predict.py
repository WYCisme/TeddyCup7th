# -*- coding: utf-8 -*-

'''
使用训练好的模型分割肿瘤区域
'''
import os
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from keras import backend as K
from skimage import io
from keras.optimizers import RMSprop

IMG_WIDTH = 192
IMG_HEIGHT = 200
IMG_CHANNELS = 1

BATCH_SIZE = 8

K.set_image_data_format('channels_last')

import SimpleITK as sitk

from DataPretreat import *
from Unet_2gpu import *

class UnetModel:

    def predict(self):
        model = get_unet()
        model.load_weights('FinalModel.h5')
        BATCH_SIZE = 10  # 测试使用
        #TODO 需要分割的图像文件路径
        dataPath = "..\\B_SimpleData\\DataSet1"
        traim_folder = "\\arterial phase"  # 对门脉期训练分割
        patienNum = os.listdir(dataPath)  # CT图像目录
        testDataSize = 0  # 测试数量
        for i in patienNum[80:90]:  # 取5个人做测试
            FIleTestNamePath_dcm = []  # 用于保存每个文件名完整路径
            print("testP_%s" % i)
            ################读取保存文件名用于输出打印################

            data_path = dataPath + '\\%s' % i + traim_folder
            # 读取mask掩膜图像
            for dirName, subdirList, fileList in os.walk(data_path):
                for filename in fileList:
                    if ".dcm" in filename.lower():  # 判断文件是否为dicom文件
                        # print(filename)
                        FIleTestNamePath_dcm.append(data_path + "\\" + os.path.splitext(filename)[0])  # 加入到列表中
            #########################################################
            data_path = dataPath + '/%s' % i + traim_folder
            # 读取mask掩膜图像
            lstFile_mask = []
            for dirName, subdirList, fileList in os.walk(data_path):
                for filename in fileList:
                    if "mask" in filename:  # 判断文件是否为mask文件
                        lstFile_mask.append(os.path.join(dirName, filename))  # 加入到列表中
            liver_slices = []
            for mpath in lstFile_mask:
                liver_slices.append(cv2.imread(mpath, cv2.IMREAD_GRAYSCALE))  # 读取图片，第二个参数表示以灰度图像读入
            livers = np.asarray([s for s in liver_slices])
            # 读取dcm文件数据
            reader = sitk.ImageSeriesReader()
            img_names = reader.GetGDCMSeriesFileNames(data_path)  # 自动选择DCM文件提取
            reader.SetFileNames(img_names)
            image = reader.Execute()
            image_array = sitk.GetArrayFromImage(image)  # z, y, x
            src = image_array.copy()
            for x in range(0, len(image_array)):  # GetArrayFromImage(image)获取的顺序倒了，需要反转顺序
                image_array[len(image_array) - x - 1] = src[x]
            ######################

            seg_liver = livers.copy()
            seg_liver[seg_liver > 0] = 1

            testDataSize += len(image_array)  # 测试数据量

            image_array = transform_ctdata(image_array, 160, 40, normal=False)
            image_array = clahe_equalized(image_array)

            image_array[image_array >= 254.] = 0       #测试时输入数据要做阈值处理，避免骨头部位像素值过高影响分割

            image_array /= 255.

            show_src_seg(image_array , seg_liver, index=i)  # 打印

            crop_images, crop_tumors = crop_images_tiqucaijian(image_array, seg_liver)

            crop_images = np.expand_dims(crop_images, axis=-1)
            # crop_tumors = np.expand_dims(crop_tumors, axis=-1)

            print('数据量：', testDataSize)

            #########################
            print("-----------filesize:", len(crop_images))
            BATCH_SIZE = 2
            step = int(len(crop_images - 1) / BATCH_SIZE + 1)
            for idx in range(0, step):
                thisTestCrop=crop_images[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                imgs_mask_test = model.predict(thisTestCrop, verbose=1)
                print('Saving predicted masks to files...')
                print('-' * 30)
                thisFilePath = FIleTestNamePath_dcm[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                thisCrop_img = crop_images[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

                print(FIleTestNamePath_dcm[0])
                j = 0
                for maskimg in imgs_mask_test:
                    maskimg = (maskimg[:, :, 0] * 255.).astype(np.uint8)  # 预测分割的mask
                    ctimg = (thisCrop_img[j, :, :, 0] * 255.).astype(np.uint8)  # 原图
                    ctimg,maskimg=crop_images_huanyuan(ctimg,maskimg)
                    io.imsave(os.path.join(thisFilePath[j] + '_Orimg.png'), ctimg)
                    io.imsave(os.path.join(thisFilePath[j] + '_myM.png'), maskimg)
                    j += 1
        print("-----%s folder OK-----" % i)

unet = UnetModel()
#开始分割
unet.predict()
