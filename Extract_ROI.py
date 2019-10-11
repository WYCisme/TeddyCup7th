

import pandas as pd
import SimpleITK as sitk
import os
from DataPretreat import *

dataPath = "..\\B_SimpleData\\DataSet1"
traim_folder = "\\arterial phase"  # 对门脉期训练分割
patienNum = os.listdir(dataPath)
# 训练数据
if not os.path.exists('./ROI_IMG'):
    os.mkdir('./ROI_IMG')
if not os.path.exists('./ROI_IMG/transfer'):
    os.mkdir('./ROI_IMG/transfer')
if not os.path.exists('./ROI_IMG/untransfer'):
    os.mkdir('./ROI_IMG/untransfer')
lables = read_csv()  # 病人标签。
pID = []
pAera = []
pYY = []
for index, i in enumerate(patienNum[0:108]):  # 读取十个人的数据做训练 range(a,b)即 [a,b)
    print("trainP_%s" % i)
    P_path = './ROI_IMG/%s' % i
    if not os.path.exists(P_path):
        os.mkdir(P_path)
    FIleNames_dcm = []  #
    data_path = dataPath + '\\%s' % i + traim_folder
    # 读取mask掩膜图像
    lstFile_mask = []
    for dirName, subdirList, fileList in os.walk(data_path):
        for filename in fileList:
            if "mask" in filename:  # 判断文件是否为mask文件
                lstFile_mask.append(os.path.join(dirName, filename))  # 加入到列表中
            if ".dcm" in filename.lower():  # 判断文件是否为dicom文件
                # print(filename)
                FIleNames_dcm.append(os.path.splitext(filename)[0])  # 加入到列表中

    liver_slices = []
    for mpath in lstFile_mask:
        liver_slices.append(cv2.imread(mpath, cv2.IMREAD_GRAYSCALE))  # 读取图片，第二个参数表示以灰度图像读入
    maskimg = np.asarray([s for s in liver_slices])
    # 读取dcm文件数据
    # print('路径：',data_path)
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(data_path)  # 自动选择DCM文件提取
    # print(img_names)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    src = image_array.copy()
    for x in range(0, len(image_array)):  # GetArrayFromImage(image)获取的顺序倒了，需要反转顺序
        image_array[len(image_array) - x - 1] = src[x]

    seg_liver = maskimg.copy()
    seg_liver[seg_liver > 0] = 1

    start, end = getRangeImageDepth(maskimg)
    if start == end:
        print("continue")
        continue
    print("start:", start, " end:", end)
    image_array = image_array[start:end + 1]
    seg_liver = seg_liver[start:end + 1]

    image_array = transform_ctdata(image_array, 160, 40, normal=False)
    image_array = clahe_equalized(image_array)

    image_array /= 255.

    # 只选择ROI区域
    ROI_image_array = image_array * seg_liver

    max_area, all_area = get_areaOfMask(ROI_image_array)
    avg_area = all_area / (end - start + 1)

    pID.append(i)
    pAera.append(avg_area)
    pYY.append(lables[index][3])

    # P_isTransferPath = './ROI_IMG/transfer/%s' % i
    # P_noTransferPath = './ROI_IMG/untransfer/%s' % i
    # if lables[index][3] == "+":
    #     if not os.path.exists(P_isTransferPath):
    #         os.mkdir(P_isTransferPath)
    #     for idx, img in enumerate(ROI_image_array):
    #         io.imsave(os.path.join(P_isTransferPath + '/' + FIleNames_dcm[idx] + '_ROI.png'), img)
    # if lables[index][3] == "-":
    #     if not os.path.exists(P_noTransferPath):
    #         os.mkdir(P_noTransferPath)
    #     for idx, img in enumerate(ROI_image_array):
    #         io.imsave(os.path.join(P_noTransferPath + '/' + FIleNames_dcm[idx] + '_ROI.png'), img)

    # for idx, img in enumerate(ROI_image_array):
    #     if not os.path.exists('IMG/%s'%i):
    #         os.mkdir('IMG/%s'%i)
    #     io.imsave(os.path.join( 'IMG/%s'%i +'/'+ FIleNames_dcm[idx] + '.png'), img)
    # for idx, roi_img in enumerate(ROI_image_array):
    #     io.imsave(os.path.join(P_path + '/' + FIleTestNamePath_dcm[idx] + '_ROI.png'), roi_img)
# end of lop
patienL = {'id': pID, 'area': pAera, '阴性/阳性': pYY}
data = pd.DataFrame(patienL)
data.to_csv("patienLable.csv", index=False, sep=',')
