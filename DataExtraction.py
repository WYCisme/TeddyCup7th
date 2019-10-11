'''
数据准备
'''
import SimpleITK as sitk
import os
from keras.preprocessing.image import ImageDataGenerator
from HDF5DatasetWriter import HDF5DatasetWriter
from DataPretreat import *


seed = 1
data_gen_args = dict(rotation_range=3,
                     width_shift_range=0.01,
                     height_shift_range=0.01,
                     shear_range=0.01,
                     zoom_range=0.01,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

if not os.path.exists('data_train'):
    os.mkdir('data_train')

if os.path.exists('./data_train/train_liver.h5'):
    os.remove('./data_train/train_liver.h5')  # 清理原来的文件
datasetTrains = HDF5DatasetWriter(image_dims=(1000, 200, 192, 1),  # (数量，高，宽，1）数量需要提前估计
                                  mask_dims=(1000, 200, 192, 1),
                                  outputPath="./data_train/train_liver.h5")

print('开始\n')
# part1
# TODO 需要的训练数据集
dataPath = "..\\B_SimpleData\\DataSet1"
traim_folder = "\\arterial phase"  # 对门脉期训练分割
patienNum = os.listdir(dataPath)
# 训练数据
trainDataSize = 0
for i in patienNum[0:80]:  # 读取十个人的数据做训练 range(a,b)即 [a,b)
    print("trainP_%s" % i)
    data_path = dataPath + '\\%s' % i + traim_folder
    # 读取mask掩膜图像
    lstFile_mask = []
    for dirName, subdirList, fileList in os.walk(data_path):
        for filename in fileList:
            if "mask" in filename:  # 判断文件是否为mask文件
                lstFile_mask.append(os.path.join(dirName, filename))  # 加入到列表中
    liver_slices = []
    for mpath in lstFile_mask:
        liver_slices.append(cv2.imread(mpath, cv2.IMREAD_GRAYSCALE))  # 读取图片，第二个参数表示以灰度图像读入
    maskimg = np.asarray([s for s in liver_slices])
    # 读取dcm文件数据
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(data_path)  # 自动选择DCM文件提取
    reader.SetFileNames(img_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    src = image_array.copy()
    for x in range(0, len(image_array)):  # GetArrayFromImage(image)获取的顺序倒了，需要反转顺序
        image_array[len(image_array) - x - 1] = src[x]

    seg_liver = maskimg.copy()
    seg_liver[seg_liver > 0] = 1

    start, end = getRangeImageDepth(maskimg)    #获取肿瘤区域切片位置
    if start == end :
        print("continue")
        continue
    print("start:", start, " end:", end)

    if start >= 2:
        start = start - 2
    if end < len(image_array) - 3:
        end = end+3
    # 选择肿瘤区域q切片
    image_array = image_array[start:end]    #多获取肿瘤区域前后2张
    seg_liver = seg_liver[start:end]    #多获取肿瘤区域前后2张

    trainDataSize += len(image_array)

    image_array = transform_ctdata(image_array, 160, 40, normal=False)  #窗口化

    image_array = clahe_equalized(image_array)  #直方图均衡化
    image_array /= 255.

    crop_images, crop_tumors = crop_images_tiqucaijian(image_array, seg_liver)  # 获取肿瘤区域192*200

    show_src_seg(crop_images, crop_tumors, index=i)

    crop_images = np.expand_dims(crop_images, axis=-1)
    crop_tumors = np.expand_dims(crop_tumors, axis=-1)

    datasetTrains.add(crop_images, crop_tumors)

# end of lop
print(datasetTrains.close())
np.save('./data_train/train_size', trainDataSize)
print('训练数据量：', np.load('./data_train/train_size.npy'))
#验证集
# 训练数据
vailDataSize = 0
if os.path.exists('./data_train/vail_liver.h5'):
    os.remove('./data_train/vail_liver.h5')  # 清理原来的文件
vailsetTrains = HDF5DatasetWriter(image_dims=(300, 200, 192, 1),  # (数量，高，宽，1）数量需要提前估计
                                  mask_dims=(300, 200, 192, 1),
                                  outputPath="./data_train/vail_liver.h5")
for i in patienNum[80:100]:  # 读取20%做验证集
    print("trainP_%s" % i)

    data_path = dataPath + '\\%s' % i + traim_folder
    # 读取mask掩膜图像
    lstFile_mask = []
    for dirName, subdirList, fileList in os.walk(data_path):
        for filename in fileList:
            if "mask" in filename:  # 判断文件是否为mask文件
                lstFile_mask.append(os.path.join(dirName, filename))  # 加入到列表中
    liver_slices = []
    for mpath in lstFile_mask:
        liver_slices.append(cv2.imread(mpath, cv2.IMREAD_GRAYSCALE))  # 读取图片，第二个参数表示以灰度图像读入
    maskimg = np.asarray([s for s in liver_slices])
    # 读取dcm文件数据
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(data_path)  # 自动选择DCM文件提取
    reader.SetFileNames(img_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    src = image_array.copy()
    for x in range(0, len(image_array)):  # GetArrayFromImage(image)获取的顺序倒了，需要反转顺序
        image_array[len(image_array) - x - 1] = src[x]

    seg_liver = maskimg.copy()
    seg_liver[seg_liver > 0] = 1


    start, end = getRangeImageDepth(maskimg)    #获取肿瘤区域切片位置
    if start == end :
        print("continue")
        continue
    print("start:", start, " end:", end)

    if start >= 2:
        start = start - 2
    if end < len(image_array) - 3:
        end = end+3
    # 选择肿瘤区域q切片
    image_array = image_array[start:end]    #多获取肿瘤区域前后2张
    seg_liver = seg_liver[start:end]    #多获取肿瘤区域前后2张

    vailDataSize += len(image_array)

    image_array = transform_ctdata(image_array, 160, 40, normal=False)  #窗口化

    image_array = clahe_equalized(image_array)  #直方图均衡化
    image_array /= 255.

    crop_images, crop_tumors = crop_images_tiqucaijian(image_array, seg_liver)  # 获取肿瘤区域192*200
    # show_src_seg(crop_images, crop_tumors, index=i)

    crop_images = np.expand_dims(crop_images, axis=-1)
    crop_tumors = np.expand_dims(crop_tumors, axis=-1)

    vailsetTrains.add(crop_images, crop_tumors)

# end of lop
print(vailsetTrains.close())
np.save('./data_train/vail_size', vailDataSize)
print('训练数据量：', np.load('./data_train/vail_size.npy'))

if __name__ == "__main__":
    print('---ok---')
