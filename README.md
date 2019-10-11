# TeddyCup7th
        第七届泰迪杯数据挖掘挑战赛-B题-直肠癌肿瘤分割
        

## 1.区域分割（主要）
    * DataPretreat.py   
        里面是图像预处理函数以及一些工具哈数
    * HDF5DatasetGenerator.py          h5文件读操作
    * HDF5DatasetWriter.py             h5文件写操作
    * DataExtraction.py     
        训练数据以及验证数据的提取   PS：这里需要给出训练集路径
        
    * Unet_2gpu.py	训练模型
    * Predict.py	分割预测

## 2.分类 （不完善）
    * Extract_ROI.py	提取ROI面积特征
    * SVM_Texture.py	SVM二分类
