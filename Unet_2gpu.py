# -*- coding: utf-8 -*-
'''
使用Unet网络训练
'''
import os
import cv2
import numpy as np

import tensorflow as tf
from HDF5DatasetGenerator import HDF5DatasetGenerator
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K
# from skimage import io
import keras.backend.tensorflow_backend as KTF


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU编号
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)
# # 设置session
KTF.set_session(session)

# Set some parameters
IMG_WIDTH = 192
IMG_HEIGHT = 200
IMG_CHANNELS = 1
TOTAL = np.load('./data_train/train_size.npy')  # 总共的训练数据
VAIL_TOTAL=np.load('./data_train/vail_size.npy')    #验证集
outputPathTrain = './data_train/train_liver.h5'  # 训练文件
outputPathVail = './data_train/vail_liver.h5'  # 验证文件

BATCH_SIZE = 8

K.set_image_data_format('channels_last')


def dice_coef(y_true, y_pred):
    print("in loss function, y_true shape:", y_true.shape)
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def weighted_binary_cross_entropy_loss(y_true, y_pred):

    b_ce = K.binary_crossentropy(y_true, y_pred)
    one_weight = K.mean(y_true)
    zero_weight = 1 - one_weight
    weight_vector = y_true * zero_weight + (1. - y_true) * one_weight
    weighted_b_ce = weight_vector * b_ce
    return K.mean(weighted_b_ce)


def weighted_dice_loss(y_true, y_pred):
    mean = K.mean(y_true)
    w_1 = 1 / mean ** 2
    w_0 = 1 / (1 - mean) ** 2
    y_true_f_1 = K.flatten(y_true)
    y_pred_f_1 = K.flatten(y_pred)
    y_true_f_0 = K.flatten(1 - y_true)
    y_pred_f_0 = K.flatten(1 - y_pred)

    intersection_0 = K.sum(y_true_f_0 * y_pred_f_0)
    intersection_1 = K.sum(y_true_f_1 * y_pred_f_1)

    return -2 * (w_0 * intersection_0 + w_1 * intersection_1) \
           / ((w_0 * (K.sum(y_true_f_0) + K.sum(y_pred_f_0))) \
              + (w_1 * (K.sum(y_true_f_1) + K.sum(y_pred_f_1))))


def get_crop_shape(target, refer):
    cw = (target._keras_shape[2] - refer._keras_shape[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    ch = (target._keras_shape[1] - refer._keras_shape[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


def get_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up_conv5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)

    ch, cw = get_crop_shape(conv4, up_conv5)

    up_conv5 = ZeroPadding2D(padding=(ch, cw), data_format="channels_last")(up_conv5)
    up6 = concatenate([up_conv5, conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)

    ch, cw = get_crop_shape(conv3, up_conv6)
    up_conv6 = ZeroPadding2D(padding=(ch, cw), data_format="channels_last")(up_conv6)
    #
    up7 = concatenate([up_conv6, conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    up_conv7 = ZeroPadding2D(padding=(ch, cw), data_format="channels_last")(up_conv7)

    up8 = concatenate([up_conv7, conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    up_conv8 = ZeroPadding2D(padding=(ch, cw), data_format="channels_last")(up_conv8)

    up9 = concatenate([up_conv8, conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=RMSprop(lr=0.00001  , rho=0.9, epsilon=1e-06), loss=dice_coef_loss, metrics=[dice_coef])

    return model


class UnetModel:

    def train(self):
        reader = HDF5DatasetGenerator(dbPath=outputPathTrain, batchSize=BATCH_SIZE)
        train_iter = reader.generator() #做训练
        reader2 = HDF5DatasetGenerator(dbPath=outputPathVail, batchSize=BATCH_SIZE)
        train_iter_cell=reader2.generator()  #y验证集

        model = get_unet()
        model_checkpoint = ModelCheckpoint('FinalModel.h5', monitor='val_loss', save_best_only=True)

        model.fit_generator(train_iter, steps_per_epoch=int(TOTAL / BATCH_SIZE), verbose=1, epochs=200, shuffle=True,
                            validation_data=train_iter_cell, validation_steps=int(VAIL_TOTAL / BATCH_SIZE),
                            callbacks=[model_checkpoint])
        reader.close()
        reader2.close()

unet = UnetModel()
unet.train()
