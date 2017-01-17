# --------------------------------------------------------
# generate LMDB of "Convolutional Pose Machines"
# python version
# Written by Lu Tian
# --------------------------------------------------------

"""generate LMDB from our own image data"""

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', '..', 'caffe_cpm', 'python')
print caffe_path
add_path(caffe_path)

import numpy as np
import json
import cv2
import lmdb
import caffe
import os.path
import struct


def float2bytes(floats):
    if type(floats) is float:
        floats = [floats]
    return struct.pack('%sf' % len(floats), *floats)


def writelmdb(dataset, imageDir, lmdbPath, validation):
    env = lmdb.open(lmdbPath, map_size=int(1e12))
    txn = env.begin(write=True)
    textFile = open(dataset, 'r')
    data = textFile.readlines()
    numSample = len(data)
    print numSample
    random_order = np.random.permutation(numSample).tolist()
    isValidationArray = [0 for i in xrange(numSample)]
    if(validation == 1):
        totalWriteCount = isValidationArray.count(0)
    else:
        totalWriteCount = len(data)
    print 'goint to write %d images..' % totalWriteCount
    writeCount = 0
    dic = {}
    for count in xrange(numSample):
        idx = random_order[count]
        info = data[idx].split(" ")
        imageName = imageDir + info[0]
        image = cv2.imread(imageName)
        if image is None:
            continue
        if len(image.shape) != 3:
            continue
        height = image.shape[0]
        width = image.shape[1]
        if width < 64:
            image = cv2.copyMakeBorder(image, 0, 0, 0, 64 - width, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            width = 64
        meta_data = np.zeros(shape=(height, width, 1), dtype=np.uint8)
        # current line index
        clidx = 0
        datasetName = 'ZERO'
        for i in range(len(datasetName)):
            meta_data[clidx][i] = ord(datasetName[i])
        clidx += 1
        # image height, image width
        height_binary = float2bytes(float(image.shape[0]))
        for i in range(len(height_binary)):
            meta_data[clidx][i] = ord(height_binary[i])
        width_binary = float2bytes(float(image.shape[1]))
        for i in range(len(width_binary)):
            meta_data[clidx][4 + i] = ord(width_binary[i])
        clidx += 1
        # (a) isValidataion(uint8), numOtherPeople(uint8), people_index(uint8), annolist_index(float), writeCount(float)
        meta_data[clidx][0] = 0
        meta_data[clidx][1] = 0
        if info[0] in dic:
            dic[info[0]] += 1
        else:
            dic[info[0]] = 1
        meta_data[clidx][2] = dic[info[0]]
        annolist_index_binary = float2bytes(float(idx))
        for i in range(len(annolist_index_binary)):
            meta_data[clidx][3 + i] = ord(annolist_index_binary[i])
        count_binary = float2bytes(float(writeCount))
        for i in range(len(count_binary)):
            meta_data[clidx][7 + i] = ord(count_binary[i])
        totalWriteCount_binary = float2bytes(float(totalWriteCount))
        for i in range(len(totalWriteCount_binary)):
            meta_data[clidx][11 + i] = ord(totalWriteCount_binary[i])
        nop = 0
        clidx += 1
        # (b) objpos_x (float), objpos_y (float)
        joints = [(float(item)) for item in info[5:-1]]
        objpos = [round((min(joints[::2]) + max(joints[::2]))/2), round((min(joints[1::2]) + max(joints[1::2]))/2)]
        objpos_binary = float2bytes(objpos)
        for i in range(len(objpos_binary)):
            meta_data[clidx][i] = ord(objpos_binary[i])
        clidx += 1
        # (c) scale_provided (float)
        scale_provided = (max(joints[1::2])-min(joints[1::2]))*1.4/200
        scale_provided_binary = float2bytes(scale_provided)
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = ord(scale_provided_binary[i])
        clidx += 1
        # (d) joint_self (3*13) (float) (3 line)
        x_binary = float2bytes(joints[::2])
        for i in range(len(x_binary)):
            meta_data[clidx][i] = ord(x_binary[i])
        clidx += 1
        y_binary = float2bytes(joints[1::2])
        for i in range(len(y_binary)):
            meta_data[clidx][i] = ord(y_binary[i])
        clidx += 1
        visible = [1.0 for item in xrange(len(joints[::2]))]
        v_binary = float2bytes(visible)
        for i in range(len(v_binary)):
            meta_data[clidx][i] = ord(v_binary[i])
        clidx += 1

        img4ch = np.concatenate((image, meta_data), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))
        datum = caffe.io.array_to_datum(img4ch, label=0)
        key = '%07d' % writeCount
        txn.put(key, datum.SerializeToString())
        if writeCount % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
        print 'count: %d/ write count: %d/ randomized: %d/ all: %d' % (count, writeCount, idx, totalWriteCount)
        writeCount += 1

    txn.commit()
    env.close()


if __name__ == '__main__':
    dataset = 'pose_trainval.txt'
    imageDir = '/deephi_data/img/'
    lmdbPath = '/data2/tianlu/lmdb/trainval'
    writelmdb(dataset, imageDir, lmdbPath, 0)