# --------------------------------------------------------
# Demo code of "Convolutional Pose Machines"
# python version
# Written by Lu Tian
# --------------------------------------------------------

"""Test a CPM network on image."""

import time
import math

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

import caffe
import numpy as np
import cv2


class Config:
    def __init__(self):
        self.use_gpu = 1
        self.gpuID = 0
        self.octave = 2
        self.click = 1
        self.caffemodel = '../model/_trained_MPI/pose_iter_985000_addLEEDS.caffemodel'
        self.deployFile = '../model/_trained_MPI/pose_deploy_centerMap.prototxt'
        self.description = 'MPII+LSP 6-stage CPM'
        self.description_short = 'MPII_LSP_6s'
        self.boxsize = 368
        self.padValue = 128
        self.npoints = 14
        self.sigma = 21
        self.stage = 6
        self.limbs = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]]
        self.part_str = ['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
                         'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'bkg']


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def produce_centerlabelmap(im_size, x, y, sigma):
    # this function generaes a gaussian peak centered at position (x,y)
    # it is only for center map in testing
    xv, yv = np.meshgrid(np.linspace(0, im_size[0], im_size[0], False), np.linspace(0, im_size[1], im_size[1], False))
    xv = xv-x
    yv = yv-y
    D2 = xv ** 2 + yv ** 2
    Exponent = np.divide(D2, 2.0*sigma*sigma)
    return np.exp(-Exponent)


def preprocess(img, mean, param):
    img_out = img*1.0/256
    img_out = img_out - mean

    boxsize = param.boxsize
    centerMapCell = produce_centerlabelmap((boxsize, boxsize), boxsize/2, boxsize/2, param.sigma)
    img_out = np.dstack((img_out, centerMapCell))
    # change H*W*C -> C*H*W
    return np.transpose(img_out, (2, 0, 1))


def applydnn(images, net, nstage, boxsize):
    # do forward pass to get scores
    # scores are now Width * Height * Channels * Num
    net.blobs['data'].data[...] = images.reshape((1, 4, boxsize, boxsize))
    net.forward()
    blobs_names = net.blobs.keys()
    scores = [[] for item in xrange(nstage)]
    for s in xrange(nstage):
        string_to_search = 'stage' + str(s + 1)
        blob_id = ' '
        for i in xrange(len(blobs_names)):
            if blobs_names[i].find(string_to_search) != -1:
                blob_id = blobs_names[i]
        scores[s] = net.blobs[blob_id].data[0]
    return scores


def padsquare(image, boxsize, center, padValue):
    w = image.shape[1]
    h = image.shape[0]
    center_box = [boxsize/2, boxsize/2]
    output = np.ones((boxsize, boxsize, image.shape[2]), dtype=np.uint8) * padValue
    left = min(center_box[0], center[0])
    right = min(boxsize - center_box[0], w - center[0])
    up = min(center_box[1], center[1])
    down = min(boxsize - center_box[1], h - center[1])
    output[(center_box[1] - up):(center_box[1] + down), (center_box[0] - left):(center_box[0] + right), :] = image[(center[1] - up):(center[1] + down), (center[0] - left):(center[0] + right), :]
    pad = [left, right, up, down]
    return output, pad, image.shape


def resize2scaledimg(score, pad, scaledshape, center):
    # score chanel: W*H*C
    output = np.zeros((scaledshape[1], scaledshape[0], score.shape[2]), dtype=np.float)
    # output[:, :, -2] = output[:, :, -2] + 1 # ignore bk
    center_box = [score.shape[0]/2, score.shape[0]/2]
    left = pad[0]
    right = pad[1]
    up = pad[2]
    down = pad[3]
    output[(center[0] - left):(center[0] + right), (center[1] - up):(center[1] + down), :] = score[(center_box[0] - left):(center_box[0] + right), (center_box[1] - up):(center_box[1] + down), :]
    return output


def findmaxinum(map):
    index = np.where(map == np.max(map))
    return index[0][0], index[1][0]


def applymodel(net, oriImg, param, rectangle, watchstage=5):

    # Select parameters from param
    boxsize = param.boxsize
    npoints = param.npoints
    nstage = param.stage

    # Apply model, with searching through a range of scales
    octave = param.octave
    # set the center and roughly scale range (overwrite the config!) according to the rectangle
    x_start = max(rectangle.x, 0)
    x_end = min(rectangle.x + rectangle.w, oriImg.shape[1])
    y_start = max(rectangle.y, 0)
    y_end = min(rectangle.y + rectangle.h, oriImg.shape[0])
    center = [(x_start + x_end)/2, (y_start + y_end)/2]

    # determine scale range
    middle_range = float(y_end - y_start)/oriImg.shape[0] * 1.2
    starting_range = middle_range * 0.8
    ending_range = middle_range * 3.0

    starting_scale = boxsize/(oriImg.shape[0]*ending_range)
    ending_scale = boxsize/(oriImg.shape[0]*starting_range)
    multiplier = [2 ** item for item in np.arange(math.log(starting_scale, 2), math.log(ending_scale, 2), 1.0/octave)]

    # data container for each scale and stage
    score = [[] for item in xrange(len(multiplier))]
    pad = [[] for item in xrange(len(multiplier))]

    for m in xrange(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (int(round(oriImg.shape[1]*scale)), int(round(oriImg.shape[0]*scale))))
        center_s = [int(item * scale) for item in center]
        imageToTest, pad[m], scaledshape = padsquare(imageToTest, boxsize, center_s, param.padValue)
        imageToTest = preprocess(imageToTest, 0.5, param)
        print("Running FPROP for scale " + "%d" % m + "/" + "%d" % len(multiplier) + "....")
        t0 = time.time()
        score[m] = applydnn(imageToTest, net, nstage, param.boxsize)
        costtime = time.time() - t0
        print("done, elapsed time: " + "%.3f" % costtime + " sec")

        # pool_time = imageToTest.shape[0]/score[m, 0].shape[0]
        # make heatmaps into the size of original image according to pad and scale

        for i in xrange(len(score[m])):
            # change C*H*W -> W*H*C
            score[m][i] = np.transpose(score[m][i], (2, 1, 0))
            score[m][i] = cv2.resize(score[m][i], (imageToTest.shape[1], imageToTest.shape[2]))
            score[m][i] = resize2scaledimg(score[m][i], pad[m], scaledshape, center_s)
            score[m][i] = cv2.resize(score[m][i], (oriImg.shape[0], oriImg.shape[1]))

    # # summing the heatMaps results
    # heatMaps = [[] for item in xrange(nstage)]
    # final_score = [[] for item in xrange(nstage)]
    # for s in xrange(nstage):
    #     final_score[s] = np.zeros(score[0][0].shape)
    #     for m in xrange(len(score)):
    #         final_score[s] = final_score[s] + score[m][s]
    #     heatMaps[s] = np.transpose(final_score[s], (1, 0, 2))
    #     heatMaps[s] /= len(score)
    # # generate prediction from last-stage heatMaps (most refined)
    # prediction = np.zeros((npoints, 2), dtype=np.int)
    # for j in xrange(npoints):
    #     prediction[j, 0], prediction[j, 1] = findmaxinum(final_score[-2][:, :, j])

    # summing the heatMaps results
    heatMaps = []
    final_score = score[0][watchstage]
    # final_score = np.zeros(score[0][0].shape)
    for m in xrange(1, len(score)):
        final_score += score[m][watchstage]
    # generate prediction from last-stage heatMaps (most refined)
    prediction = np.zeros((npoints, 2), dtype=np.int)
    for j in xrange(npoints):
        prediction[j, 0], prediction[j, 1] = findmaxinum(final_score[:, :, j])
    return heatMaps, prediction


def draw_joints(test_image, joints, save_image):
    image = cv2.imread(test_image)
    joints = np.vstack((joints, (joints[8, :] + joints[11, :])/2))
    # bounding box
    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]
    # draw bounding box in red rectangle
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # draw joints in green spots
    for j in xrange(len(joints)):
        cv2.circle(image, (joints[j, 0], joints[j, 1]), 5, (0, 255, 0), 2)
    # draw torso in yellow lines
    torso = [[0, 1], [1, 14], [2, 14], [5, 14]]
    for item in torso:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (0, 255, 255), 2)
    # draw left part in pink lines
    lpart = [[1, 5], [5, 6], [6, 7], [5, 14], [14, 11], [11, 12], [12, 13]]
    for item in lpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 255), 2)
    # draw right part in blue lines
    rpart = [[1, 2], [2, 3], [3, 4], [2, 14], [14, 8], [8, 9], [9, 10]]
    for item in rpart:
        cv2.line(image, (joints[item[0], 0], joints[item[0], 1]), (joints[item[1], 0], joints[item[1], 1]), (255, 0, 0), 2)
    cv2.imwrite(save_image, image)


def draw_gt(test_image, joints, save_image):
    image = cv2.imread(test_image)
    # bounding box
    bbox = [min(joints[::2]), min(joints[1::2]), max(joints[::2]), max(joints[1::2])]
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # draw joints in green spots
    for j in xrange(len(joints) / 2):
        cv2.circle(image, (joints[j * 2], joints[j * 2 + 1]), 5, (0, 255, 0), 2)
    # draw torso in yellow lines
    p1 = [0, 1]
    p2 = [1, 8]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (0, 255, 255), 2)
    # draw left part in pink lines
    p1 = [1, 3, 5, 3, 8, 10]
    p2 = [3, 5, 7, 8, 10, 12]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (255, 0, 255), 2)
    # draw right part in blue lines
    p1 = [1, 2, 4, 2, 8, 9]
    p2 = [2, 4, 6, 8, 9, 11]
    for j in xrange(len(p1)):
        cv2.line(image, (joints[p1[j] * 2], joints[p1[j] * 2 + 1]), (joints[p2[j] * 2], joints[p2[j] * 2 + 1]),
                 (255, 0, 0), 2)
    cv2.imwrite(save_image, image)


if __name__ == '__main__':
    param = Config()

    if param.use_gpu:
        caffe.set_mode_gpu()
        caffe.set_device(param.gpuID)
    net = caffe.Net(param.deployFile, param.caffemodel, caffe.TEST)
    net.name = param.description_short

    # test single image
    # test_image = 'sample_image/roger.png'
    # rectangle = Rect(824, 329, 322, 574)
    # save_image = 'result/roger.jpg'
    # heatMaps, prediction = applymodel(net, test_image, param, rectangle)
    # draw_joints(test_image, prediction, save_image)

    # test a folder
    imageDir = '/deephi_data/img/'
    resultDir = 'result_test/'
    textFile = open('test.txt', 'r')
    # gt 13 joints: 0-head, 1-neck, 2-Rsho, 3-Lsho, 4-Relb, 5-Lelb, 6-Rwri, 7-Lwri, 8-crotch
    # 9-Rkne, 10-Lkne, 11-Rank, 12-Lank
    lines = textFile.readlines()
    precision = np.zeros(13)
    number = 0
    for i in xrange(len(lines)):
        print("image number: " + "%d" % i)
        info = lines[i].split(" ")
        test_image = imageDir + info[0]
        image = cv2.imread(imageDir + info[0])
        if image is None:
            continue
        if len(image.shape) != 3:
            continue
        test_image = imageDir + info[0]
        joints = [int(round(float(item))) for item in info[5:-1]]
        x = min(joints[::2])
        y = min(joints[1::2])
        w = max(joints[::2]) - x
        h = max(joints[1::2]) - y
        rectangle = Rect(max(0, int(x-0.2*w)), max(0, int(y-0.2*h)), int(w*1.4), int(h*1.4))
        save_image = resultDir + "%d" % i + ".jpg"
        heatMaps, prediction = applymodel(net, image, param, rectangle, watchstage=5)
        draw_joints(test_image, prediction, save_image)
        gt_image = resultDir + "%d" % i + "_gt.jpg"
        draw_gt(test_image, joints, gt_image)
        px = joints[::2]
        py = joints[1::2]
        threshold = (px[1]-px[0]) ** 2 + (py[1]-py[0]) ** 2
        prediction[8] = (prediction[8, :] + prediction[11, :])/2
        prediction = prediction[(0, 1, 2, 5, 3, 6, 4, 7, 8, 9, 12, 10, 13), :]
        tempprecision = np.zeros(13)
        number += 1
        writeFile = open(resultDir + "%d" % i + '.txt', 'w+')
        predictFile = open(resultDir + "%d" % i + '_predict.txt', 'w+')
        for j in xrange(len(px)):
            tempprecision[j] = (((px[j]-prediction[j, 0]) ** 2 + (py[j]-prediction[j, 1]) ** 2) * 4) < threshold
            writeFile.write(str(int(tempprecision[j])) + '\n')
            predictFile.write(str(prediction[j, 0]) + ' ' + str(prediction[j, 1]) + '\n')
        precision += tempprecision
        writeFile.close()
        predictFile.close()
        print tempprecision
    if number > 0:
        precision /= number
    print precision
    print np.mean(precision)
    print (np.sum(precision) - precision[8]) / 12





