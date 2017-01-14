# --------------------------------------------------------
# Demo code of "Convolutional Pose Machines"
# python version
# Written by Lu Tian
# --------------------------------------------------------

"""Calculate mAP of CPM"""


import os
import numpy as np


if __name__ == '__main__':
    resultDir = 'result_test_single/'
    npoints = 13
    number = 0
    precision = np.zeros(npoints)
    for i in xrange(2000):
        temp = resultDir + "%d" % i + '.txt'
        if not os.path.exists(temp):
            continue
        textFile = open(temp, 'r')
        lines = textFile.readlines()
        number += 1
        for j in xrange(npoints):
            precision[j] += int(lines[j])
    if number > 0:
        precision /= number
    print precision
    print np.mean(precision)
    print (np.sum(precision) - precision[8])/(npoints - 1)
