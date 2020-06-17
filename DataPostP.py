# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:53:26 2020

@author: pariya pourmohammadi
"""

import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir('D:/MyCourseWorks WVU/Research/DownloadsNRACpc/EncoderDecoderIEEE')
elev = np.genfromtxt('./elev.txt', delimiter=' ')

elev_patches = np.array(
    [elev[i:i + 32, j:j + 32] for i in range(0, elev.shape[0], int(32)) for j in range(0, elev.shape[1], int(32))])
exc_ind = []
for i in range(elev_patches.shape[0]):
    if (np.sum(elev_patches[i]) == -10238976.0):
        exc_ind.append(i)

np.random.seed(1364)

# generate random indices for the partitioned data
range_ind = np.arange(52455)

range_ind = [x for x in range_ind if x not in exc_ind]

np.random.shuffle(range_ind)
# ind_train= range_ind[:int(len(range_ind)*.66)]
# ind_test= range_ind[int(len(range_ind)*.66):]
train = np.load('D:\MyCourseWorks WVU\Research\DownloadsNRACpc\EncoderDecoderIEEE\Data\y_patches.npy')
y_train = train[range_ind[0:int(len(range_ind) * .66)], :, :]
y_test = train[range_ind[int(len(range_ind) * .66):], :, :]

from matplotlib.colors import LinearSegmentedColormap

# define the colorMap
cmap = LinearSegmentedColormap.from_list('wr', ["w", "r"], N=256)
cmap1 = LinearSegmentedColormap.from_list('wg', ["w", "g"], N=256)

# reverse blocks back to map
def texmaker(target, name, fmt, flag_train, flag_file):
    mask = np.zeros(shape=(8608, 6240), dtype=float)
    blocks_in_row = 6240 / 32

    count = 0
    for i in range(y_train.shape[0]):
        #    print(i)
        ind = range_ind[i]
        row = (int(ind / blocks_in_row)) * 32
        col = (int(ind % blocks_in_row)) * 32
        if flag_train:
            if len(target.shape) == 4:
                mask[row:(row + 32), col:(col + 32)] = y_train[i, :, :]
                count += 1

            if len(target.shape) == 5:
                mask[row:(row + 32), col:(col + 32)] = y_train[i, :, :]

        else:
            mask[row:(row + 32), col:(col + 32)] = np.zeros((32, 32))

    #    print(count)

    for i in range(y_test.shape[0]):
        #    print(i)
        ind = range_ind[i + y_train.shape[0]]
        row = (int(ind / blocks_in_row)) * 32
        col = (int(ind % blocks_in_row)) * 32

        if len(target.shape) == 4:
            mask[row:(row + 32), col:(col + 32)] = target[i, 0, :, :]

        if len(target.shape) == 5:
            mask[row:(row + 32), col:(col + 32)] = target[i, 0, :, :, 0]

        if len(target.shape) == 3:
            mask[row:(row + 32), col:(col + 32)] = target[i, :, :]
        #        print(i)

    mask = mask[:8581, :6220]
    # np.savetxt("mask.txt",mask, fmt='%d', delimiter=' ')
    if flag_file:
        mask[elev == -9999] = int(-9999)
        all_lines = open("Lines.txt", "r")
        lines = all_lines.readlines()
        all_lines.close()

        header = ''
        for line in lines: header = header + line
        np.savetxt(name, mask, fmt=fmt, delimiter=' ', header=header, comments='')

    return mask

#visualize the results
def visualize(t):
    blocks_row = 100
    shape = np.zeros(shape=(2400, 3200), dtype=float)
    for i in range(y_test.shape[0]):
        #    print(i)
        row = (int(i / blocks_row)) * 32
        col = (int(i % blocks_row)) * 32

        if len(t.shape) == 4:
            shape[row:(row + 32), col:(col + 32)] = t[i, 0, :, :]
            print(np.sum(t[i, 0, :, :]))

        if len(t.shape) == 5:
            shape[row:(row + 32), col:(col + 32)] = t[i, 0, :, :, 0]

        if len(t.shape) == 3:
            shape[row:(row + 32), col:(col + 32)] = t[i, :, :]

        #            print(i)

    plt.imshow(shape)

    return shape

# moving average filter
def moving_average(shape, n):
    from scipy import ndimage
    density = ndimage.uniform_filter(shape, size=n, mode='constant')
    return density


# the misses (FN/Type II Error)
def miss(res, ground_t):
    shape = np.zeros(shape=(8581, 6220), dtype=float)
    shape[(res == 0) & (ground_t == 1)] = 1.0

    return (shape)


#the hits (TP)
def hit(res, ground_t):
    shape = np.zeros(shape=(8581, 6220), dtype=float)
    shape[(res == 1) & (ground_t == 1)] = 1.0
    return (shape)


# the false alarms (FP/ Type I error)
def false_alarm(res, ground_t):
    shape = np.zeros(shape=(8581, 6220), dtype=float)
    shape[(res == 1) & (ground_t == 0)] = 1.0
    return (shape)

#dice_coef calculation
def dice_coef(input1, input2):
    a = hit(input1, input2)
    b = false_alarm(input1, input2)
    c = miss(input1, input2)
    dice_coef = 2 * np.sum(a) / (2 * np.sum(a) + np.sum(b) + np.sum(c))

    return (dice_coef)



#auc calculation
def auc(inpu1, inpu2):
    TP = np.sum(hit(S1, inpu2))
    FN = np.sum(miss(inpu1, inpu2))
    FP = np.sum(false_alarm(inpu1, inpu2))
    TN = 7407 * 32 * 32 - (TP + FP + FN)

    Recall = TP / (TP + FN)
    Percision = TP / (TP + FP)
    Specificity = TN / (TN + FP)
    FPR = 1 - Specificity

    outAUC = {'Recall': Recall, 'Percision': Percision, 'Specificity': Specificity, 'FPR': FPR}

    return (outAUC)


#Write results to file
def wrt_file(mask, name, fmt):
    mask[elev == -9999] = int(-9999)
    all_lines = open("Lines.txt", "r")
    lines = all_lines.readlines()
    all_lines.close()

    header = ''
    for line in lines: header = header + line
    np.savetxt(name, mask, fmt=fmt, delimiter=' ', header=header, comments='')