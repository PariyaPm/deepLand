import DataPostP
import numpy as np


np.random.seed(1364)
#generate random indices for the partitioned data
range_ind = np.arange(52455)

range_ind = [x for x in range_ind if x not in exc_ind]

np.random.shuffle(range_ind)
train = np.load('D:\MyCourseWorks WVU\Research\DownloadsNRACpc\EncoderDecoderIEEE\Data\y_patches.npy')
y_train = train[range_ind[0:int(len(range_ind)*.66)],:,:]
y_test= train[range_ind[int(len(range_ind)*.66):],:,:]


labeledData = ['y_classdeepLandS.npy', 'y_classdeepLandU.npy']
scoreData = ['y_predictdeepLandS.npy', 'y_predictdeepLandU.npy']

for file1 in scoreData:
    res1 = np.load('Results/' + file1)
    DataPostP.texmaker(res1, '.txtFiles/' + file1[2:-4] + '.txt', '%.6f', False, True)

for file in labeledData:
    res = np.load('Results/' + file)
    DataPostP.texmaker(res, ('.txtFiles/' + file[2:-4] + '.txt'), '%d', False, True)

groundTruth = DataPostP.texmaker(y_test, 'y_test.txt', '%d', False, False)
L1 = DataPostP.texmaker(np.load('Results/' + 'y_classdeepLandS.npy'), ('.txtFiles/' + file[2:-4] + '.txt'), '%d', False, False)
L2 = DataPostP.texmaker(np.load('Results/' + 'y_classdeepLandU.npy'), ('.txtFiles/' + file[2:-4] + '.txt'), '%d', False, False)


groundTruth = DataPostP.texmaker(y_test, 'y_test.txt', '%d', False, False)
S1 = DataPostP.texmaker(np.load('Results/' + 'y_predictdeepLandS.npy'), ('.txtFiles/' + file[2:-4] + '.txt'), '%.6f', False,
              False)
S2 = DataPostP.texmaker(np.load('Results/' + 'y_predictdeepLandU.npy'), ('.txtFiles/' + file[2:-4] + '.txt'), '%.6f', False,
              False)

