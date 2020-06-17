# -*- coding: utf-8 -*-
"""
Created on Thu 040518
@author: pariys pourmohammadi
"""
import DataType

def patch_Maker(x, y, n):
    import numpy as np
    i = 0
    y_patches = np.array([y[i:i + n, j:j + n] for i in range(0, nrows, n) for j in range(0, ncols, n)])
    x_patches = np.array([x[:, i:i + n, j:j + n] for i in range(0, nrows, n) for j in range(0, ncols, n)])

    a = x_patches.shape[0]

    while i < a:
        t = x_patches[i][11, :, :]
        if np.sum(t) == 0:
            x_patches = np.delete(x_patches, i, 0)
            y_patches = np.delete(y_patches, i, 0)
            a -= 1
            i += 1
        a -= 1
        i += 1

    y_patches = y_patches.reshape(y_patches.shape[0], 1, y_patches.shape[1], y_patches.shape[2])
    return x_patches, y_patches


###############  Partition data /since our data is class imbalanced we partitioned the data
# using stratified samppling in this function
def partition_data(rows, cols, train, test, valid, base_train):
    import numpy as np
    a = np.zeros((rows, cols))
    ones = np.where(base_train == 1)
    zeros = np.where(base_train == 0)
    len_ones, len_zeros = np.arange(len(ones[0])), np.arange(len(zeros[0]))

    np.random.shuffle(len_ones)
    np.random.shuffle(len_zeros)

    rng_train_one = len_ones[:int(train * len(ones[0]))]
    a[ones[0][rng_train_one], ones[1][rng_train_one]] = 1

    rng_test_one = len_ones[int(train * len(ones[0])):int((train + test) * len(ones[0]))]
    a[ones[0][rng_test_one], ones[1][rng_test_one]] = 2

    rng_val_one = len_ones[int((train + test) * len(ones[0])):]
    a[ones[0][rng_val_one], ones[1][rng_val_one]] = 3

    rng_train_zeros = len_zeros[:int(train * len(zeros[0]))]
    a[zeros[0][rng_train_zeros], zeros[1][rng_train_zeros]] = 1

    rng_test_zeros = len_zeros[int(train * len(zeros[0])):int((train + test) * len(zeros[0]))]
    a[zeros[0][rng_test_zeros], zeros[1][rng_test_zeros]] = 2

    rng_val_zeros = len_zeros[int((train + test) * len(zeros[0])):]
    a[zeros[0][rng_val_zeros], zeros[1][rng_val_zeros]] = 3

    return a


###############partition data and remove nulls
def split(data, check, partition, n):
    import numpy as np
    data = data.reshape((n, 1))

    n_train = ((check != -9999.0) & (partition == 1)).sum()
    n_test = ((check != -9999.0) & (partition == 2)).sum()
    n_valid = ((check != -9999.0) & (partition == 3)).sum()

    data_train = np.zeros((n_train, 1))
    data_test = np.zeros((n_test, 1))
    data_valid = np.zeros((n_valid, 1))

    data_train[:, 0] = data[(check != -9999.0) & (partition == 1)]
    data_test[:, 0] = data[(check != -9999.0) & (partition == 2)]
    data_valid[:, 0] = data[(check != -9999.0) & (partition == 3)]

    data_train = data_train.reshape(data_train.size)
    data_test = data_test.reshape(data_test.size)
    data_valid = data_valid.reshape(data_valid.size)

    return data_train, data_test, data_valid


def line_extract(inFileName):
    file = open(inFileName, 'r')

    all_lines = file.readlines()
    lines = all_lines[0:6][:]

    return lines


###############################################################################
#####################  Load and Preprocess Data  ##############################
###############################################################################
def load(code_path, data_path):

    import os
    os.chdir(code_path)
    import numpy as np

    import DataType

    os.chdir(data_path)

    lines=line_extract('./elev.txt')
    global ncols
    global nrows
    ncols=int(lines[0][14:18])
    nrows=int(lines[1][14:18])
    total_size=nrows * ncols

    landuse_base=np.genfromtxt('landuse_base.txt', delimiter=' ', skip_header=6)
    landuse_base_var = DataType.get_var_names(landuse_base)

    partition = partition_data(nrows, ncols, 0.6, 0.2, 0.2, landuse_base)
    np.array(partition)
    partition = partition.reshape(total_size, 1)

    check = np.genfromtxt('ag_den.txt', delimiter=' ', skip_header=6)
    np.array(check)
    check = check.reshape(total_size, 1)

    n_train = ((check != -9999.0) & (partition == 1)).sum()
    n_test = ((check != -9999.0) & (partition == 2)).sum()
    n_valid = ((check != -9999.0) & (partition == 3)).sum()

    landuse_base = np.genfromtxt('landuse_base.txt', delimiter=' ', skip_header=6)
    landuse_base_var = DataType.get_var_names(landuse_base)

    maj_1000 = np.genfromtxt('maj_1000.txt', delimiter=' ', skip_header=6)
    maj_1000_var = DataType.get_var_names(maj_1000)
    # maj_1000 = None

    maj_100 = np.genfromtxt('maj_100.txt', delimiter=' ', skip_header=6)
    maj_100_var = DataType.get_var_names(maj_100)
    # maj_100 = None

    # take the length of each vaiables names and add it to them up
    final_len = 13 + len(landuse_base_var) + len(maj_1000_var) + len(maj_100_var) + - 3

    data_train = np.zeros((n_train, final_len))
    label_train = np.zeros((n_train, 1))

    data_test = np.zeros((n_test, final_len))
    label_test = np.zeros((n_test, 1))

    data_val = np.zeros((n_valid, final_len))
    label_val = np.zeros((n_valid, 1))

    a=0

    ag_den = np.genfromtxt('ag_den.txt', delimiter=' ', skip_header=6)
    ag_den = DataType.normalize(ag_den)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(ag_den, check, partition, ag_den.size)
    # ag_den = None
    a+=1

    aspect = np.genfromtxt('aspect.txt', delimiter=' ', skip_header=6)
    aspect = DataType.normalize(aspect)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(aspect, check, partition, aspect.size)
    # aspect = None
    a+=1

    dev_den = np.genfromtxt('dev_den.txt', delimiter=' ', skip_header=6)
    dev_den = DataType.normalize(dev_den)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(dev_den, check, partition, dev_den.size)
    # dev_den = None
    a+=1

    dist_dev = np.genfromtxt('dist_dev.txt', delimiter=' ', skip_header=6)
    dist_dev = DataType.normalize(dist_dev)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(dist_dev, check, partition, dist_dev.size)
    # dist_dev = None
    a+=1

    dist_mine = np.genfromtxt('dist_mines.txt', delimiter=' ', skip_header=6)
    dist_mine = DataType.normalize(dist_mine)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(dist_mine, check, partition, dist_mine.size)
    # dist_mine = None
    a+=1

    dist_pascrop = np.genfromtxt('dist_pastcrop.txt', delimiter=' ', skip_header=6)
    dist_pascrop = DataType.normalize(dist_pascrop)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(dist_pascrop, check, partition, dist_pascrop.size)
    # dist_pascrop = None
    a += 1

    dist_rec = np.genfromtxt('dist_rec.txt', delimiter=' ', skip_header=6)
    dist_rec = DataType.normalize(dist_rec)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(dist_rec, check, partition, dist_rec.size)
    # dist_rec = None
    a += 1

    dist_road = np.genfromtxt('dist_road.txt', delimiter=' ', skip_header=6)
    dist_road = DataType.normalize(dist_road)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(dist_road, check, partition, dist_road.size)
    # dist_road = None
    a += 1

    dist_stream = np.genfromtxt('dist_stream.txt', delimiter=' ', skip_header=6)
    dist_stream = DataType.normalize(dist_stream)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(dist_stream, check, partition, dist_stream.size)
    # dist_stream = None
    a += 1

    elev = np.genfromtxt('elev.txt', delimiter=' ', skip_header=6)
    elev = DataType.normalize(elev)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(elev, check, partition, elev.size)
    # elev = None
    a += 1

    pop_2000 = np.genfromtxt('pop_2000.txt', delimiter=' ', skip_header=6)
    pop_2000 = DataType.normalize(pop_2000)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(pop_2000, check, partition, pop_2000.size)
    # pop_2000 = None
    a += 1

    slope = np.genfromtxt('slope.txt', delimiter=' ', skip_header=6)
    slope = DataType.normalize(slope)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(slope, check, partition, slope.size)
    # slope = None
    a += 1

    OG_den_kernel = np.genfromtxt('og_den.txt', delimiter=' ', skip_header=6)
    OG_den_kernel = DataType.normalize(OG_den_kernel)
    data_train[:, a], data_test[:, a], data_val[:, a] = \
        split(OG_den_kernel, check, partition, OG_den_kernel.size)
    # OG_den_kernel = None
    a += 1

    cat_file = open(data_path + '/categories.txt', 'w')
    fedlands = np.genfromtxt('fedland.txt', delimiter=' ')
    fedlands_var = DataType.get_var_names(fedlands)
    if (len(fedlands_var) > 1):
       fedlands = DataType.get_dummy(fedlands, fedlands_var[1])
       data_train[:,a], data_test[:,a] , data_val[:,a] = \
           split(fedlands, check, partition, fedlands.size)
       a +=1
    cat_file.write('fedLand of this region has '+str(len(fedlands_var)) +' classes\n')
    # fedlands = None

    uarea = np.genfromtxt('uarea2000.txt', delimiter=' ')
    uarea_var = DataType.get_var_names(uarea)
    if (len(uarea_var)>1):
       uarea = DataType.get_dummy(uarea, uarea_var[1])
       data_train[:,a], data_test[:,a] , data_val[:,a] = \
           split(uarea, check, partition, uarea.size)
       a +=1
    cat_file.write('Urban area of this region has '+str(len(uarea_var)) +' classes\n')
    # uarea = None

    Carea = np.genfromtxt('carea2000.txt', delimiter=' ')
    Carea_var = DataType.get_var_names(Carea)
    if (len(Carea_var) > 1):
       Carea = DataType.get_dummy(Carea, Carea_var[1])
       data_train[:,a], data_test[:,a] , data_val[:,a] = \
           split(Carea, check, partition, Carea.size)
       a+=1
    cat_file.write('Clustered Urban area of this region has '+str(len(Carea_var)) +' classes\n')
    # Carea = None

    state = np.genfromtxt('state.txt', delimiter=' ')
    state_var = DataType.get_var_names(state)
    if(len(state_var) > 1):
       state_dummy = {}
       for t in range(1, len(state_var)):
           state_dummy['%s' % t] = DataType.get_dummy(state, state_var[t])
           data_train[:,a], data_test[:,a] , data_val[:,a] = \
           split(state_dummy['%s' % t], check, partition, state_dummy['%s' % t].size)
           state_dummy['%s' % t] = None
           a+=1
           print (a)
    #        a = a+ len(state_var) -1
    cat_file.write('state of this region has '+str(len(state_var)) +' classes\n')
    state = None

    counties = np.genfromtxt('counties.txt', delimiter=' ')
    counties_var = DataType.get_var_names(counties)
    if(len(counties_var) >1):
       counties_dummy = {}
       for t in range(1, len(counties_var)):
           counties_dummy['%s' % t] = DataType.get_dummy(counties, counties_var[t])
           data_train[:,a], data_test[:,a] , data_val[:,a] = \
           split(counties_dummy['%s' % t], check, partition, counties_dummy['%s' % t].size)
           counties_dummy['%s' % t] = None
           a+=1
           print (a)
    cat_file.write('counties of this region has '+str(len(counties_var)) +' classes\n')
    print(a)
    counties = None

    landuse_base = np.genfromtxt('landuse_base.txt', delimiter=' ', skip_header=6)
    landuse_base_var = DataType.get_var_names(landuse_base)
    if (len(landuse_base_var) > 1):
        landuse_base = DataType.get_dummy(landuse_base, landuse_base_var[1])
        data_train[:,a], data_test[:, a], data_val[:, a] = \
            split(landuse_base, check, partition, landuse_base.size)
        a += 1

    else:
        landuse_base = DataType.get_dummy(landuse_base, landuse_base_var[0])
    cat_file.write('landuse_base of this region has ' + str(len(landuse_base_var)) + ' classes\n')

    maj_1000 = np.genfromtxt('maj_1000.txt', delimiter=' ', skip_header=6)
    maj_1000_var = DataType.get_var_names(maj_1000)
    if (len(maj_1000_var) > 1):
        maj_1000_dummy = {}
        for t in range(1, len(maj_1000_var)):
            maj_1000_dummy['%s' % t] = DataType.get_dummy(maj_1000, maj_1000_var[t])
            data_train[:, a], data_test[:, a], data_val[:, a] = \
                split(maj_1000_dummy['%s' % t], check, partition, maj_1000_dummy['%s' % t].size)
            maj_1000_dummy['%s' % t] = None
            a += 1
            print(a)
    cat_file.write(
        'major classes of land within 1000 meters of this region has ' + str(len(maj_1000_var)) + ' classes\n')


    maj_100 = np.genfromtxt('maj_100.txt', delimiter=' ', skip_header=6)
    maj_100_var = DataType.get_var_names(maj_100)
    if (len(maj_100_var) > 1):
        maj_100_dummy = {}
        for t in range(1, len(maj_100_var)):
            maj_100_dummy["%s" % t] = DataType.get_dummy(maj_100, maj_100_var[t])
            data_train[:, a], data_test[:, a], data_val[:, a] = \
                split(maj_100_dummy['%s' % t], check, partition, maj_100_dummy['%s' % t].size)
            maj_100_dummy['%s' % t] = None
            a += 1
            # print(a)

    cat_file.write('major classes of land within 100 meters of this region has ' + str(len(maj_100_var)) + ' classes\n')
    # maj_100 = None

    cat_file.close()

    landuse_final = np.genfromtxt('landuse_final.txt', delimiter=' ', skip_header=6)
    landuse_final_var = DataType.get_var_names(landuse_final)
    landuse_final = DataType.get_dummy(landuse_final, landuse_final_var[0])
    label_train[:, 0], label_test[:, 0], label_val[:, 0] = \
        split(landuse_final, check, partition, landuse_final.size)
    # landuse_final = None

    np.save("partition", partition)
    np.save("check", check)

    np.save("data_train", data_train)
    np.save("data_test", data_test)
    np.save("data_val", data_val)

    np.save("label_train", label_train)
    np.save("label_test", label_test)
    np.save("label_val", label_val)

    xTrainValid = np.concatenate((data_train, data_val), axis=0)
    yTrainValid = np.concatenate((label_train, label_val), axis=0)

    np.save("xTrainValid", xTrainValid)
    np.save("yTrainValid", yTrainValid)