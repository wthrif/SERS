import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import time
from joblib import Parallel, delayed
import multiprocessing



directory = "/media/ragan/19d71545-4e3f-4c1b-892b-8ab42e708096/transfer/"
save_directory = "/media/ragan/19d71545-4e3f-4c1b-892b-8ab42e708096/transfer/matrixboth/"
analyte = 1
copies = 32
features = 1011
start_time = time.time()
num_folds = 10
group = 8
num_cores = 8



def circular_array(matrix, index):
    counter = 0

    # Start at the next array
    index += 1
    i = index

    compiled_array = 0
    # loop starts the next element, and ends at the element before
    while i < num_folds + (index - 1):

        # Creates shape if its the first time in the loop
        if i == index:
            compiled_array = matrix[i % num_folds]

        # If not first time, stacks on the bottom
        else:
            compiled_array = np.concatenate((compiled_array, matrix[i % num_folds]))

        # Move loop and counter
        i += 1
        counter += 1

    #print("Number of Arrays added: {}".format(counter))
    #print("Start Index: {}\n".format(index))
    return compiled_array



def bundler(data, val, p, analyte, copies, group, k):

    label = []
    for j in range(0, len(file_list)):
        for l in range(0, analyte):
            label_temp = int(os.path.basename(file_list[j])[2 + 3 * l:5 + 3 * l])
            label = np.append(label, label_temp)
    label = np.reshape(label, (-1, analyte))
    label = np.unique(label, axis=0)

    j = 0
    single_conc = np.array([data[z] for z in range(len(data)) if np.array_equal(data[z, 1:analyte + 1], label[j])])
    single_conc_temp = single_conc[group * group * 0:group * group * (0 + 1)]
    y_temp = single_conc[group * group * 0:group * group * (0 + 1)][0, 1:analyte + 1]
    for i in range(1, len(single_conc[:, 0]) // (group * group)):
        x_temp = single_conc[group * group * i:group * group * (i + 1)]
        single_conc_temp = np.append(single_conc_temp, x_temp, axis=0)
        y_single_conc = x_temp[0, 1:analyte + 1]
        y_temp = np.append(y_temp, y_single_conc, axis=0)
    x = np.reshape(single_conc_temp, (-1, group, group, features + 1 + analyte))
    y = y_temp

    for j in range(1, len(label)):
        single_conc = np.array(
            [data[z] for z in range(len(data)) if np.array_equal(data[z, 1:analyte + 1], label[j])])
        single_conc_temp = single_conc[group * group * 0:group * group * (0 + 1)]
        y_temp = single_conc[group * group * 0:group * group * (0 + 1)][0, 1:analyte + 1]
        for i in range(1, len(single_conc[:, 0]) // (group * group)):
            x_temp = single_conc[group * group * i:group * group * (i + 1)]
            single_conc_temp = np.append(single_conc_temp, x_temp, axis=0)
            y_single_conc = x_temp[0, 1:analyte + 1]
            y_temp = np.append(y_temp, y_single_conc, axis=0)
        x = np.append(x, np.reshape(single_conc_temp, (-1, group, group, features + 1 + analyte)), axis=0)
        y = np.append(y, y_temp, axis=0)
    x = x[:, :, :, 1 + analyte:]

    if val==True:
        np.save(save_directory+f"x_val_{p}_{k}", x)
        np.save(save_directory+f"y_val_{p}_{k}", y)
    else:
        np.save(save_directory + f"x_{p}_{k}", x)
        np.save(save_directory + f"y_{p}_{k}", y)
    #print("data --- %s seconds ---" % (time.time() - start_time))



# import data
file_list = sorted(glob.glob(directory + '/*.txt'))
x = [pd.read_table(f, header=None, usecols=[3]) for f in file_list]
x = [np.array(l) for l in x]
y = []
# define the labels
for i in range(0, len(file_list)):
    y_temp = np.repeat(i, len(x[i]) // features)
    y = np.append(y, y_temp, axis=0)

for l in range(0, analyte):
    y_temp = []
    for i in range(0, len(file_list)):
        y_temp = np.append(y_temp, np.repeat(int(os.path.basename(file_list[i])[2 + 3 * l:5 + 3 * l]),
                                             len(x[i]) // features))
    y = np.column_stack((y, y_temp))

x = np.concatenate(x).ravel()
x = np.reshape(x, (-1, features))
x = np.transpose(MinMaxScaler().fit_transform(np.transpose(x)))

data = np.column_stack((y, x))
del x, y, y_temp
np.random.shuffle(data)
data = np.array_split(data, num_folds)
data = np.array(data)

#for m in range(num_folds): print("index {},first map number {}".format(m,data[m][0,0]))

# Here is where we can do the outer loop, changing the starting points
# Adding to the big Collection
for p in range(0, num_folds):
    compiled_array_collection = []
    compiled_array_collection.append(circular_array(data, p))
    compiled_array_collection = np.array(compiled_array_collection)
    compiled_array_collection = np.array_split(compiled_array_collection[0],num_folds)
    Parallel(n_jobs=num_cores)(
        delayed(bundler)(compiled_array_collection[0], True, p, analyte, copies, group, k) for k in range(copies))
    compiled_array_collection = np.array(compiled_array_collection)[1:]
    compile_temp = 0
    for m in range(num_folds-1):
        compile_temp = np.append(compile_temp,compiled_array_collection[m])
    compiled_array_collection = compile_temp[1:].reshape(-1,features+2)
    del compile_temp
    Parallel(n_jobs=num_cores)(
        delayed(bundler)(compiled_array_collection, False, p, analyte, copies, group, k) for k in range(copies))
    #for m in range(num_folds): print("index {},first map number {}".format(m, compiled_array_collection[m][0,0]))
    print(f"fold {p} done")




print("data --- %s seconds ---" % (time.time() - start_time))

