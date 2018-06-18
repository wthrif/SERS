import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from tkinter import Tk
from tkinter.filedialog import askdirectory
import os
import time
from keras.layers import Conv2D, Add, SeparableConv2D, BatchNormalization, Activation, Flatten, MaxPooling2D, Input, AveragePooling2D, Reshape, Dropout, LeakyReLU, UpSampling2D
from sklearn.metrics import r2_score, accuracy_score
from keras.models import Model, load_model
from keras import backend as K
from keras.optimizers import adam
start_time = time.time()

#define useful parameters
features = 1011
group = 8
copies = 32
activation = 0.25
dropout = 0.3
test = False
patience = 4

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


# import data
#filename = askdirectory()
filename = "/home/ragan/Downloads/chip_test/chip18"
file_list = sorted(glob.glob(filename + '/*.txt'))
list_array = sorted(glob.glob(filename + '/*.txt'))

#validation data
#filename_validation = askdirectory()
filename_validation = "/home/ragan/Downloads/chip_test/chip18"
file_list_validation = sorted(glob.glob(filename_validation + '/*.txt'))
list_array_validation = sorted(glob.glob(filename_validation + '/*.txt'))


#load data
X_validation_new = np.load("/home/ragan/Downloads/chip_test/chip15/matrix/x_val.npy")
y_validation_new = np.load("/home/ragan/Downloads/chip_test/chip15/matrix/y_val.npy")
X_validation_tensor = np.load("/home/ragan/Downloads/chip_test/chip15/matrix/x_val.npy")
y_validation_bundle = np.load("/home/ragan/Downloads/chip_test/chip15/matrix/y_val.npy")


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


print("data import --- %s seconds ---" % (time.time() - start_time))


#neural net
input = Input(shape=(group, group, features))

x = Dropout(dropout)(input)

x = Dense(512, kernel_initializer='normal', activation='linear')(input)
x = LeakyReLU(alpha=activation)(x)

x = Dropout(dropout)(x)

x = Dense(256, kernel_initializer='normal', activation='linear')(x)
x = LeakyReLU(alpha=activation)(x)

x = Dropout(dropout)(x)

x = Dense(128, kernel_initializer='normal', activation='linear')(x)
x = LeakyReLU(alpha=activation)(x)

x = Dropout(dropout)(x)

x = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='linear', padding='same')(x)
x = LeakyReLU(alpha=activation)(x)

x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same')(x)
x = LeakyReLU(alpha=activation)(x)

x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='linear', padding='same')(x)
x = LeakyReLU(alpha=activation)(x)

x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same')(x)
x = LeakyReLU(alpha=activation)(x)

x = SeparableConv2D(256, (2, 2), padding='same', use_bias=False)(x)
x = Activation('relu')(x)

x = SeparableConv2D(256, (2, 2), padding='same', use_bias=False)(x)
x = Activation('relu')(x)

for i in range(8):
	residual = x
	x = Activation('relu')(x)
	x = SeparableConv2D(256, (2, 2), padding='same', use_bias=False)(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(256, (2, 2), padding='same', use_bias=False)(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(256, (2, 2), padding='same', use_bias=False)(x)
	x = Add()([x, residual])

x = Activation('relu')(x)

x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)

x = Dense(16, kernel_initializer='normal', activation='linear')(x)
x = LeakyReLU(alpha=activation)(x)

x = Dropout(dropout)(x)

output = Dense(1, kernel_initializer='normal')(x)

model = Model(inputs=input, outputs=output)
#optimizer = adam(lr=1e-4)
model.compile(loss='mean_squared_error', optimizer="adam", metrics=['acc', coeff_determination])
model.summary()

#Train model, early stopping, history
best_loss_validation = 10000000
best_acc_validation = -1
loss_history_train = []
acc_history_train = []
r2_history_train = []
loss_history_validation = []
acc_history_validation = []
r2_history_validation = []
for i in range(1000):
    for j in range(copies):
        loss_list_train = []
        acc_list_train = []
        r2_list_train = []
        loss_list_validation = []
        acc_list_validation = []
        r2_list_validation = []
        history_train = model.train_on_batch(np.load(f"/home/ragan/Downloads/chip_test/chip14/matrix/x_{j}.npy"), np.load(f"/home/ragan/Downloads/chip_test/chip14/matrix/y_{j}.npy"))
        history_validation = model.test_on_batch(X_validation_tensor, y_validation_bundle)
        loss_list_train = np.append(loss_list_train, history_train[0])
        acc_list_train = np.append(acc_list_train, history_train[1])
        r2_list_train = np.append(r2_list_train, history_train[2])
        loss_list_validation = np.append(loss_list_validation, history_validation[0])
        acc_list_validation = np.append(acc_list_validation, history_validation[1])
        r2_list_validation = np.append(r2_list_validation, history_validation[2])
    mean_loss_train = sum(loss_list_train) / len(loss_list_train)
    loss_history_train = np.append(loss_history_train, mean_loss_train)
    mean_acc_train = sum(acc_list_train) / len(acc_list_train)
    acc_history_train = np.append(acc_history_train, mean_acc_train)
    mean_r2_train = sum(r2_list_train) / len(r2_list_train)
    r2_history_train = np.append(r2_history_train, mean_r2_train)
    mean_loss_validation = sum(loss_list_validation) / len(loss_list_validation)
    loss_history_validation = np.append(loss_history_validation, mean_loss_validation)
    mean_acc_validation = sum(acc_list_validation) / len(acc_list_validation)
    acc_history_validation = np.append(acc_history_validation, mean_acc_validation)
    mean_r2_validation = sum(r2_list_validation) / len(r2_list_validation)
    r2_history_validation = np.append(r2_history_validation, mean_r2_validation)
    if mean_loss_validation < best_loss_validation:
        best_loss_validation = mean_loss_validation
        rounds_without_improvement = 0
        model.save("/home/ragan/Downloads/r_800_new/results/best_model.h5")
        print(mean_loss_validation, mean_acc_validation, mean_r2_validation)
    else:
        rounds_without_improvement +=1
        print("No Improvement")
    if rounds_without_improvement == patience:
        break
print("training NN--- %s seconds ---" % (time.time() - start_time))

#prepare validation data for errorplot
model = load_model("/home/ragan/Downloads/r_800_new/results/best_model.h5", custom_objects={'coeff_determination': coeff_determination})
predictions_validation = model.predict(X_validation_new)
loss_validation = model.evaluate(X_validation_new, y_validation_new)
print(loss_validation)
guess_validation = np.column_stack((predictions_validation, y_validation_new))
e_one_validation = []
e_validation = []
m_one_validation = []
m_validation = []
for i in range(0, len(list_array_validation)):
	e_one_validation = np.std([predictions_validation for (predictions_validation,y_validation_new) in guess_validation if y_validation_new == int(os.path.basename(file_list_validation[i])[3:5])])
	e_validation = np.append(e_validation, e_one_validation)
	m_one_validation = np.mean([predictions_validation for (predictions_validation, y_validation_new) in guess_validation if y_validation_new == int(os.path.basename(file_list_validation[i])[3:5])])
	m_validation = np.append(m_validation, m_one_validation)
real_validation = []
for i in range(0, len(list_array_validation)):
    real_validation = np.append(real_validation, int(os.path.basename(file_list_validation[i])[3:5]))
for i in range(0, len(list_array_validation)):
    print(m_validation[i],e_validation[i])
validation_r2 = format(r2_score(y_validation_new, predictions_validation), ".2f")
print("validation --- %s seconds ---" % (time.time() - start_time))


#np.save("/home/ragan/Downloads/chip_test/chip4/results/lowest_loss", loss_validation[0])

#plot error vs epoch
plt.subplot(311)
plt.plot(loss_history_train)
plt.plot(loss_history_validation)
plt.title('Model Accuracy/Loss')
plt.ylabel('Loss')
plt.subplot(312)
plt.plot(acc_history_train)
plt.plot(acc_history_validation)
plt.ylabel('Accuracy')
plt.subplot(313)
plt.plot(r2_history_train)
plt.plot(r2_history_validation)
plt.ylim(-1,1)
plt.ylabel('r2')
plt.xlabel('epoch')


#plot predictions
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
x_labels = []
for i in range(0, len(list_array_validation)):
    x_labels = np.append(x_labels, os.path.basename(file_list_validation[i])[-10:-3])
labels = f7(x_labels)
plt.xticks(np.unique(real_validation), labels)
plt.yticks(np.unique(real_validation), labels)
plt.ylabel('Predicted Concentration', fontsize=16)
plt.xlabel('Real Concentration', fontsize=16)
plt.title(f'Methylene Blue', fontsize=20)
plt.margins(tight=False)
plt.errorbar(real_validation, m_validation, yerr=e_validation, fmt='o', label=f'Validation Data r^2: {validation_r2}')
plt.plot((0,1,2,3,4,5,6,7,8,9), label='Ideal Fit')
legend = plt.legend(loc='upper left', shadow=True, fontsize=14)
plt.show()
