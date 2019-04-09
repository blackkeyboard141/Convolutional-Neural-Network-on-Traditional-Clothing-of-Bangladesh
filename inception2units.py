#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 13:38:29 2018

@author: rasik
"""

import keras
from keras.layers import Input
x=32

input_img = Input(shape = (x, x, 3))

from keras.layers import Conv2D, MaxPooling2D
tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_3)



output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)

tower_4 = Conv2D(64, (1,1), padding='same', activation='relu')(output)
tower_4 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_4)
tower_5 = Conv2D(64, (1,1), padding='same', activation='relu')(output)
tower_5 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_5)
tower_6 = MaxPooling2D((3,3), strides=(1,1), padding='same')(output)
tower_6 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_6)

output2 = keras.layers.concatenate([tower_4, tower_5, tower_6], axis = 3)

from keras.layers import Flatten, Dense
output2 = Flatten()(output2)

out    = Dense(5, activation='softmax')(output2)

from keras.models import Model
model = Model(inputs = input_img, outputs = out)
# print model.summary()

from keras.optimizers import SGD
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer= 'sgd', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (x, x),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('data/validation',
                                            target_size = (x , x),
                                            batch_size = 32,
                                            class_mode = 'categorical')

labels = test_set.class_indices

model.fit_generator(training_set,
                         samples_per_epoch = 1494,
                         nb_epoch = 30,
                         validation_data = test_set,
                         nb_val_samples = 389)

from sklearn.metrics import confusion_matrix
import numpy as np 
from keras.preprocessing import image

#X-train is training images , y_train training labels
#X-test is testing images , y_test testing labels

pred = []



for i in range(1,51):
    if i<21 or i>30:
        stri = 'data/single/{}.jpg'.format(i)
    else:
        stri = 'data/single/{}.jpeg'.format(i)
    testI = image.load_img(stri, target_size=(x,x))
    testI = image.img_to_array(testI)
    testI = np.expand_dims(testI , axis = 0)
    result = model.predict(testI)
    if result[0][0]==1.0:
        pred.append(0)
    elif result[0][1]==1.0:
        pred.append(1)
    elif result[0][2]==1.0:
        pred.append(2)
    elif result[0][3]==1.0:
        pred.append(3)
    elif result[0][4]==1.0:
        pred.append(4)

pred

n_classes = 5
y_test =[0,0,0,0,0,0,0,0,0,0,
         1,1,1,1,1,1,1,1,1,1,
         2,2,2,2,2,2,2,2,2,2,
         3,3,3,3,3,3,3,3,3,3,
         4,4,4,4,4,4,4,4,4,4]

y_pred = pred 

cnf_matrix = confusion_matrix(y_test, y_pred, labels=range(n_classes))

print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix for Inception model',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
class_names = ['fatua','kamiz','lungi','panjabi', 'saree']
#class_names = {'panjabi','lungi','saree','kamiz', 'fatua'}

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()