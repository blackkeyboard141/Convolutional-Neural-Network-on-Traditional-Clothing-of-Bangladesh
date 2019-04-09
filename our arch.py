#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:37:14 2018

@author: rasik
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#loss: 0.3953 - acc: 0.8530 - val_loss: 0.3844 - val_acc: 0.8913
#(accuracy on train        )  (accuracy on test                 )

x = 115

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (x, x, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#loss: 0.2199 - acc: 0.9259 - val_loss: 0.3725 - val_acc: 0.9114

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))
#loss: 0.1734 - acc: 0.9416 - val_loss: 0.3957 - val_acc: 0.9053
#with rmsprop loss: 0.1977 - acc: 0.9242 - val_loss: 0.3992 - val_acc: 0.9138

#classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dropout(0.5))

classifier.add(Dense(units = 5, activation = 'softmax'))

from keras.optimizers import SGD
lrate = 0.01
decay = lrate/30
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

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

classifier.fit_generator(training_set,
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
    result = classifier.predict(testI)
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
                          title='Confusion matrix',
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
                      title='Confusion matrix, Our Tuned architecture')

plt.show()