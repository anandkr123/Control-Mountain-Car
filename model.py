import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

cwd = os.getcwd()
join_dir = lambda *args: os.path.join(*args)
loadedImages = []
valImages = []
outputVectors=[]
valVectors=[]


def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def list_files(directory):
    files = os.listdir(join_dir(cwd, directory))
    files = filter(lambda x: not x.startswith('.'), files)
    return list(files)


def prepare_images(dir_name):
    """
    :param dir_name: prepare train and val data set images of specific class
    :return:
    """
    single_class_images = len(list_files(join_dir('Dataset', 'left')))
    val_count = 0

   # Load Images From specific class dir
    for i in range(1, single_class_images+1):
        image = cv2.imread(join_dir(cwd, 'resized', dir_name, f'{dir_name}_{i}.png'), 0)
        image = np.expand_dims(image, axis=-1)

    # 100 val images from specific class
        if val_count >= 400:
            valImages.append(image)
        else:
            loadedImages.append(image)
        val_count = val_count+1


def prepare_output_vectors():
    # divide by 3 , since there are 3 classes
    single_class_vectors = int(len(loadedImages)/3)
    # labels for left push
    for i in range(1, single_class_vectors+1):
        outputVectors.append([1, 0, 0])

    # labels for right push
    for i in range(1, single_class_vectors + 1):
        outputVectors.append([0, 0, 1])

    # labels for no push
    for i in range(1, single_class_vectors+1):
        outputVectors.append([0, 1, 0])


def prepare_val_vectors():
# val class labels, divide by 3 , since there are 3 classes
    total_val_images = int (len(valImages) / 3)
    # val labels for left push
    for i in range(1, total_val_images+1):
        valVectors.append([1, 0, 0])

    # val labels for right push
    for i in range(1, total_val_images+ 1):
        valVectors.append([0, 0, 1])

    # val labels for no push
    for i in range(1, total_val_images+1):
        valVectors.append([0, 1, 0])


# Define the CNN Model
# tf.reset_default_graph()
convnet=input_data(shape=[None, 56, 100, 1], name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet, 1000, activation='relu')
convnet=dropout(convnet, 0.75)

convnet=fully_connected(convnet, 3, activation='softmax')

convnet=regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',name='regression')

model = tflearn.DNN(convnet, tensorboard_verbose=0)


# Data set preparation , This is a naive approach,
# could be efficiently implemented using tf.keras.utils.image_dataset_from_directory
prepare_images('left')
prepare_images('right')
prepare_images('stop')
prepare_output_vectors()
prepare_val_vectors()

loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=2)


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, patience, lowest_val_loss):
        """ Note: We are free to define our init function however we please. """
        self.patience = patience
        self.lowest_val_loss = lowest_val_loss

    def on_epoch_end(self, training_state):
        if training_state.val_loss < self.lowest_val_loss:
            self.lowest_val_loss = training_state.val_loss
            self.patience = 8
        else:
            self.patience = self.patience - 1
            if self.patience < 0:
                raise StopIteration


# Initialize our callback.
early_stopping_cb = EarlyStoppingCallback(patience=8, lowest_val_loss = 100)

# Train model
try:
    model.fit(loadedImages, outputVectors, n_epoch=50,
           validation_set = (valImages, valVectors),
           snapshot_step=100, show_metric=True, run_id='convnet_handgesture', batch_size=64, callbacks=early_stopping_cb)

except StopIteration:
    model.save("TrainedModel/GestureRecogModel.tfl")
    print("Caught callback exception. Early stopping.")

