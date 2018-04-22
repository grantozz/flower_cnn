# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout, Flatten ,MaxPooling2D ,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from splitdata import test_train_split

from plot import plot_model_history

from config import testdir,traindir


#split data into test and train sets

numclasses=5
test_train_split(traindir,testdir)


name ="flower_model_v4.0.6"


# define the architecture of the model 


# Initialising the Network
model = Sequential()

# First the input image is convolved with 32 3x3 kernals to produce 32 output filters
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# then max pooling is applyed with the usual size of 2x2
model.add(MaxPooling2D(pool_size = (2, 2)))


# then a second identical set of convolutionan and max pooling layers are applyed
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# then droppout is applied before the first fully connected layer
# I found this helped prevent overfitting to some degree

model.add(Dropout(0.4))

# then the output filters are flattend before being fed into the fully connected layers
model.add(Flatten())

# Fully connected layers. the Last layer uses softmax activation to obtaian a probibility distribution from the logits?
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.7))
model.add(Dense(units = numclasses, activation = 'softmax'))

# the model is then compiled using adam 
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# I found that accuracy decreased after 4 epochs of training
epoch_num = 4



#TODO add  COMMENTS
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

print("training set ")
training_set = train_datagen.flow_from_directory(traindir,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
print("test set ")
test_set = test_datagen.flow_from_directory(testdir,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

#print and save label map for testing
#invert the map 
label_map = {v: k for k, v in training_set.class_indices.items()}
with open(name+"_labels.txt", "w") as text_file:
    text_file.write(str(label_map))


start = time.time()

print("training {}".format(name))
model_info =model.fit_generator(training_set,
                         steps_per_epoch =1000,
                         epochs = epoch_num,
                         validation_data = test_set,
                         validation_steps = 500)

total = time.time() - start
print ("Model took {0:.2f} min to train".format(total/60))

model.save(name+'.h5') 
print ("Model saved as {0}.h5".format(name))
plot_model_history(model_info,name)
