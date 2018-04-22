
"""## Testing the Model"""
import numpy as np
import os, random, sys, ast 
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from splitdata import test_train_split

from config import testdir,traindir

def randimg(src):
    if os.path.isfile("flower_photos/LICENSE.txt"): os.remove("flower_photos/LICENSE.txt")
    #pick a random class 
    subdir=random.choice(os.listdir(src))
    subdir_path=os.path.join(src,subdir)
    #pick random image of that class
    img=random.choice(os.listdir(subdir_path))
    return os.path.join(src,subdir,img)

def predict(model,labels):
    path = randimg(testdir)
    print(path)
    
    img = image.load_img(path,target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    print (labels[classes[0]])
    
def test_model(name,labelpath):
    model = load_model(name)
    lables = loadLabels(labelpath)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    for _ in range(10): predict(model,lables)

def loadLabels(filepath):
  with open(filepath, "r") as text_file:
    return ast.literal_eval(text_file.read())
    
if __name__ == "__main__":
    #arg 1 is saved model file 
    #arg 2 is dictionary of class labels
    name = sys.argv[1]
    labelpath = sys.argv[2]
    #setup the data if we haven't already
    test_train_split(traindir,testdir)

    test_model(name,labelpath)
