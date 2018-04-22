"""### Split data into test and train sets"""

import os
import shutil
import sys
import tarfile
import wget

from config import archive
#given a data set of sub dirs of images copy 20% of files to a
# new dir preserving dir structure

numfiles=0
numclasses=0

def split(src,dest_dir):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if not (os.path.isfile(full_file_name)):   #if it is a dir 
            global numclasses
            numclasses+=1
            dest_name = os.path.join(dest_dir, file_name)
            os.makedirs(dest_name)
            copy(full_file_name,dest_name)
            

#copy 20% of files from src to dest
def copy(src,dest):
    src_files = os.listdir(src)
    num=len(src_files)//5
    for file_name in src_files:
        num-=1
        if not num:
            break
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.move(full_file_name, dest)
            global numfiles
            numfiles+=1



def test_train_split(src,dest_dir):
    #this function should only be run once or else to many files will be copied to the test set
    if(os.path.exists(dest_dir)):
      return
    dl_data()
    extract()
    os.makedirs(dest_dir)
    split(src,dest_dir)
    print('moved files {0} from {1} to {2} '.format(numfiles,src,dest_dir),flush=True)

def count_classes(src):
      return len(list(os.walk(src))) - 1 

def extract():
    tar = tarfile.open(archive)
    tar.extractall()
    tar.close()

def dl_data():
    url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
    print('downloading dataset please wait')
    wget.download(url)