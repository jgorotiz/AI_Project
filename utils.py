import os
import glob
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model

# this is for normalization
def prep_pixels(x):
    # convertir integers to floats
    for i, elem in enumerate(x):
        elem = elem.astype('float32')
        elem = elem / 255.0
        x[i] = elem

    
    return x

def classification(name):
    if (name == "no"):
        value = 0
    elif (name == "si"):
        value = 1
    return value

def classification_inverse(name):
    if (name == 0):
        value = "no"
    elif (name == 1):
        value = "si"
    return value

def load_data(folder_name):
    currentPath = os.getcwd()
    arrayOfPaths = glob.glob(f"{currentPath}{os.sep}data{os.sep}{folder_name}{os.sep}*.jpeg")
    arrayOfImages = [img_to_array(load_img(x)) for x in arrayOfPaths]
    arrayOfIndex = [classification(x.split(os.sep)[-1].split("_")[0]) for x in arrayOfPaths]
    arrayOfTags = []
    for value in arrayOfIndex:
        zeroArray = np.zeros(2, dtype=int)
        zeroArray[value] = 1
        arrayOfTags.append(zeroArray)
    return np.array(arrayOfImages), np.array(arrayOfTags)
