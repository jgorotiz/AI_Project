from PIL import Image
import os
import glob
import sys
import matplotlib.pyplot as plt
#from keras.preprocessing.image import img_to_array

def resize(folder_name):
    currentPath = os.getcwd()
    arrayOfPaths = glob.glob(f"{currentPath}{os.sep}data{os.sep}{folder_name}{os.sep}*.jpeg")
    
    print(arrayOfPaths)
    for path in arrayOfPaths:
        name = path.split(os.sep)[-1].split(".")[0] + "_128_72.jpeg"
        im = Image.open(path)
        #a = img_to_array(im)
        im = im.resize((128,72), Image.ANTIALIAS)
        #a = img_to_array(im)
        im.save(f"{currentPath}{os.sep}data{os.sep}train_image_resized{os.sep}{name}")
        print(path)
   
if __name__ == "__main__":
    currentPath = sys.argv[1]
    resize(currentPath)
