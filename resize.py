from PIL import Image
import os
import glob
import sys

def resize(folder_name):
    currentPath = os.getcwd()
    arrayOfPaths = glob.glob(f"{currentPath}{os.sep}data{os.sep}{folder_name}{os.sep}*.jpeg")
    
    print(arrayOfPaths)
    for path in arrayOfPaths:
        name = path.split(os.sep)[-1].split(".")[0] + "_128_72.jpeg"
        print(name)
        im = Image.open(path)
        print(name)
        im = im.resize((128,72), Image.ANTIALIAS)
        print(name)
        im.save(f"{currentPath}{os.sep}data{os.sep}validation_image_resized{os.sep}{name}")
        print(path)
   
if __name__ == "__main__":
    currentPath = sys.argv[1]
    resize(currentPath)
