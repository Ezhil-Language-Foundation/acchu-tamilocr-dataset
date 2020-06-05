# June 2nd 2020.
# custom script to convert 128x128 pix PNG images to a
# 28x28 pix mono image in Numpy format.
import sys
import glob
import PIL
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

def make_28x28_image(png128px_filename):
    imgO = Image.open(png128px_filename)
    img = imgO.resize((28,28),PIL.Image.BILINEAR)
    array = np.zeros((28,28))
    centroid = [0.0,0.0]
    pix_count = 0.0
    for i in range(28):
        for j in range(28):
            val = np.mean(img.getpixel((j,i))[0:3])#get RGB only of RGBA
            if val < 128.0: val = 0
            else: val = 255
            if ( val == 0.0 ):
                pix_count += 1.0
                centroid[0] +=  j
                centroid[1] +=  i
    centroid[0] = 14-1.0*centroid[0]/pix_count
    centroid[1] = 14-1.0*centroid[1]/pix_count
    print("centroid of image =>",centroid)
    #params_affine=(1.0,0.0,centroid[1],0.0,1.0,centroid[0])
    #img = img.transform((28,28),PIL.Image.AFFINE,params_affine,PIL.Image.)
    #img = img.rotate(0.0,translate=centroid)
    for i in range(28):
        for j in range(28):
            val = np.mean(img.getpixel((j,i))[0:3])#get RGB only of RGBA
            #if val < 128.0: val = 0.0
            #else: val = 255.0
            array[i,j]=255.0 - val
    plt.imshow(array)#astype(np.uint8)
    plt.show()
    np.save(png128px_filename.replace('.png','.npy'),array)

if __name__ == '__main__':
    for name in glob.glob('letters-hand-drawn-corrected/*.png'):
        print("Processing ... ",name)
        make_28x28_image(name)
        sleep(1)
