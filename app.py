#!/usr/bin/python3
from PIL import ImageFont, ImageDraw, Image
import tamil
import numpy as np
import time
import copy
from pprint import pprint
#import unicodedata
import sys, os
import random
from fontdb import get_font_names, FONTDB
MNIST_R = 60000
W=28
H=W
fsize= [24,24,18] #fontsize
def img2array(img):
    igray = img.convert('L')
    #pprint(igray.tobytes())
    bytes = [ (float(val) > 0.0)*255.0 for val in igray.tobytes() ]
    return np.array(bytes).reshape(W*H)

# 0) setup font db and regular/smalls across available fonts.
#we skip TAM, TAB fonts.

# 1) Setup letters to be built
uyir_plus_ayutham = copy.copy(tamil.utf8.uyir_letters)
uyir_plus_ayutham.append( tamil.utf8.ayudha_letter )

# 1.1) Initialize MNIST variables
data_image = np.zeros((MNIST_R,W*H))
data_label = np.zeros((MNIST_R,1))

n_rows = 0
def print_completion():
    print("Completed %g%%"%(n_rows/float(MNIST_R)*100.0))

# 2) Build set given a font specification and return an array of 13-row images and Labels
def build_letter_set(fontobj,rotate=False,translate=False):
    data_img = np.zeros((13,784))
    data_lbl = np.zeros((13,1))
    shuffle_idx = list(range(0,len(uyir_plus_ayutham)))
    random.shuffle(shuffle_idx)
    for pos,idx in enumerate(shuffle_idx):
        u = uyir_plus_ayutham[idx]
        image = Image.new('RGBA',(W,H),(0,0,0,255))#,(0,0,0,0))#grayscale
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0,0),(W,H)],fill=(0,0,0,255))
        if u == tamil.utf8.uyir_letters[-1]:
            #au.is over-rendered
            font = fontobj.M
        else:
            font = fontobj.L
        tw,th=(draw.textsize(u,font=font))
        tw,th = min(tw,W), min(th,H)
        draw.text(((W-tw)/2,0),u, font=font,fill=(255,255,255,255))
        if translate:
            # +/-5 on X,Y centered
            tvec =np.floor(np.random.random((2))*10-5)
        else:
            tvec = np.zeros((2))
        tvec = (tvec[0],tvec[1])
        if rotate:
            theta=random.choice(range(-15,15))
            image = image.rotate(theta,translate=tvec)
        if rotate or translate:
            image=image.crop([0,0,W,H])
        image=image.resize((W,H),Image.BILINEAR)
        data_img[pos,:] = img2array(image)
        data_lbl[pos] = idx
    return data_img,data_lbl

def main():
    n_rows = 0
    FONTNAME = list(FONTDB.keys())
    while n_rows < MNIST_R:
        # pick a font.
        fontobj = FONTDB[ FONTNAME[random.choice(range(0,len(FONTDB)))] ]
        rotate =  n_rows > 30000
        translate = n_rows > 50000
        data_img,data_lbl = build_letter_set(fontobj,rotate,translate)
        pos = 0
        while (n_rows < MNIST_R) and (pos < len(data_lbl)):
            data_image[n_rows,:]=data_img[pos,:]
            data_label[n_rows] = data_lbl[pos]
            n_rows += 1
            pos += 1
        print("Added %d rows (total %d / %d)"%(pos,n_rows,MNIST_R))
        #print_completion()
    data_label_onehot = np.zeros((max(data_label.shape),13))
    for idx,pos in enumerate(data_label): data_label_onehot[idx][int(pos)]=1.0;
    np.save(os.path.join( os.getcwd(),'data','train-image-'+str(time.time())),data_image)
    np.save(os.path.join( os.getcwd(),'data','train-label-'+str(time.time())+'-onehot'),data_label_onehot)

def draw_composite():
    #run after main.
    im = Image.new('RGBA',(28*13,28*16),(0,0,0,255))
    for rows in range(13):
        for col in range(16):
            while True:
                lbl = random.choice(range(MNIST_R))
                if data_label[lbl] == rows:
                    break
            letter = data_image[lbl].reshape(W,H)
            sub_im = Image.fromarray(letter)
            im.paste(sub_im,(rows*W,col*H))
    im.show()

if __name__ == "__main__":
    pprint(FONTDB)
    main()
