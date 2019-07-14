#!/usr/bin/python3
from PIL import ImageFont, ImageDraw, Image
import tamil
import numpy as np
import time
import copy
from pprint import pprint
#import unicodedata
import sys

W=32
H=W
image = Image.new('RGBA',(W,H),(0,0,0,255))#,(0,0,0,0))#grayscale
draw = ImageDraw.Draw(image)

def img2array(img):
    igray = img.convert('L')
    pprint(igray.tobytes())
    bytes = [ (float(val) > 0.0)*255.0 for val in igray.tobytes() ]
    return np.array(bytes).reshape(W,H)

#Avarangal1.
font32= ImageFont.truetype("/Library/fonts/InaiMathi-MN.ttc", 28, encoding="uni")
font = font32
uyir_plus_ayutham = copy.copy(tamil.utf8.uyir_letters)
uyir_plus_ayutham.append( tamil.utf8.ayudha_letter )
for u in uyir_plus_ayutham:
    draw.rectangle([(0,0),(W,H)],fill=(0,0,0,255))
    if u == tamil.utf8.uyir_letters[-1]:
        #au.is over-rendered
        font = ImageFont.truetype("/Library/fonts/InaiMathi-MN.ttc", 20, encoding="uni")
    else:
        font = font32
    tw,th=(draw.textsize(u,font=font))
    print(tw,th)
    tw,th = min(tw,W), min(th,H)
    draw.text(((W-tw)/2,0),u, font=font,fill=(255,255,255,255))
    image.show()
    img_nz = img2array(image)
    np.set_printoptions(threshold=sys.maxsize)
    pprint(img_nz)
    np.save(u+u'img.mat',img_nz)
    time.sleep(4)
