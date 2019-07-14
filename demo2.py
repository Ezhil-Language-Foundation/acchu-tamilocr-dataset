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

W=28
H=W
f= [24,24,18] #earlier 28,28,20
def img2array(img):
    igray = img.convert('L')
    #pprint(igray.tobytes())
    bytes = [ (float(val) > 0.0)*255.0 for val in igray.tobytes() ]
    return np.array(bytes).reshape(W,H)

#Avarangal1.
font32= ImageFont.truetype("/Library/fonts/InaiMathi-MN.ttc", f[1], encoding="uni")
font20=ImageFont.truetype("/Library/fonts/InaiMathi-MN.ttc", f[2], encoding="uni")
font = font32
uyir_plus_ayutham = copy.copy(tamil.utf8.uyir_letters)
uyir_plus_ayutham.append( tamil.utf8.ayudha_letter )
for idx,u in enumerate(uyir_plus_ayutham):
    image = Image.new('RGBA',(W,H),(0,0,0,255))#,(0,0,0,0))#grayscale
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0,0),(W,H)],fill=(0,0,0,255))
    if u == tamil.utf8.uyir_letters[-1]:
        #au.is over-rendered
        font = font20
    else:
        font = font32
    tw,th=(draw.textsize(u,font=font))
    #print(tw,th)
    tw,th = min(tw,W), min(th,H)
    draw.text(((W-tw)/2,0),u, font=font,fill=(255,255,255,255))
    theta=(random.choice([-90,-45,45,90]))
    theta=random.choice(range(-15,15))
    image=image.rotate(theta)
    image=image.crop([0,0,W,H])
    image=image.resize((W,H),Image.BILINEAR)
    #pprint(theta)
    image.show()
    img_nz = img2array(image)
    np.set_printoptions(threshold=sys.maxsize)
    #pprint(img_nz)
    np.save(os.path.join( os.getcwd(),'data',str(idx)+u'-letter'),img_nz)
    time.sleep(4)
