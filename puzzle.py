#!/usr/bin/python3
from PIL import ImageFont, ImageDraw, Image
import time
from fontdb import FONTDB
from paper import USLetter
from tamil import utf8

paper = USLetter()

#font32= ImageFont.truetype("/Library/fonts/InaiMathi-MN.ttc", "32", encoding="uni")

i=0;
for font_name,fobj  in FONTDB.items():
    i+=1
    if i < 6: continue #Arial Unicode. Font size = 92,
    print(font_name)
    #font_name = font32
    size = list(map(int,paper.canvas()))
    size[0]//=2
    size[1]//=2
    print(size)
    #size=(64*4,64*4)
    image = Image.new('RGBA',size,(255,255,255))
    draw = ImageDraw.Draw(image)
    kwargs = {'font':fobj.L,'fill':(0,0,0,255)}
    W = size[0]
    H = size[1]
    dW = size[0]//3
    dH = size[1]//8
    offW = dW/2*0.65
    offH = dH/2*0.25
    for row in range(8):
        for col in range(3):
            idx = (row)*3 + col
            letter = utf8.uyir(idx%12)
            x,y = col*dW,row*dH
            draw.line((x,0,x,H),fill=(0,255,0,255))
            draw.line((0,y,W,y),fill=(0,0,255,255))
            draw.text((x+offW, y+offH), letter, **kwargs)

    image.show()
    time.sleep(2)
    break
