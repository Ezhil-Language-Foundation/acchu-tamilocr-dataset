#!/usr/bin/python3
from PIL import ImageFont, ImageDraw, Image
import time
from fontdb import FONTDB
from paper import USLetter
paper = USLetter()
for font_name,fobj  in FONTDB.items():
    print(font_name)
    size = list(map(int,paper.canvas()))
    size[0]//=2
    size[1]//=2
    print(size)
    size=(64*4,64*4)
    image = Image.new('RGBA',size,(255,255,255))
    draw = ImageDraw.Draw(image)
    draw.text((40, 100), u"அ ஆ இ ஈ ஐ ஒ ஓ ஔ", font=fobj.M,fill=(0,0,0,255))
    draw.text((40,200),u""+font_name,font=fobj.M,fill=(0,0,0,255))
    image.show()
    time.sleep(2)
