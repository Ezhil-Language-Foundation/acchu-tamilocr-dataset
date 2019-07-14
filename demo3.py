#!/usr/bin/python3
from PIL import ImageFont, ImageDraw, Image
import time
from fontdb import FONTDB
for font_name,fobj  in FONTDB.items():
    print(font_name)
    image = Image.new('RGBA',(4*64,4*64),(0,0,0))
    draw = ImageDraw.Draw(image)
    draw.text((40, 100), u"அ ஆ இ ஈ ஐ ஒ ஓ ஔ", font=fobj.M)
    draw.text((40,200),u""+font_name,font=fobj.M)
    image.show()
    time.sleep(2)
