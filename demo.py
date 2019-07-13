#!/usr/bin/python3
from PIL import ImageFont, ImageDraw, Image
image = Image.new('RGBA',(4*64,4*64),(0,0,0))
draw = ImageDraw.Draw(image)

#Avarangal1.
font = ImageFont.truetype("/Library/fonts/InaiMathi-MN.ttc", 16, encoding="uni")

draw.text((10, 10), u"வணக்கம் ", font=font)

# use a truetype font
font = ImageFont.truetype("./latha.ttf", 15)

draw.text((20, 25), u"தமிழ் ", font=font)
image.show()
