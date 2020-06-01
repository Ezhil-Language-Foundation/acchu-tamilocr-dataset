import glob
import os
from pprint import pprint
from collections import namedtuple
from PIL import ImageFont
FontType = namedtuple('FontType',('name','path','L','M'))

FONTPATH=os.path.join( os.path.split(__file__)[0], 'fonts' )

def get_font_names():
    print(FONTPATH)
    return glob.glob(FONTPATH+'/*')

# 0) setup font db and regular/smalls across available fonts.
FONTDB={}
fsize= [24,92,18] #fontsize
def build_fontdb():
    for fontpath in get_font_names():
        fontname = fontpath.split('/')[-1]
        name = fontname.split('.')[0]
        fontL= ImageFont.truetype(fontpath, fsize[1], encoding="uni")
        fontM=ImageFont.truetype(fontpath, fsize[2], encoding="uni")
        FONTDB[name] = FontType(name,fontname,fontL,fontM)
build_fontdb()

def get_font_like(name_hint):
        for name,font in FONTDB.items():
            if name.find(name_hint) >= 0:
                return font
        raise Exception("Cannot find font {0}".format(name_hint))

if __name__ == "__main__":
    pprint(get_font_names())
