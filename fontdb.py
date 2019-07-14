import glob
import os
from pprint import pprint
FONTPATH=os.path.join( os.path.split(__file__)[0], 'fonts' )

def get_font_names():
    print(FONTPATH)
    return glob.glob(FONTPATH+'/*')

if __name__ == "__main__":
    pprint(get_font_names())
