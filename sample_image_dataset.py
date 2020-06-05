import numpy as np
from matplotlib import pyplot as plt
import sys
import random
import PIL
from PIL import Image
from time import sleep

def process(npyfile):
    """ Build a 16x16 tiny squares in an image."""
    data = np.load(npyfile)
    data = data.astype(np.uint8)
    assert data.shape[1] == 784
    img=Image.new('L',(16*28,16*28))
    for i in range(16):
        for j in range(16):
            img28=Image.new('L',(28,28))
            while True:
                row=random.choice(range(data.shape[0]))
                img_row=data[row,:].reshape(28,28)
                hasTopFilled=any(img_row[0,:])
                hasBotFilled=any(img_row[27,:])
                hasLeftFilled=any(img_row[:,0])
                hasRightFilled=any(img_row[:,27])
                if sum([hasBotFilled, hasTopFilled, hasLeftFilled, hasRightFilled]) < 1:
                    break
            for l in range(28):
                for m in range(28):
                    img28.putpixel((m,l),(img_row[l,m],))
            img.paste(img28,(i*28,j*28))
    img.show()
    sleep(5)
if __name__ == "__main__":
    process(sys.argv[1])
