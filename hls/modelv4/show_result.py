import numpy as np
from PIL import Image
import os
import argparse
from matplotlib import pylab as plt
#from resize import resize_images
from chainer.functions import resize_images

parser = argparse.ArgumentParser()
parser.add_argument('image')
args = parser.parse_args()

img = np.loadtxt(args.image,delimiter=",")
pred = img.reshape(int(np.sqrt(img.size)), int(np.sqrt(img.size)))
print(pred.shape)

pred = 1000./pred
plt.imshow(pred)
plt.show()

