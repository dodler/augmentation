import os
from skimage_augmentation import *

from skimage.io import imsave
from skimage.io import imread
from skimage.transform import *

from skimage.color import *

import os
import os.path as osp
import numpy as np

source = 'source'
target = 'target'

bsize = 8

files  = os.listdir(osp.join(source,'img/'))
print(files)

imgs = []
masks = []

dsize=1024
target_shape = (dsize,dsize)

imgs.copy()


for i in range(0,8):
    imgs.append(resize(imread(osp.join(source, 'img',files[i])), target_shape))
    m = resize(imread(osp.join(source, 'mask',files[i])), target_shape)
    m = rgb2gray(m).reshape((dsize,dsize,1))
    m[m > 0] = 1
    bg = m.copy()
    bg[bg==0] = 2
    bg[bg==1]=0
    bg[bg==2]=1
    masks.append(np.dstack((m,bg)))

imgs = np.array(imgs)
masks = np.array(masks)

print(imgs.shape)
print(masks.shape)

bt = BatchTransform(0.0, 3, 30)
imgs,masks = bt(imgs, masks, (512,512))

for i in range(0,8):
    imsave(osp.join(target, 'img','out' + str(i) + '.jpg'), imgs[i])
    imsave(osp.join(target, 'mask','out_m' + str(i) + '.jpg'), masks[i,:,:,0])
    imsave(osp.join(target, 'mask','out_bg' + str(i) + '.jpg'), masks[i,:,:,1])
