#coding: utf-8

# In[11]:

import cv2
import os
import os.path as osp
import random
import argparse

random.seed(42)

# In[6]:

import numpy as np
import matplotlib.pyplot as plt
import time


# In[3]:

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# In[20]:

def rotateImageWithProb(image, angle, prob, x1,y1,x2,y2):
    ang = random.randint(-angle, angle)
    if ang < 0:
        ang = 360 + ang

    nx1 = np.abs(x1 * np.cos(ang) - y1 * np.sin(ang))
    ny1 = np.abs(x1 * np.sin(ang) + y1 * np.cos(ang))

    nx2 = np.abs(x2 * np.cos(ang) - y2 * np.sin(ang))
    ny2 = np.abs(x2 * np.sin(ang) + y2 * np.cos(ang))

    if random.random() > prob:
        return rotateImage(image, ang), np.array([nx1,ny1,nx2,ny2])
    else:
        return image,  np.array([nx1,ny1,nx2,ny2])


# In[50]:

def changeBrightness(image, value, prob):
    if random.random() > prob:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #convert it to hsv

        inc = random.randint(0,value)

        h, s, v = cv2.split(hsv)
        lim = 255 - inc
        v[v > lim] = 255
        v[v <= lim] += inc
        final_hsv = cv2.merge((h, s, v))

        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img
    else:
        return image




#img = cv2.cvtColor(cv2.imread('images.jpeg'), cv2.COLOR_BGR2RGB)
#np.save('test.npy', img)


parser = argparse.ArgumentParser(description='Simple augmentator 0.0.1')
parser.add_argument('source', help='Folder with source images to augment', type=str, default='source', action='store')
parser.add_argument('n', help='Number to augmentate', type=int, default=10, action='store')
parser.add_argument('a', help='Angle for rotation', type=int, default=10, action='store')
parser.add_argument('v', help='Value for brightness increase', type=int, default=20,  action='store')
parser.add_argument('pa', help='Image will be rotated with probability > pa',type=float,  default=0.3,  action='store')
parser.add_argument('pv', help='Image will be saturated with probability > pv', type=float, default=0.3, action='store')
parser.add_argument('vis', help='Augmented image will be saved as separated jpeg', type=bool, default=False, action='store')
parser.add_argument('per_second', help='How many ops per second will be performed', type=int, default=3, action='store')

args = parser.parse_args()

print(args)

cnt = 0
t_start = time.time()
for i,im_p in enumerate(os.listdir(args.source)):
    img = cv2.cvtColor(cv2.imread(osp.join(args.source, im_p)), cv2.COLOR_BGR2RGB)
    coords = np.load('coords.npy')
    print(coords)
    img = img[coords[i*4]:coords[i*4+2], coords[i*4+1]:coords[i*4+3]]

    for j in range(int(args.n)):
        print('handling ',j)

        aug,newp = rotateImageWithProb(changeBrightness(img, args.v, args.pv), args.a, args.pa, coords[i*4],coords[i*4+1], coords[i*4+2],coords[i*4+3])
        cnt+= 1
#        print(newp)
        if cnt >= int(args.per_second):
            time.sleep(1-time.time() + t_start)
            cnt = 0
            t_start = time.time()


#        plt.imshow(aug)
#        plt.show()
#        np.save(osp.join(os.getcwd(),'aug_c', 'c_' + str(i) + '_' + str(j) + '.npy'), newp)
        np.save('c' + str(i) + str(j) + '.npy', newp)
        if args.vis:
            cv2.imwrite(osp.join('aug','out' + str(i) + '_' + str(j) + '.jpeg'), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
