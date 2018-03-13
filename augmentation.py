
# coding: utf-8

# In[11]:

import cv2
import os
import os.path as osp
import random
import argparse


# In[6]:

import numpy as np


# In[3]:

import matplotlib.pyplot as plt

img = np.load('test.npy')
coords = np.load('coords.npy')
img = img[coords[0]:coords[2], coords[1]:coords[3]]

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# In[20]:

def rotateImageWithProb(image, angle, prob):
    if random.random() > prob:
        ang = random.randint(-angle,angle)
        return rotateImage(image, ang)
    else:
        return image


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

        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    else:
        return image




#img = cv2.cvtColor(cv2.imread('images.jpeg'), cv2.COLOR_BGR2RGB)
#np.save('test.npy', img)


parser = argparse.ArgumentParser(description='Simple augmentator 0.0.1')
parser.add_argument('n', help='Number to augmentate', type=int, default=10, action='store')
parser.add_argument('a', help='Angle for rotation', type=int, default=10, action='store')
parser.add_argument('v', help='Value for brightness increase', type=int, default=20,  action='store')
parser.add_argument('pa', help='Image will be rotated with probability > pa',type=float,  default=0.3,  action='store')
parser.add_argument('pv', help='Image will be saturated with probability > pv', type=float, default=0.3, action='store')
parser.add_argument('vis', help='Augmented image will be saved as separated jpeg', type=bool, default=False, action='store')

args = parser.parse_args()

print(args)

for i in range(int(args.n)):
    print('handling ',i)
    aug = rotateImageWithProb(changeBrightness(img, args.v, args.pv), args.a, args.pa)
    if args.vis:
        cv2.imwrite('out' + str(i) + '.jpeg', aug)
