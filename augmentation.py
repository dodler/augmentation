
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

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# In[20]:

def rotateImageWithProb(image, angle, prob):
    if random.random() > prob:
        return rotateImage(image, angle)
    else:
        return image


# In[50]:

def changeBrightness(image, value, prob):
    if random.random() > prob:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert it to hsv

        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv2.merge((h, s, v))

        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    else:
        return image




#img = cv2.cvtColor(cv2.imread('images.jpeg'), cv2.COLOR_BGR2RGB)
#np.save('test.npy', img)


parser = argparse.ArgumentParser(description='Awesome augmentator 0.0.1')
parser.add_argument('n',type=int, help='Number to augmentate', default=10)
parser.add_argument('a',type=int, help='Angle for rotation', default=10)
parser.add_argument('v',type=int, help='Value for brightness increase', default=20)
parser.add_argument('pa', type=int, help='Image will be rotated with probability > pa', default=0.3)
parser.add_argument('pv',type=int, help='Image will be saturated with probability > pv', default=0.3)
parser.add_argument('sa', type=bool, help='sa - Stohastic angle, with a as maximum angle', default=False)
parser.add_argument('sv', type=bool, help='sv - Stohastic value, with v as maximum value', default=False)

args = parser.parse_args()

for i in range(args.n):
    aug = rotateImageWithProb(changeBrightness(img, args.v, args.pv), args.a, args.pa)
    np.save('out_' + str(i) + '.npy', aug)
