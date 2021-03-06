import random

random.seed(42)

# In[6]:

import numpy as np

import skimage.transform as T
import skimage.exposure as E


class BatchTransform(object):
    def __init__(self, prob, gamma=-1, angle=-1):
        self._hflip = DoRandom(prob, HorizontalFlip())
        self._vflip = DoRandom(prob, VerticalFlip())

        self._prob = prob
        self._gamma = gamma
        self._angle = angle
        if gamma != -1:
            self._gamma_tr = RandomGamma(gamma, prob)
        else:
            self._gamma_tr = None
        if angle != -1:
            self._angle_tr = Dim2RandomRotation(angle, prob)
        else:
            self._angle_tr = None

    def __call__(self, image_batch, mask_batch, target_shape=None):
        blen = image_batch.shape[0]
        if target_shape is not None:
            img_cpy = np.zeros((blen, target_shape[0], target_shape[1], image_batch.shape[3]))
            msk_cpy = np.zeros((blen, target_shape[0], target_shape[1], mask_batch.shape[3]))

            print('source shape:', image_batch.shape)

            for cnt in range(blen):
                img_cpy[cnt, :] = T.resize(image_batch[cnt, :], target_shape)
                msk_cpy[cnt, :] = T.resize(mask_batch[cnt, :], target_shape)

            print('result shape:', img_cpy.shape)
        else:
            img_cpy = image_batch;
            msk_cpy = mask_batch

        for cnt in range(blen):
            img_cpy[cnt, :], msk_cpy[cnt, :] = self._hflip(img_cpy[cnt, :], msk_cpy[cnt, :])
            img_cpy[cnt, :], msk_cpy[cnt, :] = self._vflip(img_cpy[cnt, :], msk_cpy[cnt, :])

            if self._angle_tr is not None:
                img_cpy[cnt, :], msk_cpy[cnt, :] = self._angle_tr(img_cpy[cnt, :], msk_cpy[cnt, :])
            if self._gamma_tr is not None:
                img_cpy[cnt, :], msk_cpy[cnt, :] = self._gamma_tr(img_cpy[cnt, :], msk_cpy[cnt, :])

        return img_cpy, msk_cpy


class VerticalFlip(object):
    def __call__(self, image, mask):
        return image[::-1, :], mask[::-1, :]


class HorizontalFlip(object):
    def __call__(self, image, mask):
        return image[:, ::-1], mask[:, ::-1]


class DoRandom(object):
    def __init__(self, prob, transform):
        self._prob = prob
        self._transform = transform

    def __call__(self, image, mask):
        if random.random() > self._prob:
            return self._transform(image, mask)
        else:
            return image, mask


class Dim2RandomRotation(object):
    def __init__(self, angle, prob):
        self._angle = angle
        self._prob = prob

    def __call__(self, image, mask):
        assert mask.shape[2] == 2
        rangle = random.uniform(-self._angle, self._angle)
        if random.random() > self._prob:
            rimg = T.rotate(image, rangle)

            rmask = T.rotate(mask[:, :, 0], rangle)
            rbg = T.rotate(mask[:, :, 1], rangle)

            # xor_bg = np.logical_xor(rmask, rbg)
            # rbg[xor_bg == 0] = 1
            # rbg[rmask == 0] = 1
            comp = np.dstack((rmask, rbg))

            return rimg, comp
        else:
            return image, mask


class RandomRotation(object):
    def __init__(self, angle, prob):
        self._angle = angle
        self._prob = prob

    def __call__(self, image, mask):
        rangle = random.uniform(-self._angle, self._angle)
        if random.random() > self._prob:
            rimg = T.rotate(image, rangle)
            rmask = T.rotate(mask, rangle)
            return rimg, rmask
        else:
            return image, mask


class RandomGamma(object):
    def __init__(self, gamma, prob):
        assert gamma > 0
        self._gamma = gamma
        self._prob = prob

    def __call__(self, image, mask):
        rgamma = random.uniform(0, self._gamma)
        if random.random() > self._prob:
            return E.adjust_gamma(image, rgamma), mask
        else:
            return image, mask
