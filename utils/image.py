#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image, ImageChops, ImageMath
import numpy as np
from utils import noisy
import cv2

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('L')
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res  = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    
    ow, oh = img.size
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    img = cropped.resize(shape)

    #img = random_distort_image(sized, hue, saturation, exposure)
    
    
    return img, flip, dx,dy,sx,sy 


def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy, num_keypoints, max_num_gt):
    num_labels = 2 * num_keypoints + 10
    label = np.zeros((max_num_gt,num_labels))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        #print(bs)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, num_labels))
        cc = 0
        for i in range(bs.shape[0]):
            xs = list()
            ys = list()
            for j in range(num_keypoints):
                xs.append(bs[i][2*j+1])
                ys.append(bs[i][2*j+2])

            # Make sure the centroid of the object/hand is within image
            xs[0] = min(0.999, max(0, xs[0] * sx - dx)) 
            ys[0] = min(0.999, max(0, ys[0] * sy - dy)) 
            for j in range(1,num_keypoints):
                xs[j] = xs[j] * sx - dx 
                ys[j] = ys[j] * sy - dy 

            for j in range(num_keypoints):
                bs[i][2*j+1] = xs[j]
                bs[i][2*j+2] = ys[j]
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def change_background(img, mask, bg, opacity = False):
    # oh = img.height  
    # ow = img.width
    ow, oh = img.size
    bg = bg.resize((ow, oh)).convert('L')

    #bg = noisy("poisson", bg)
    if opacity:
        rand_val =random.random()
        bg = Image.blend(img, bg, rand_val)
        #bg =  cv2.addWeighted(img, 1-rand_val, bg, rand_val, 0)

    imcs = list(img.split())
    bgcs = list(bg.split())
    maskcs = list(mask.split())
    fics = list(Image.new(img.mode, img.size).split())
    
    for c in range(len(imcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i / 255)
        fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
    out = Image.merge(img.mode, tuple(fics))

    return out

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure, bgpath, num_keypoints, max_num_gt):
    labpath = imgpath.replace('images', 'labels').replace('Images', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    maskpath = imgpath.replace('Images', 'mask').replace('/00', '/').replace('.jpg', '.png')

    ## data augmentation
    img = Image.open(imgpath).convert('L')
    mask = Image.open(maskpath).convert('L')
    bg = Image.open(bgpath).convert('L')

    val = random.randint(0, 9)
    
    if val > 1:
        img = change_background(img, mask, bg, opacity=True)

    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    
    ow, oh = img.size
    label = fill_truth_detection(labpath, ow, oh, flip, dx, dy, 1./sx, 1./sy, num_keypoints, max_num_gt)
    return img,label


