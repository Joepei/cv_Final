"""
Copyright (C) 2018 NVIDIA Corporation.    All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# from __future__ import division
from PIL import Image
from torch import nn
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter


class GIFSmoothing(nn.Module):
    def forward(self, *input):
        pass
        
    def __init__(self, r, eps):
        super(GIFSmoothing, self).__init__()
        self.r = r
        self.eps = eps

    def process(self, initImg, contentImg):
        return self.process_opencv(initImg, contentImg)

    def process_opencv(self, initImg, contentImg):
        '''
        :param initImg: intermediate output. Either image path or PIL Image
        :param contentImg: content image output. Either path or PIL Image
        :return: stylized output image. PIL Image
        '''
        if type(initImg) == str:
            init_img = cv2.imread(initImg)
            #print(init_img.shape)
            # init_img = init_img[2:-2,2:-2,:]
        else:
            init_img = np.array(initImg)[:, :, ::-1].copy()
        
        if type(contentImg) == str:
            cont_img = cv2.imread(contentImg)
            #cont_img = cont_img[1:-1, :, :]
            #print(cont_img.shape)
        else:
            cont_img = np.array(contentImg)[:, :, ::-1].copy()

        lc, hc, _ = cont_img.shape
        #print(lc, hc)
        li, hi, _ = init_img.shape
        #print(li, hi)
        if lc > li: 
            cont_img = cont_img[int(np.floor((lc-li)/2)):int(-np.ceil((lc-li)/2)),:,:]
        if lc < li: 
            init_img = init_img[int(np.floor((li-lc)/2)):int(-np.ceil((li-lc)/2)),:,:]
        if hc > hi:
            cont_img = cont_img[:,int(np.floor((hc-hi)/2)):int(-np.ceil((hc-hi)/2)),:]
        if hc < hi: 
            #print(int(np.floor((hi-hc)/2)))
            #print(int(-np.ceil((hi-hc)/2)))
            init_img = init_img[:,int(np.floor((hi-hc)/2)):int(-np.ceil((hi-hc)/2)),:]
        print(cont_img.shape, init_img.shape)
        output_img = guidedFilter(guide=cont_img, src=init_img, radius=self.r, eps=self.eps)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        output_img = Image.fromarray(output_img)
        return output_img