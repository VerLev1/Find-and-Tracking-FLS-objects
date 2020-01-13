# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2

@author: vered
"""

import cv2
import numpy as np
# import matplotlib.pyplot as plt


class ROIfind:
    def __init__(self, img):
        self.img = img
        self.i = 1

    def window(self, I, width):  # integral image, window's width
        rows, cols = I.shape
        up_left = I[0:rows - width, 0:cols - width]
        up_right = I[0:rows - width, width:cols]
        down_left = I[width:rows, 0:cols - width]
        down_right = I[width:rows, width:cols]

        window_map = down_right - down_left - up_right + up_left
        return window_map

    def __create_rois_map(self, bg_out=50, bg_in=20, echo=30):

        rows, cols = self.img.shape
        I = np.ones((rows, cols), np.int32)
        I = cv2.integral(self.img, I, -1)
        # I2=I/np.amax(I)
        I = I[1:rows][1:cols]

        # create echo map:
        echo_map = self.window(I, echo)
        echo_sz = echo_map.shape

        echo_map = echo_map / (echo ** 2)
        echo_map = echo_map.astype(np.uint8)


        # create threshold
        rois_map_gray = echo_map
        '''
        h = cv2.calcHist([rois_map_gray], [0], None, [256], [0,256])
        th = np.argmax(h[50:256], 0)
        th = th.astype(np.float)/256
        th = th[0]
        '''
        th = 0.3*256

        rois_map = np.zeros((rows, cols))
        r_index = np.where(rois_map_gray > th)

        # there is a shift because the echo_map is smaller then the original image, d1 is the shift-fixing array:
        d1 = int(0.5 * (cols - echo_sz[1]))
        r_ind_sz = r_index[0].shape
        r_ind_sz = r_ind_sz[0]
        tu = np.ones((2, r_ind_sz))* d1
        r_index = np.asarray(r_index) + tu
        r_index = r_index.astype(np.int64)
        r_index=tuple((r_index[0], r_index[1]))

        rois_map[r_index] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rois_map = cv2.dilate(rois_map, kernel)

        return rois_map

    def __find_center(self):
        # this method find the pcenter of the image: where the FLS is
        rows, cols = self.img.shape
        lastLine = np.array(self.img[rows - 1, :])
        l = np.where(lastLine > 0)
        l=l[0] # l is a tuple of list. extract the list.
        # if the last line is empty:
        i = 2
        while not l.size: # while l is empty
            lastLine = np.array(self.img[rows - i, :])
            l = np.where(lastLine > 0)
            l = l[0]
            i = i+1

        sz = l.shape
        p1 = np.array([l[0], rows]).astype(np.float)
        p2 = np.array([l[sz[0]-1], rows]).astype(np.float)
        middle = int((p1[0] + p2[0]) / 2)
        col = np.where(self.img[:, middle] > 0)
        c = np.amax(col)
        p3 = np.array([middle, c]).astype(np.float)

        # method from: https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/

        x12 = p1[0] - p2[0]
        x13 = p1[0] - p3[0]
        y12 = p1[1] - p2[1]
        y13 = p1[1] - p3[1]
        y21 = p2[1] - p1[1]
        y31 = p3[1] - p1[1]
        x31 = p3[0] - p1[0]
        x21 = p2[0] - p1[0]

        sx13 = pow(p1[0], 2) - pow(p3[0], 2)
        sy13 = pow(p1[1], 2) - pow(p3[1], 2)
        sx21 = pow(p2[0], 2) - pow(p1[0], 2)
        sy21 = pow(p2[1], 2) - pow(p1[1], 2)

        f = ((sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) // (2 * (y31 * x12 - y21 * x13)))
        g = ((sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) // (2 * (x31 * y12 - x21 * y13)))
        c = (-pow(p1[0], 2) - pow(p1[1], 2) - 2 * g * p1[0] - 2 * f * p1[1])

        xc = -g
        yc = -f

        return [xc, yc]

    def apply(self):
        rois_map = ROIfind.__create_rois_map(self)
        mask = rois_map * self.img
        mask = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask1 = mask.astype(np.uint8)
        ret, mask = cv2.threshold(mask1, 0.7 * 255, 255, cv2.THRESH_BINARY)
        # xc, yc = self.__find_center()

        centroids = np.array([1, 2])
        bboxes = np.array([1, 2, 3, 4])
        tiny_masks= []
        index = 0
        c = {}

        # getting features of every ROI:
        mask = np.uint8(mask)

        # in opencv version up to 2.4
        im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #in opencv version from 2.4
        #contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        moments = np.array([])
        for cnt in range(len(contours)):

            x, y, w, h = cv2.boundingRect(contours[cnt])

            if w > 15 or h > 15:

                moments = cv2.moments(contours[cnt])
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                centroids = np.vstack((centroids, [cx, cy]))
                bboxes = np.vstack((bboxes, [x, y, w, h]))

                maskToPix = np.zeros(mask.shape, np.uint8)
                maskToPix = cv2.drawContours(maskToPix, [contours[cnt]], 0, 255, -1)
                pixelList = np.nonzero(maskToPix)
                m = np.zeros(mask1.shape)
                m[pixelList] = mask1[pixelList]
                m = m[y:y + h, x:x + w]
                tiny_masks.append(m)

                index = index + 1

            else: #delete the object

                mask[x:x + w + 1][y:y + h + 1] = 0

        centroids = np.delete(centroids, 0, 0)
        bboxes = np.delete(bboxes, 0, 0)
        return mask, centroids, bboxes, tiny_masks
