# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2

@author: vered
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import time


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

        # background estimation map
        bg_map = np.zeros((rows, cols))
        echo_map = np.zeros((rows, cols))

        rows1, cols1 = I.shape

        # this calculation is instead of the for loops in comment below:
        out_map = self.window(I, bg_out)
        in_map = self.window(I, bg_in)

        # make the in_map the same size as out_map
        out_sz = out_map.shape
        in_sz = in_map.shape
        d = int(0.5 * (in_sz[0] - out_sz[0]))
        in_map = in_map[d:in_sz[0] - d, d:in_sz[1] - d]

        # create background map:
        bg_map = out_map - in_map

        # create echo map:
        echo_map = self.window(I, echo)
        echo_sz = echo_map.shape
        bg_sz = bg_map.shape
        d = int(0.5 * (echo_sz[0] - bg_sz[0]))
        echo_map = echo_map[d:echo_sz[0] - d, d:echo_sz[1] - d]

        '''
        for x in range(bg_out-1, cols1-bg_out-1):
            for y in range(bg_out-1, rows1-bg_out-1):

                out_sum = I[y-bg_out,x-bg_out] -I[y+bg_out,x-bg_out] +I[y+bg_out,x+bg_out] -I[y-bg_out,x+bg_out]
                in_sum = I[y-bg_in,x-bg_in] -I[y+bg_in,x-bg_in] +I[y+bg_in,x+bg_in] -I[y-bg_in,x+bg_in]
                bg_map[y,x] = out_sum-in_sum
                echo_map[y,x] = I[y-echo_sz,x-echo_sz] -I[y+echo_sz,x-echo_sz] +I[y+echo_sz,x+echo_sz] -I[y-echo_sz,x+echo_sz]
        '''

        # normalizing:
        bg_map = bg_map / ((bg_out ** 2) - (bg_in ** 2))
        echo_map = echo_map / (echo ** 2)

        rois_map = np.zeros((rows, cols))
        rois_map[np.where((echo_map - bg_map) > 0.1)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rois_map = cv2.dilate(rois_map, kernel)

        return rois_map

    def __find_center(self):
        # this method find the pcenter of the image: where the FLS is
        rows, cols = self.img.shape
        lastLine = np.array(self.img[rows - 1, :])
        l = np.where(lastLine > 0)
        l = l[0]
        # l=np.array(l)
        sz = l.shape
        p1 = np.array([cols, l[0]])
        p2 = np.array([cols, sz[0]])
        middle = (l[0] + sz[0]) / 2
        col = np.where(self.img[:, int(middle)] > 0)
        c = np.amax(col)
        p3 = [c, middle]
        ma = (p2[0] - p1[0]) / (p2[1] - p1[1])
        mb = (p3[0] - p2[0]) / (p3[1] - p2[1])
        xc = ((ma * mb * (p1[0] - p3[0])) + (mb * (p1[1] + p1[1])) - (ma * (p2[1] + p3[1]))) / (2 * (mb - ma))
        y1 = (xc - p1[1]) ** 2
        y2 = (xc - p3[1]) ** 2
        y3 = (p3[0] ** 2) - (p1[0] ** 2)
        y4 = 2 * (p1[0] - p3[0])
        yc = (y1 - y2 - y3) / y4
        return [xc, yc]

    def __get_circ(self, c, r_max=20):  # c is dictionary- the output of "regionprops" for one label
        rows, cols = self.img.shape
        mask = np.zeros((rows, cols), dtype=int)
        pixList = np.array(c['pixelList'])
        ran = pixList.shape
        for i in range(ran[1]):
            mask[pixList[0, i], pixList[1, i]] = 1.0

        if c['majorAxisLength'] < r_max and c['minorAxisLength'] < r_max:
            r = np.amax([c['majorAxisLength'], c['minorAxisLength']])
            circ = np.array([c['centroid'][0], c['centroid'][1], r])
            return circ
        circXmin = int(np.amin([c['pixelList'][1]]) + 0.5 * r_max)
        circYmin = int(np.amin([c['pixelList'][0]]) + 0.5 * r_max)
        circXmax = np.amax([c['pixelList'][1]])
        circYmax = np.amax([c['pixelList'][0]])

        circ = np.array([])
        distances = np.array([])

        for circY in range(circYmin - 1, circYmax):
            for circX in range(circXmin - 1, circXmax):
                if mask[circY][circX]:
                    circShape = circ.shape
                    if circShape[0] == 0:
                        r = np.amin([r_max, circXmax - circX])
                        circ = np.array([circX, circY, r])
                    else:
                        m = circ.shape
                        if len(m) == 1:
                            distances = ((circ[0] - circX) ** 2) + ((circ[1] - circY) ** 2)
                        else:
                            for i in range(m[0]):
                                d = ((circ[i, 0] - circX) ** 2) + ((circ[i, 1] - circY) ** 2)
                                distances = np.append(distances, d)
                        if np.amin(distances) >= r_max:
                            r = np.amin([r_max, circXmax - circX])
                            circ = np.vstack([circ, [circX, circY, r]])
                        distances = np.array([])
        return circ

    def __target_rating(self, c, xc, yc):
        distanceFromFLS = np.sqrt(((c['centroid'][0] - xc) ** 2) + ((c['centroid'][1] - yc) ** 2))

        majorAxisEst = distanceFromFLS / 9
        minorAxisEst = distanceFromFLS / 20

        majorAxisRate = ((c['majorAxisLength'] / majorAxisEst) ** 2) - 1  # [-1,1]
        minorAxisRate = ((c['minorAxisLength'] / minorAxisEst) ** 2) - 1  # [-1,1]

        if majorAxisRate > 1:
            majorAxisRate = ((1 / majorAxisRate) * 2) - 1
        if minorAxisRate > 1:
            minorAxisRate = ((1 / minorAxisRate) * 2) - 1

        b = yc / (xc - 1)
        a = -b / xc
        xm = -b / (2 * a)
        distanceNorm = a * (xm ** 2) - b * xm
        distanceRate = (a * (distanceFromFLS ** 2) - b * distanceFromFLS) / distanceNorm

        targetRate = (0.33 * majorAxisRate) + (0.33 * minorAxisRate) + (0.33 * distanceRate)
        return targetRate

    def apply(self):
        rois_map = ROIfind.__create_rois_map(self)

        mask = rois_map * self.img
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        ret, mask = cv2.threshold(mask, 0.7 * 255, 255, cv2.THRESH_BINARY)
        xc, yc = self.__find_center()

        centroids = np.array([1, 2])
        bboxes = np.array([1, 2, 3, 4])
        targetRate = np.array([1])
        circles = []
        index = 0
        c = {}

        # making "regionprops" and apply
        mask = np.uint8(mask)

        # in opencv version up to 2.4
        #im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #in opencv version from 2.4
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                c['pixelList'] = np.nonzero(maskToPix)
                c['centroid'] = [cx, cy]
                (x, y), (c['majorAxisLength'], c['minorAxisLength']), angle = cv2.fitEllipse(contours[cnt])
                circle = self.__get_circ(c)
                circles.append(circle)
                tr = self.__target_rating(c, xc, yc)
                targetRate = np.append(targetRate, [tr], 0)
                index = index + 1
            else:
                x, y, w, h = cv2.boundingRect(contours[cnt])
                mask[x:x + w + 1][y:y + h + 1] = 0
        centroids = np.delete(centroids, 0, 0)
        bboxes = np.delete(bboxes, 0, 0)
        targetRate = np.delete(targetRate, 0)
        return mask, centroids, bboxes, circles, targetRate


'''          
img=cv2.imread('Try134.jpg',0)
rois=ROIfind(img)
mask, centroids, bboxes, circles, targetRate=rois.apply()
'''
