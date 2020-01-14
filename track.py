# -*- coding: utf-8 -*-

import numpy as np
from random import random
import cv2
# import matplotlib.pyplot as plt


class track:
    def __init__(self, idCount, bbox, centroid, tiny_mask):
        self.state = np.array([[centroid[0], 0, centroid[1], 0]])
        self.bbox = bbox
        self.age = 1
        self.visibleCount = 1
        self.invisibleCount = 0
        self.id = idCount
        self.color = (random(), random(), random())
        self.tiny_mask = tiny_mask

    def updateTrack(self, visible, bbox=0, centroid=0, tiny_mask=0):
        if visible:
            self.visibleCount += 1
            self.invisibleCount = 0
            self.bbox = bbox
            latest_state = self.state.shape
            vx = centroid[0] - self.state[latest_state[0] - 1][0]
            vy = centroid[1] - self.state[latest_state[0] - 1][3]
            new_state = [centroid[0], vx, centroid[1], vy]
            self.state = np.vstack((self.state, new_state))
            self.age += 1
            self.tiny_mask = tiny_mask
        else:
            self.invisibleCount += 1
            self.age += 1

    def get_circles(self, r_max=20.0):

        ys, xs = self.tiny_mask.shape
        tiny_gray = self.tiny_mask
        tiny = np.zeros((ys, xs, 3))
        tiny[:, :, 0] = tiny_gray
        tiny[:, :, 1] = tiny_gray
        tiny[:, :, 2] = tiny_gray
        circ = np.array([])

        radius = r_max * 0.75

        x, y = np.meshgrid(np.arange(xs), np.arange(ys), sparse=True)

        # fig, ax = plt.subplots(2, 1)
        # ax[0].imshow(self.tiny_mask)

        for i in range(int(-0.5 * ys / radius), int(ys / radius) + 1):
            for j in range(int(xs / radius) + 1):
                x0 = 2 * radius * (i + 0.5 * j)
                y0 = 2 * radius * np.sqrt(3) / 2 * j

                r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

                circ_indicator = r < radius
                img_indicator = tiny_gray > 0

                if np.any(tiny[circ_indicator, 0] != 0):

                    mask = np.zeros((ys, xs))
                    mask[circ_indicator & img_indicator] = 1
                    # ax[1].imshow(mask)
                    # plt.show()
                    im, contours, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cnt.size >= 10:
                            (x1, y1), (ma, MA), _ = cv2.fitEllipse(cnt)

                    if circ.size:
                        circ = np.vstack([circ, [x1 + self.bbox[0], y1 + self.bbox[1], int(MA / 2)]])
                    else:
                        circ = np.array([[x1 + self.bbox[0], y1 + self.bbox[1], int(MA / 2)]])

        # check if there are circles which contained by another one:
        contained = []
        for c1 in circ:
            ind = 0
            for c2 in circ:
                distance = np.sqrt(((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2))
                if (c2[2] + distance) < c1[2]:
                    contained.append(ind)
                ind += 1
        contained = np.array(contained)
        circ = np.delete(circ, contained, 0)

        return circ
