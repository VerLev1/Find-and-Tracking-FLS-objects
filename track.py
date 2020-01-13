# -*- coding: utf-8 -*-

import numpy as np
from random import random


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
