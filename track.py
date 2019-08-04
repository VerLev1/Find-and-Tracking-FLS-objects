# -*- coding: utf-8 -*-

import numpy as np
from filterpy.kalman import KalmanFilter
from random import randint


class track:
    def __init__(self, circles, idCount, bbox, centroid, targetRate=0):
        self.state = np.array([[centroid[0], 0, centroid[1], 0]])
        self.bbox = bbox
        self.circles = [circles]
        self.kalman = self.createKalmanTracker()
        self.age = 1
        self.visibleCount = 1
        self.invisibleCount = 0
        self.targetRate = targetRate
        self.id = idCount
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))

    def createKalmanTracker(self):
        dt = 0.03

        kf = KalmanFilter(dim_x=4, dim_z=2)
        state = np.array(self.state)
        kf.x = np.reshape(state, (4, 1))
        kf.F = np.array([[1, dt, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, dt],
                         [0, 0, 0, 1]])  # transition_matrix

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])  # measurement function

        return kf

    def updateTrack(self, visible, circles=[0, 0, 0], targetRate=0, bbox=0, centroid=0):
        if visible:
            self.visibleCount += 1
            self.invisibleCount = 0
            self.circles.append(circles)
            self.targetRate = targetRate
            self.bbox = bbox
            latest_state = self.state.shape
            vx = centroid[0] - self.state[latest_state[0] - 1][0]
            vy = centroid[1] - self.state[latest_state[0] - 1][3]
            new_state = [centroid[0], vx, centroid[1], vy]
            self.state = np.vstack((self.state, new_state))
            self.kalman.update(centroid)
        else:
            self.invisibleCount += 1
            self.age += 1
